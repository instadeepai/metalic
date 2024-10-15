import functools
from typing import Callable

import numpy as np
import torch
from torch.nn import functional as F  # noqa: N812


def mse(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """MSE loss on a batch."""
    assert (
        preds.shape == targets.shape
    ), f"preds shape {preds.shape} does not match targets shape {targets.shape}"
    mse = ((preds - targets) ** 2).mean(0)
    return mse


def full_ranking_bce(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Formulate a pairwise classification problem between all items in batch, and compute bce.

    For each pair of items in the batch, we aim to predict which member of the
    pair has the higher target value. We derive logits from pairs of predictions as
    follows:

    Given input tensors of shape b, we form a bxb preference matrix with entries
    Pij = preds_i - preds_j. We then treat these latent preferences as the logits
    of a binary classifier, aiming to predict y_ij = I(targets_i > targets_j).

    This is equivalent to the contrastive loss from direct preference optimisation -
    the KL constraint can be added by appropriate definition of preds.

    Args:
        preds: Tensor of shape (b,)
        targets: Tensor of shape (b,)
    """
    pairwise_logits = preds[:, None] - preds[None, :]  # b, b
    targets = targets[:, None] > targets[None, :]
    # commented out: a smoothed version...see cdpo
    # targets = torch.sigmoid((scores[:, None] - scores[None, :]) / 0.041)
    ranking_xent = 0.5 * F.binary_cross_entropy_with_logits(
        pairwise_logits, targets.float(), reduction="none"
    )
    diag_mask = 1 - torch.eye(pairwise_logits.shape[0], device=pairwise_logits.device)
    ranking_xent = (ranking_xent * diag_mask).mean((-1, -2))
    return ranking_xent


def label_smoothed_ranking_loss(
    ranking_loss: Callable,
    preds: torch.Tensor,
    targets: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Label smoothing is used to define a conservative DPO loss as described in
    https://ericmitchell.ai/cdpo.pdf.

    The value of label_smoothing should be interpreted as the probability that
    any given pair of labels has been flipped.

    We can invert the logits or the targets to calculate the loss in
    the case where the labels are flipped.

    TODO: check this flipping makes sense
    """
    # https://github.com/eric-mitchell/direct-preference-optimization/blob/f8b8c0f49dc92a430bae41585f9d467d3618fe2f/trainers.py#L82
    loss = ranking_loss(preds, targets)
    flipped_loss = ranking_loss(-preds, targets)
    return (1 - label_smoothing) * loss + label_smoothing * flipped_loss


def adaptive_label_smoothed_full_ranking_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    label_smoothing_beta: float = 1.0,
) -> torch.Tensor:
    """We allow the label smoothing parameter itself
    to be a function of the magnitude of difference in labels.

    label_smoothing_beta: multiplies difference in targets followed by sigmoid
        to produce per-comparison label smoothing parameter,

    to enable a fixed scale for this, it'd probably be helpful
    to standardise inputs.
    """
    # https://github.com/eric-mitchell/direct-preference-optimization/blob/f8b8c0f49dc92a430bae41585f9d467d3618fe2f/trainers.py#L82
    pairwise_logits = preds[:, None] - preds[None, :]  # b, b
    target_diffs = targets[:, None] - targets[None, :]
    targets = target_diffs > 0
    ranking_xent = 0.5 * F.binary_cross_entropy_with_logits(
        pairwise_logits, targets.float(), reduction="none"
    )
    diag_mask = 1 - torch.eye(pairwise_logits.shape[0], device=pairwise_logits.device)
    ranking_xent = ranking_xent * diag_mask

    flipped_ranking_xent = 0.5 * F.binary_cross_entropy_with_logits(
        pairwise_logits, (~targets).float(), reduction="none"
    )
    flipped_ranking_xent = flipped_ranking_xent * diag_mask

    label_smoothing = torch.sigmoid(-target_diffs.abs() * label_smoothing_beta)

    # TODO print label smoothing or return as a metric somehow
    ranking_xent = (
        1 - label_smoothing
    ) * ranking_xent + label_smoothing * flipped_ranking_xent
    return ranking_xent.mean((-1, -2))


def derangement_vectorized(n: int) -> np.ndarray:
    """Sample random permutations until we find one in which nothing is mapped to itself."""
    assert n > 1
    original = np.arange(n)
    while True:
        shuffled = np.random.permutation(n)
        if np.all(original != shuffled):
            return shuffled


def perm_ranking_bce(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Formulate a classification task by pairing each item in a batch with one other,
    and compute bce.

    We first sample a `derangement` (pairing) of items in the batch.
    Then for each pair of items we use the difference in predictions as the
    logits of a binary classifier aiming to predict which item has the
    higher target value.

    This function therefore computes a pairwise classification loss over a subset of the
    complete pairwise comparisons in `full_ranking_bce`.

    This is the loss used in `Don't throw away that linear head` (Krause et al.)

    Args:
        preds: Tensor of shape (b,)
        targets: Tensor of shape (b,)
    """
    perm = derangement_vectorized(preds.shape[0])
    pairwise_logits = preds - preds[perm]  # type: ignore
    targets = targets > targets[perm]  # type: ignore
    ranking_xent = F.binary_cross_entropy_with_logits(
        pairwise_logits, targets.float(), reduction="mean"
    )
    return ranking_xent


def get_loss(
    loss_name: str, label_smoothing: float = 0.0, label_smoothing_beta: float = 1.0
) -> Callable:
    if loss_name in ["mse", "mse_head"]:
        assert label_smoothing == 0.0, "label_smoothing not compatible with mse"
        return mse
    elif loss_name == "ranking":
        loss_fn = perm_ranking_bce
        if label_smoothing > 0:
            loss_fn = functools.partial(
                label_smoothed_ranking_loss, loss_fn, label_smoothing=label_smoothing
            )
        return loss_fn
    elif loss_name == "ranking_full":
        loss_fn = full_ranking_bce
        if label_smoothing > 0:
            loss_fn = functools.partial(
                label_smoothed_ranking_loss, loss_fn, label_smoothing=label_smoothing
            )
        return loss_fn
    elif loss_name == "adaptively_smoothed_ranking_full":
        loss_fn = functools.partial(
            adaptive_label_smoothed_full_ranking_loss,
            label_smoothing_beta=label_smoothing_beta,
        )
        return loss_fn
    else:
        raise ValueError(f"Unsupported loss_name {loss_name}")
