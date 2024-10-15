import gc
import logging
import os
import time

import hydra
import torch
from neptune.integrations.python_logger import NeptuneHandler
from omegaconf import DictConfig, open_dict

from meta.evaluation import run_metasupervised_evaluation
from meta.tools.logger import get_logger_from_config
from meta.utils import (
    display_info,
    get_current_git_commit_hash,
    set_seeds,
)

logging.basicConfig(level="NOTSET", format="%(message)s", datefmt="[%X]")
log = logging.getLogger("rich")


@hydra.main(
    config_path="config", config_name="metasupervised", version_base="1.2"
)  # n.b. config_name can be overridden by config-name command line flag
def run_metasupervised(cfg: DictConfig) -> None:  # noqa: CCR001
    display_info(cfg)
    set_seeds(cfg.seed)
    task = hydra.utils.instantiate(cfg.task)
    log.info("Setting up task (splitting datasets etc.)")
    # load 0-shot data if using auxillary zero-shot predictions
    load_zero_shot =  cfg.surrogate.model_config.aux_pred
    task.setup_datasets(load_zero_shot=load_zero_shot)
    commit_hash = get_current_git_commit_hash()
    log.info(f"Commit hash: {commit_hash}")
    with open_dict(cfg):
        cfg.commit_hash = commit_hash

    neptune_tags = [
        str(cfg.task.task_name),
        str(cfg.surrogate.name),
        str(type(task)),  # task.task_type,
        commit_hash,
    ]
    logger = get_logger_from_config(cfg, file_system=None, neptune_tags=neptune_tags)
    if cfg.logging.type == "neptune":
        log.addHandler(NeptuneHandler(run=logger.run))  # type: ignore
    log.info("Instantiating surrogate")
    surrogate = hydra.utils.instantiate(cfg.surrogate, _recursive_=True)
    run_name = "".join(cfg.logging.tags)
    surrogate.set_dir(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "results/",
            run_name,
            str(cfg.seed),
        )
    )
    if task.has_metadata:
        log.info("Providing surrogate with task metadata")
        surrogate.set_metadata(task.metadata)

    # val_data should be determiend automatically in the evaluation function,
    # but to get it directly:
    # val_data = task.data_splits.get("validation", None)

    if cfg.evaluate_first:
        log.info("Evaluating surrogate at initialisation")

        init_metrics, _ = run_metasupervised_evaluation(task, surrogate)
        log.info(
            "Init metrics: {}".format(
                "\t".join(f"{k}: {v:.3f}" for k, v in init_metrics.items())
            )
        )
        logger.write(init_metrics, label="init_metrics", timestep=0)

    if cfg.exit_after_first_eval:
        assert cfg.evaluate_first, "Must evaluate first to exit after first eval."
        del surrogate
        del task
        torch.cuda.empty_cache()
        gc.collect()
        log.info("Run metasupervised exiting after first evaluation.")
        exit()

    log.info(
        "Dataset sizes: "
        + " ".join(
            [
                f"{split}: {str(len(task.data_splits[split]))}"
                for split in ["train", "validation"]
                if split in task.data_splits
            ]
        ),
    )

    log.info("Training surrogate")
    t0 = time.time()
    eval_func = lambda surrogate_in: run_metasupervised_evaluation(
        task, surrogate_in  # noqa: F821
    )
    surrogate.fit(
        task.data_splits["train"], cfg.seed, logger=logger, eval_func=eval_func
    )

    if cfg.evaluate_end:
        t1 = time.time()
        log.info("Evaluating surrogate at end")
        train_end_metrics, _ = run_metasupervised_evaluation(
            task, surrogate
        )
        train_end_metrics.update(surrogate.get_training_summary_metrics())
        train_end_metrics.update(task.data_summary())
        train_end_metrics.update({"train_time": t1 - t0})
        log.info(
            "Metrics: {}".format(
                "\t".join(f"{k}: {v:.4f}" for k, v in train_end_metrics.items())
            )
        )
        logger.write(train_end_metrics, label="end_metrics")

    logger.close()
    surrogate.cleanup()

    del surrogate
    del task
    torch.cuda.empty_cache()
    gc.collect()

    log.info("Run metasupervised completed successfully")


if __name__ == "__main__":
    run_metasupervised()
