# Copyright 2023 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from transformers import GPT2Config


class TranceptionConfig(GPT2Config):
    """
    Config subclass for Tranception model architecture.
    """

    def __init__(
        self,
        attention_mode="tranception",
        position_embedding="grouped_alibi",
        tokenizer=None,
        retrieval_aggregation_mode=None,
        retrieval_inference_weight=0.6,
        MSA_filename=None,
        MSA_weight_file_name=None,
        MSA_start=None,
        MSA_end=None,
        full_protein_length=None,
        clustal_omega_location=None,
        scoring_window="optimal",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_type = "tranception"
        self.attention_mode = attention_mode
        self.position_embedding = position_embedding
        self.tokenizer = tokenizer
        self.retrieval_aggregation_mode = retrieval_aggregation_mode
        self.retrieval_inference_weight = retrieval_inference_weight
        self.MSA_filename = MSA_filename
        self.MSA_weight_file_name = MSA_weight_file_name
        self.MSA_start = MSA_start
        self.MSA_end = MSA_end
        self.full_protein_length = full_protein_length
        self.clustal_omega_location = clustal_omega_location
        self.scoring_window = scoring_window
