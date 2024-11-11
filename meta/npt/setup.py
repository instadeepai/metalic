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

from setuptools import setup

with open("README.md") as f:
    readme = f.read()

setup(
    name="proteinnpt",
    description="ProteinNPT: Improving Protein Property Prediction and Design with Non-Parametric Transformers",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Pascal Notin and Ruben Weitzman",
    version="1.0",
    license="MIT",
    url="https://github.com/OATML-Markslab/ProteinNPT",
    packages=["proteinnpt"],  # modified for cleaner namespace
)
