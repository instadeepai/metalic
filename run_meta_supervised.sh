#!/bin/bash
echo $AM_I_IN_A_DOCKER_CONTAINER
if [[ $AM_I_IN_A_DOCKER_CONTAINER ]]
then nvidia-smi
else
# If you are not using a Docker container, you may need to install the following dependencies.
# You can do this by running, from the root of the repository:
pip install -r requirements.txt
fi
pwd

# If using logging.type=neptune at the bottom, then set the NEPTUNE_API_TOKEN environment variable here!
# export NEPTUNE_API_TOKEN= SET_YOUR_API_TOKEN_HERE

export PYTHONHASHSEED=0
export QT_QPA_PLATFORM=xcb # Needed for matplotlib to save figures

# installing proteinnpt.
cd ./meta/npt
pip install -e .
cd ../..

# A name given to the experiment
export tag="V5_Single_0shot"

# Run the experiment, with results saved to ./results/$tag
HYDRA_FULL_ERROR=1 python run_metasupervised.py +experiment/metasupervised=gym logging.tags=[$tag] logging.type=terminal
