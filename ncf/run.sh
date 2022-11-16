#!/bin/bash

# Set the variables below
export DATASET="ml-20m"
export DATA_DIR="./cache/"${DATASET} # Path to test and training data files
export method="tcp://"
export hostip="${hostip:-localhost}"
export port=${port:-23456}
export SHARED_PATH=$method$hostip":"$port #a shared path visible to all processes for rendezvous
export RANK="$SLURM_PROCID" #rank of this process
export USE_WANDB=0
export WANDB_API_KEY=0 #wandb api key
export WANDB_ENTITY=0 #wandb entity

world_size=2

############### threshold=1/(2*(31832577*density)^(1/2)) ###########
############### density=0.095 -> threshold=0.0002875     ###########

########################## Exact ###################################################
# reducer='exact'
# python ncf.py --mode train --data=${DATA_DIR} --backend=nccl --shared_path=${SHARED_PATH} --reducer=$reducer --seed=1 --rank=$RANK --world_size=${world_size} --use_wandb=${USE_WANDB}
####################################################################################

########################### Top-k ###################################################
# reducer='topk'
# python ncf.py --mode train --data=${DATA_DIR} --backend=nccl --shared_path=${SHARED_PATH} --reducer=$reducer --seed=1 --comp_ratio=0.095 --rank=$RANK --world_size=${world_size} --use_wandb=${USE_WANDB}
#######################################################################################

########################### SAGE ################################################
 reducer='sage'
 python ncf.py --mode train --data=${DATA_DIR} --backend=nccl --shared_path=${SHARED_PATH} --reducer=$reducer --seed=1 --thresh=0.0002875 --comp_ratio=0.095 --rank=$RANK --world_size=${world_size} --use_wandb=${USE_WANDB}
########################################################################################

########################### Threshold ################################################
# reducer='thresh'
# python ncf.py --mode train --data=${DATA_DIR} --backend=nccl --shared_path=${SHARED_PATH} --reducer=$reducer --seed=1 --thresh=0.0002875 --rank=$RANK --world_size=${world_size} --use_wandb=${USE_WANDB}
########################################################################################
