CUDA_VISIBLE_DEVICES=1,2 \
torchrun --nproc_per_node=2 train.py -c configs/lane.yaml

#   NCCL_P2P_DISABLE=1 \