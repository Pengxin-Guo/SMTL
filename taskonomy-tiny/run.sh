CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.launch --nproc_per_node=4 train.py --task_index 0 > out/single_seg.out

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.launch --nproc_per_node=4 train.py --task_index 1 > out/single_depth.out

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.launch --nproc_per_node=4 train.py --task_index 2 > out/single_sn.out

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.launch --nproc_per_node=4 train.py --task_index 3 > out/single_keypoint.out

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.launch --nproc_per_node=4 train.py --task_index 4 > out/single_edge.out

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.launch --nproc_per_node=4 train.py --model DMTL > out/dmtl.out

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.launch --nproc_per_node=4 train.py --model CROSS > out/cross.out

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.launch --nproc_per_node=4 train.py --model MTAN > out/mtan.out

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.launch --nproc_per_node=4 train.py --model AdaShare > out/adashare.out

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.launch --nproc_per_node=4 train.py --model NDDRCNN > out/nddrcnn.out

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.launch --nproc_per_node=4 train.py --model SMTL --version v1 > out/smtl_v1.out

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.launch --nproc_per_node=4 train.py --model SMTL --version v2 > out/smtl_v2.out

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.launch --nproc_per_node=4 train.py --model SMTL --version v3 > out/smtl_v3.out

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.launch --nproc_per_node=4 train.py --model SMTL_new --version v1 > out/smtl_new_v1.out

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.launch --nproc_per_node=4 train.py --model SMTL_new --version v2 > out/smtl_new_v2.out

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.launch --nproc_per_node=4 train.py --model SMTL_new --version v3 > out/smtl_new_v3.out