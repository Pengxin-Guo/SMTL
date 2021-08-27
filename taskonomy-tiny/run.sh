CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.launch --nproc_per_node=4 train.py --task_index 0 > out/single_seg.out

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.launch --nproc_per_node=4 train.py --task_index 1 > out/single_depth.out

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.launch --nproc_per_node=4 train.py --task_index 2 > out/single_sn.out

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.launch --nproc_per_node=4 train.py --task_index 3 > out/single_keypoint.out

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.launch --nproc_per_node=4 train.py --task_index 4 > out/single_edge.out

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.launch --nproc_per_node=4 train.py --model DMTL > out/dmtl.out

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.launch --nproc_per_node=4 train.py --model CROSS > out/cross.out

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.launch --nproc_per_node=4 train.py --model MTAN > out/matn.out

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.launch --nproc_per_node=4 train.py --model AdaShare > out/adashare.out

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.launch --nproc_per_node=4 train.py --model NDDRCNN > out/nddrcnn.out

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.launch --nproc_per_node=4 train.py --model AMTL --version v1 > out/amtl_v1.out

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.launch --nproc_per_node=4 train.py --model AMTL --version v2 > out/amtl_v2.out

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.launch --nproc_per_node=4 train.py --model AMTL --version v3 > out/amtl_v3.out

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.launch --nproc_per_node=4 train.py --model AMTL_new --version v1 > out/amtl_new_v1.out

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.launch --nproc_per_node=4 train.py --model AMTL_new --version v2 > out/amtl_new_v2.out

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.launch --nproc_per_node=4 train.py --model AMTL_new --version v3 > out/amtl_new_v3.out