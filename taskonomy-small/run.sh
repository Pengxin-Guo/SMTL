python -u  train.py --task_index 0 --gpu_id 1 > out/single_seg.out

python -u  train.py --task_index 1 --gpu_id 1 > out/single_depth.out

python -u  train.py --task_index 2 --gpu_id 1 > out/single_sn.out

python -u  train.py --task_index 3 --gpu_id 1 > out/single_keypoint.out

python -u  train.py --task_index 4 --gpu_id 1 > out/single_edge.out

python -u  train.py --model DMTL --gpu_id 1 > out/dmtl.out

python -u  train.py --model CROSS --gpu_id 1 > out/cross.out

python -u  train.py --model MTAN --gpu_id 1 > out/mtan.out

python -u  train.py --model AdaShare --gpu_id 1 > out/adashare.out

python -u  train.py --model NDDRCNN --gpu_id 1 > out/nddrcnn.out

python -u  train.py --model AFA --gpu_id 1 > out/afa.out

python -u  train.py --model AMTL --version v1 --gpu_id 1 > out/amtl_v1.out

python -u  train.py --model AMTL --version v2 --gpu_id 1 > out/amtl_v2.out

python -u  train.py --model AMTL --version v3 --gpu_id 1 > out/amtl_v3.out

python -u  train.py --model AMTL_new --version v1 --gpu_id 1 > out/amtl_new_v1.out

python -u  train.py --model AMTL_new --version v2 --gpu_id 1 > out/amtl_new_v2.out

python -u  train.py --model AMTL_new --version v3 --gpu_id 1 > out/amtl_new_v3.out