python -u  train.py --task_index 0 > out/single_seg.out

python -u  train.py --task_index 1 > out/single_depth.out

python -u  train.py --task_index 2 > out/single_sn.out

python -u  train.py --task_index 3 > out/single_keypoint.out

python -u  train.py --task_index 4 > out/single_edge.out

python -u  train.py --model DMTL > out/dmtl.out

python -u  train.py --model CROSS > out/cross.out

python -u  train.py --model MTAN > out/mtan.out

python -u  train.py --model AdaShare > out/adashare.out

python -u  train.py --model NDDRCNN > out/nddrcnn.out

python -u  train.py --model AMTL --version v1 > out/amtl_v1.out

python -u  train.py --model AMTL --version v2 > out/amtl_v2.out

python -u  train.py --model AMTL --version v3 > out/amtl_v3.out

python -u  train.py --model AMTL_new --version v1 > out/amtl_new_v1.out

python -u  train.py --model AMTL_new --version v2 > out/amtl_new_v2.out

python -u  train.py --model AMTL_new --version v3 > out/amtl_new_v3.out