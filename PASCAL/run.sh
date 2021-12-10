python -u train.py --gpu_id 6 --task_index 0 > out/single_semseg.out

python -u train.py --gpu_id 6 --task_index 1 > out/single_human_parts.out

python -u train.py --gpu_id 6 --task_index 2 > out/single_sal.out

python -u train.py --gpu_id 6 --task_index 3 > out/single_normals.out

python -u train.py --gpu_id 7 --model DMTL > out/dmtl.out

python -u train.py --gpu_id 7 --model CROSS > out/cross.out

python -u train.py --gpu_id 7 --model MTAN > out/mtan.out

python -u train.py --gpu_id 7 --model AdaShare > out/adashare.out

python -u train.py --gpu_id 7 --model NDDRCNN > out/nddrcnn.out

python -u train.py --gpu_id 7 --model AFA > out/afa.out

python -u train.py --gpu_id 7 --model SMTL --version v1 > out/smtl_v1.out

# python -u train.py --gpu_id 7 --model SMTL --version v2 > out/smtl_v2.out

python -u train.py --gpu_id 7 --model SMTL --version v3 > out/smtl_v3.out

python -u train.py --gpu_id 7 --model SMTL_new --version v1 > out/smtl_new_v1.out

# python -u train.py --gpu_id 7 --model SMTL_new --version v2 > out/smtl_new_v2.out

python -u train.py --gpu_id 7 --model SMTL_new --version v3 > out/smtl_new_v3.out