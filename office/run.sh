# office-31
python -u train.py --dataset office-31 --model DMTL --gpu_id 3 --task_index 0 > out/office31_single_task1.out

python -u train.py --dataset office-31 --model DMTL --gpu_id 3 --task_index 1 > out/office31_single_task2.out

python -u train.py --dataset office-31 --model DMTL --gpu_id 3 --task_index 2 > out/office31_single_task3.out

python -u train.py --dataset office-31 --model DMTL --gpu_id 3 > out/office31_dmtl.out

python -u train.py --dataset office-31 --model MTAN --gpu_id 3 > out/office31_mtan.out

python -u train.py --dataset office-31 --model AMTL --version v1 --gpu_id 3 > out/office31_amtl_v1.out

python -u train.py --dataset office-31 --model AMTL --version v2 --gpu_id 3 > out/office31_amtl_v2.out

python -u train.py --dataset office-31 --model AMTL --version v3 --gpu_id 3 > out/office31_amtl_v3.out

python -u train.py --dataset office-31 --model AMTL_new --version v1 --gpu_id 3 > out/office31_AMTL_new_v1.out

python -u train.py --dataset office-31 --model AMTL_new --version v2 --gpu_id 3 > out/office31_AMTL_new_v2.out

python -u train.py --dataset office-31 --model AMTL_new --version v3 --gpu_id 3 > out/office31_AMTL_new_v3.out

# office-home
python -u train.py --dataset office-home --model DMTL --gpu_id 3 --task_index 0 > out/officehome_single_task1.out

python -u train.py --dataset office-home --model DMTL --gpu_id 3 --task_index 1 > out/officehome_single_task2.out

python -u train.py --dataset office-home --model DMTL --gpu_id 3 --task_index 2 > out/officehome_single_task3.out

python -u train.py --dataset office-home --model DMTL --gpu_id 3 --task_index 3 > out/officehome_single_task4.out

python -u train.py --dataset office-home --model DMTL --gpu_id 3 > out/officehome_dmtl.out

python -u train.py --dataset office-home --model MTAN --gpu_id 3 > out/officehome_mtan.out

python -u train.py --dataset office-home --model AMTL --version v1 --gpu_id 3 > out/officehome_amtl_v1.out

python -u train.py --dataset office-home --model AMTL --version v2 --gpu_id 3 > out/officehome_amtl_v2.out

python -u train.py --dataset office-home --model AMTL --version v3 --gpu_id 3 > out/officehome_amtl_v3.out

python -u train.py --dataset office-home --model AMTL_new --version v1 --gpu_id 3 > out/officehome_AMTL_new_v1.out

python -u train.py --dataset office-home --model AMTL_new --version v2 --gpu_id 3 > out/officehome_AMTL_new_v2.out

python -u train.py --dataset office-home --model AMTL_new --version v3 --gpu_id 3 > out/officehome_AMTL_new_v3.out
