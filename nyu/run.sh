python -u train.py --gpu_id 6 --model DMTL > out/dmtl.out

python -u train.py --gpu_id 6 --model CROSS > out/cross.out

python -u train.py --gpu_id 6 --model MTAN > out/mtan.out

python -u train.py --gpu_id 6 --model AdaShare > out/adashare.out

python -u train.py --gpu_id 6 --model NDDRCNN > out/nddrcnn.out

python -u train.py --gpu_id 6 --model AFA > out/afa.out

python -u train.py --gpu_id 6 --model SMTL --version v1 > out/smtl_v1.out

python -u train.py --gpu_id 6 --model SMTL --version v3 > out/smtl_v3.out

python -u train.py --gpu_id 6 --model SMTL_new --version v1 > out/smtl_new_v1.out

python -u train.py --gpu_id 6 --model SMTL_new --version v3 > out/smtl_new_v3.out