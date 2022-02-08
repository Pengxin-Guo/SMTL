# for POS task
python train.py --gpu_id 6 --model STL --dataset udpos --lang en  > out/stl_pos_en.out

python train.py --gpu_id 3 --model STL --dataset udpos --lang zh  > out/stl_pos_zh.out

python train.py --gpu_id 6 --model STL --dataset udpos --lang te  > out/stl_pos_te.out

python train.py --gpu_id 0 --model STL --dataset udpos --lang vi  > out/stl_pos_vi.out

python train.py --gpu_id 1 --model DMTL --dataset udpos > out/dmtl_pos_all.out

python train.py --gpu_id 1 --model SMTL --version v1 --dataset udpos > out/smtl_v1_pos_all.out

python train.py --gpu_id 2 --model SMTL --version v3 --dataset udpos > out/smtl_v3_pos_all.out

python train.py --gpu_id 3 --model SMTL_new --version v1 --dataset udpos > out/smtl_new_v1_pos_all.out

python train.py --gpu_id 1 --model SMTL_new --version v3 --dataset udpos > out/smtl_new_v3_pos_all.out

# for PI task
python train.py --gpu_id 3 --model STL --dataset pawsx --lang en  > out/stl_pawsx_en.out

python train.py --gpu_id 7 --model STL --dataset pawsx --lang zh  > out/stl_pawsx_zh.out

python train.py --gpu_id 1 --model STL --dataset pawsx --lang de  > out/stl_pawsx_de.out

python train.py --gpu_id 6 --model STL --dataset pawsx --lang es  > out/stl_pawsx_es.out

python train.py --gpu_id 0 --model DMTL --dataset pawsx > out/dmtl_pawsx_all.out

python train.py --gpu_id 2 --model SMTL --version v1 --dataset pawsx > out/smtl_v1_pawsx_all.out

python train.py --gpu_id 1 --model SMTL --version v3 --dataset pawsx > out/smtl_v3_pawsx_all.out

python train.py --gpu_id 3 --model SMTL_new --version v1 --dataset pawsx > out/smtl_new_v1_pawsx_all.out

python train.py --gpu_id 3 --model SMTL_new --version v3 --dataset pawsx > out/smtl_new_v3_pawsx_all.out

# for NER task
python train.py --gpu_id 0 --model STL --dataset panx --lang en  > out/stl_panx_en.out

python train.py --gpu_id 7 --model STL --dataset panx --lang zh  > out/stl_panx_zh.out

python train.py --gpu_id 2 --model STL --dataset panx --lang de  > out/stl_panx_de.out

python train.py --gpu_id 5 --model STL --dataset panx --lang es  > out/stl_panx_es.out

python train.py --gpu_id 1 --model DMTL --dataset panx > out/dmtl_panx_all.out

python train.py --gpu_id 7 --model SMTL --version v1 --dataset panx > out/smtl_v1_panx_all.out

python train.py --gpu_id 6 --model SMTL --version v3 --dataset panx > out/smtl_v3_panx_all.out

python train.py --gpu_id 1 --model SMTL_new --version v1 --dataset panx > out/smtl_new_v1_panx_all.out

python train.py --gpu_id 1 --model SMTL_new --version v3 --dataset panx > out/smtl_new_v3_panx_all.out

# for NLI task
python train.py --gpu_id 0 --model STL --dataset xnli --lang en  > out/stl_xnli_en.out

python train.py --gpu_id 6 --model STL --dataset xnli --lang zh  > out/stl_xnli_zh.out

python train.py --gpu_id 4 --model STL --dataset xnli --lang de  > out/stl_xnli_de.out

python train.py --gpu_id 7 --model STL --dataset xnli --lang es  > out/stl_xnli_es.out

python train.py --gpu_id 7 --model DMTL --dataset xnli > out/dmtl_xnli_all.out

python train.py --gpu_id 2 --model SMTL --version v1 --dataset xnli > out/smtl_v1_xnli_all.out

python train.py --gpu_id 1 --model SMTL --version v3 --dataset xnli > out/smtl_v3_xnli_all.out

python train.py --gpu_id 7 --model SMTL_new --version v1 --dataset xnli > out/smtl_new_v1_xnli_all.out

python train.py --gpu_id 7 --model SMTL_new --version v3 --dataset xnli > out/smtl_new_v3_xnli_all.out