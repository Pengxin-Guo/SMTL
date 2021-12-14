# for POS task
python train.py --gpu_id 6 --model STL --dataset udpos --lang en  > out/stl_pos_en.out

python train.py --gpu_id 3 --model STL --dataset udpos --lang zh  > out/stl_pos_zh.out

python train.py --gpu_id 6 --model STL --dataset udpos --lang te  > out/stl_pos_te.out

python train.py --gpu_id 0 --model STL --dataset udpos --lang vi  > out/stl_pos_vi.out

python train.py --gpu_id 1 --model DMTL --dataset udpos > out/dmtl_pos_all.out

python train.py --gpu_id 7 --model SMTL --version v1 --dataset udpos > out/smtl_v1_pos_all.out

# for PI task
python train.py --gpu_id 3 --model STL --dataset pawsx --lang en  > out/stl_pawsx_en.out

python train.py --gpu_id 7 --model STL --dataset pawsx --lang zh  > out/stl_pawsx_zh.out

python train.py --gpu_id 1 --model STL --dataset pawsx --lang de  > out/stl_pawsx_de.out

python train.py --gpu_id 6 --model STL --dataset pawsx --lang es  > out/stl_pawsx_es.out

python train.py --gpu_id 0 --model DMTL --dataset pawsx > out/dmtl_pawsx_all.out

# for NER task
python train.py --gpu_id 0 --model STL --dataset panx --lang en  > out/stl_panx_en.out

python train.py --gpu_id 7 --model STL --dataset panx --lang zh  > out/stl_panx_zh.out

python train.py --gpu_id 2 --model STL --dataset panx --lang de  > out/stl_panx_de.out

python train.py --gpu_id 5 --model STL --dataset panx --lang es  > out/stl_panx_es.out

python train.py --gpu_id 1 --model DMTL --dataset panx > out/dmtl_panx_all.out