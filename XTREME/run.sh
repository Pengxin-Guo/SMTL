python train.py --gpu_id 6 --model STL --dataset udpos --lang en  > out/stl_pos_en.out

python train.py --gpu_id 3 --model STL --dataset udpos --lang zh  > out/stl_pos_zh.out

python train.py --gpu_id 6 --model STL --dataset udpos --lang te  > out/stl_pos_te.out

python train.py --gpu_id 0 --model STL --dataset udpos --lang vi  > out/stl_pos_vi.out

python train.py --gpu_id 1 --model DMTL --dataset udpos > out/dmtl_pos_all.out