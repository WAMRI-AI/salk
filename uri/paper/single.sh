# python -m fastai.launch  train.py --bs 8 --lr 3e-4 --size 256 --tile_sz 1024 --datasetname singlemito_001 --cycles 40 --save_name single_unet
# python image_gen.py /scratch/bpho/datasets/movies_002/test tmp/single256 --models single_unet_256 --gpu 1
# python -m fastai.launch  train.py --bs 8 --lr 1e-4 --size 512 --tile_sz 1024 --datasetname singlemito_001 --cycles 20 --save_name single_unet --load_name single_unet_best_256
# python image_gen.py /scratch/bpho/datasets/movies_002/test tmp/single512 --models single_unet_512 --gpu 1
python -m fastai.launch  train.py --bs 1 --lr 1e-4 --size 1024 --tile_sz 1024 --datasetname singlemito_001 --cycles 20 --save_name single_unet --load_name single_unet_best_512
python image_gen.py /scratch/bpho/datasets/movies_002/test tmp/single1024 --models single_unet_1024 --gpu 1
echo "done"
