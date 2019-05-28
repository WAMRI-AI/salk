
python gen_sample_info.py --only mitotracer --out live_mito.csv data/live
python tile_from_info.py --out datasets/multimito_001 --info live_mito.csv --n_train 9000 --n_valid 1000 --n_frames 5 --lr_type t --tile 1024 --only mitotracker
python tile_from_info.py --out datasets/multimito_001 --info live_mito.csv --n_train 9000 --n_valid 1000 --tile 1024 --only mitotracker
python -m fastai.launch  train.py --bs 8 --lr 3e-4 --size 256 --tile_sz 1024 --datasetname multimito_001 --cycles 40 --save_name multit_5_unet --lr_type t --n_frames 5
python -m fastai.launch  train.py --bs 8 --lr 1e-4 --size 512 --tile_sz 1024 --datasetname multimito_001 --cycles 20 --save_name multit_5_unet --lr_type t --n_frames 5 --load_name multit_5_unet_best_256
python -m fastai.launch  train.py --bs 1 --lr 1e-4 --size 1024 --tile_sz 1024 --datasetname multimito_001 --cycles 20 --save_name multit_5_unet --lr_type t --n_frames 5 --load_name multit_5_unet_best_512
python -m image_gen.py /scratch/bpho/datasets/movies_002/test tmp/newmito3 --modles multit_5_unet_512 --gpu 1
echo "done"
