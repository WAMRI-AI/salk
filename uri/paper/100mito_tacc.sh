#100% mitotracker only
#python gen_sample_info.py --only mitotracker --out live_100mitotracker.csv data/live
#python tile_from_info.py --out datasets/single_100mitotracker --info live_100mitotracker.csv --n_train 5000 --n_valid 200 --lr_type s --tile 512 --only mitotracker --crap_func new_crap
#python tile_from_info.py --out datasets/multi_100mitotracker --info live_100mitotracker.csv --n_train 5000 --n_valid 200 --n_frames 5 --lr_type t --tile 512 --only mitotracker --crap_func new_crap
start=$(date +%s.%N)
python3 -m fastai.launch train.py --bs 24 --lr 4e-4 --size 256 --tile_sz 512 --datasetname single_100mitotracker --cycles 40 --save_name single_100mito --lr_type s --n_frames 1 | tee -a 100mito_trainlog.txt
dur=$(echo "$(date +%s.%N) - $start" | bc) 
printf "single_100mito_round1 - Execution time: %.6f seconds" $dur >> 100mito_log.txt

start=$(date +%s.%N)
python3 -m fastai.launch train.py --bs 8 --lr 1e-4 --size 512 --tile_sz 512 --datasetname single_100mitotracker --cycles 30 --save_name single_100mito --lr_type s --n_frames 1 --load_name single_100mito_best_256 --freeze | tee -a 100mito_trainlog.txt
dur=$(echo "$(date +%s.%N) - $start" | bc)
printf "single_100mito_round2 - Execution time: %.6f seconds" $dur >> 100mito_log.txt

start=$(date +%s.%N)
python3 -m fastai.launch train.py --bs 8 --lr 1e-4 --lr_start 1e-6 --size 512 --tile_sz 512 --datasetname single_100mitotracker --cycles 40 --save_name single_100mito --lr_type s --n_frames 1 --load_name single_100mito_best_512 | tee -a 100mito_trainlog.txt
dur=$(echo "$(date +%s.%N) - $start" | bc)
printf "single_100mito_round3 - Execution time: %.6f seconds" $dur >> 100mito_log.txt

echo "single_100mito training finished."

start=$(date +%s.%N)
python3 image_gen.py stats/input stats/output --models single_100mito_best_512 --gpu 1 --baseline
dur=$(echo "$(date +%s.%N) - $start" | bc) 
printf "single_100mito_image_gen - Execution time: %.6f seconds" $dur >> 100mito_log.txt

echo "single_100mito inference finished."

python3 metric_gen.py -e semisynth_mito -p single_100mito_best_512
python3 metric_gen.py -e non-moving_mito -p single_100mito_best_512
echo "single_100mito stats generated."

start=$(date +%s.%N)
python3 -m fastai.launch  train.py --bs 24 --lr 4e-4 --size 256 --tile_sz 512 --datasetname multi_100mitotracker --cycles 40 --save_name multit_100mito --lr_type t --n_frames 5
dur=$(echo "$(date +%s.%N) - $start" | bc) | tee -a 100mito_trainlog.txt
printf "multi_100mito_round1 - Execution time: %.6f seconds" $dur >> 100mito_log.txt

start=$(date +%s.%N)
python3 -m fastai.launch  train.py --bs 8 --lr 1e-4 --size 512 --tile_sz 512 --datasetname multi_100mitotracker --cycles 30 --save_name multit_100mito --lr_type t --n_frames 5 --load_name multit_100mito_best_256 --freeze | tee -a 100mito_trainlog.txt
dur=$(echo "$(date +%s.%N) - $start" | bc)
printf "multi_100mito_round2 - Execution time: %.6f seconds" $dur >> 100mito_log.txt

start=$(date +%s.%N)
python3 -m fastai.launch  train.py --bs 8 --lr 1e-4 --lr_start 1e-6 --size 512 --tile_sz 512 --datasetname multi_100mitotracker --cycles 30 --save_name multit_100mito --lr_type t --n_frames 5 --load_name multit_100mito_best_512 | tee -a 100mito_trainlog.txt
dur=$(echo "$(date +%s.%N) - $start" | bc)
printf "multi_100mito_round3 - Execution time: %.6f seconds" $dur >> 100mito_log.txt

echo "multi_100mito training finished."

start=$(date +%s.%N)
python3 image_gen.py stats/input stats/output --models multit_100mito_best_512 --gpu 1 --baseline
dur=$(echo "$(date +%s.%N) - $start" | bc)
printf "multi_100mito_image_gen - Execution time: %.6f seconds" $dur >> 100mito_log.txt

echo "multi_100mito inference finished."

python3 metric_gen.py -e semisynth_mito -p multit_100mito_best_512
python3 metric_gen.py -e non-moving_mito -p multit_100mito_best_512
echo "multi_100mito stats generated."

echo "100mito ALL DONE YAY!!"
