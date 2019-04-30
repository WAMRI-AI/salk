python gen_sample_info.py --skip random --out tif_sources.csv data/fixed data/live
python tile_from_info.py --out datasets/tilenorm_001 --info tif_sources.csv --tile 512 1024  --n_train 8000  --n_valid 1000

