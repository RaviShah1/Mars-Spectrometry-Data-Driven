python generate_dataset.py -f max -i 100 -s data/savgol_features

python train_pipeline.py -m lgbm -f max100 -s saved_models/mskf_lgbm_savgol_max100.pkl

python inference_pipeline.py -m saved_models/mskf_lgbm_savgol_max100.pkl -f max100 -s submissions/mskf_lgbm_savgol_max100.csv
