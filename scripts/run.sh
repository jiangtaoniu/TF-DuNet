#!/bin/bash

# =================================================================================================
# Master script for running all TF-DuNet experiments.
#
# This script executes a series of experiments for the TF-DuNet model across various datasets.
# Using the unified run_PEMS08.py script.
# =================================================================================================

itr=1
DECOMP_METHOD="dft_decomp"
FINAL_FUSION="add"

# --- ETTh1 Dataset ---
echo "Running experiments for ETTh1 dataset..."
python -u run_PEMS08.py --is_training 1 --task_name long_term_forecast --data ETTh1 --root_path ./dataset/ETT/ --data_path ETTh1.csv --model_id ETTh1_96_96 --enc_in 7 --dec_in 7 --c_out 7 --pred_len 96 --d_model 128 --e_layers 5 --asst_e_layers 3 --batch_size 16 --itr $itr --decomp_method $DECOMP_METHOD --final_fusion_method $FINAL_FUSION
python -u run_PEMS08.py --is_training 1 --task_name long_term_forecast --data ETTh1 --root_path ./dataset/ETT/ --data_path ETTh1.csv --model_id ETTh1_96_192 --enc_in 7 --dec_in 7 --c_out 7 --pred_len 192 --d_model 128 --e_layers 5 --asst_e_layers 3 --batch_size 16 --itr $itr --decomp_method $DECOMP_METHOD --final_fusion_method $FINAL_FUSION
python -u run_PEMS08.py --is_training 1 --task_name long_term_forecast --data ETTh1 --root_path ./dataset/ETT/ --data_path ETTh1.csv --model_id ETTh1_96_336 --enc_in 7 --dec_in 7 --c_out 7 --pred_len 336 --d_model 128 --e_layers 5 --asst_e_layers 3 --batch_size 16 --itr $itr --decomp_method $DECOMP_METHOD --final_fusion_method $FINAL_FUSION
python -u run_PEMS08.py --is_training 1 --task_name long_term_forecast --data ETTh1 --root_path ./dataset/ETT/ --data_path ETTh1.csv --model_id ETTh1_96_720 --enc_in 7 --dec_in 7 --c_out 7 --pred_len 720 --d_model 128 --e_layers 5 --asst_e_layers 3 --batch_size 16 --itr $itr --decomp_method $DECOMP_METHOD --final_fusion_method $FINAL_FUSION

# --- ETTh2 Dataset ---
echo "Running experiments for ETTh2 dataset..."
python -u run_PEMS08.py --is_training 1 --task_name long_term_forecast --data ETTh2 --root_path ./dataset/ETT/ --data_path ETTh2.csv --model_id ETTh2_96_96 --enc_in 7 --dec_in 7 --c_out 7 --pred_len 96 --d_model 128 --e_layers 5 --asst_e_layers 3 --batch_size 16 --itr $itr --decomp_method $DECOMP_METHOD --final_fusion_method $FINAL_FUSION
python -u run_PEMS08.py --is_training 1 --task_name long_term_forecast --data ETTh2 --root_path ./dataset/ETT/ --data_path ETTh2.csv --model_id ETTh2_96_192 --enc_in 7 --dec_in 7 --c_out 7 --pred_len 192 --d_model 128 --e_layers 5 --asst_e_layers 3 --batch_size 16 --itr $itr --decomp_method $DECOMP_METHOD --final_fusion_method $FINAL_FUSION
python -u run_PEMS08.py --is_training 1 --task_name long_term_forecast --data ETTh2 --root_path ./dataset/ETT/ --data_path ETTh2.csv --model_id ETTh2_96_336 --enc_in 7 --dec_in 7 --c_out 7 --pred_len 336 --d_model 128 --e_layers 5 --asst_e_layers 3 --batch_size 16 --itr $itr --decomp_method $DECOMP_METHOD --final_fusion_method $FINAL_FUSION
python -u run_PEMS08.py --is_training 1 --task_name long_term_forecast --data ETTh2 --root_path ./dataset/ETT/ --data_path ETTh2.csv --model_id ETTh2_96_720 --enc_in 7 --dec_in 7 --c_out 7 --pred_len 720 --d_model 128 --e_layers 5 --asst_e_layers 3 --batch_size 16 --itr $itr --decomp_method $DECOMP_METHOD --final_fusion_method $FINAL_FUSION

# --- ETTm1 Dataset ---
echo "Running experiments for ETTm1 dataset..."
python -u run_PEMS08.py --is_training 1 --task_name long_term_forecast --data ETTm1 --root_path ./dataset/ETT/ --data_path ETTm1.csv --model_id ETTm1_96_96 --enc_in 7 --dec_in 7 --c_out 7 --pred_len 96 --d_model 128 --e_layers 5 --asst_e_layers 3 --batch_size 16 --itr $itr --decomp_method $DECOMP_METHOD --final_fusion_method $FINAL_FUSION
python -u run_PEMS08.py --is_training 1 --task_name long_term_forecast --data ETTm1 --root_path ./dataset/ETT/ --data_path ETTm1.csv --model_id ETTm1_96_192 --enc_in 7 --dec_in 7 --c_out 7 --pred_len 192 --d_model 128 --e_layers 5 --asst_e_layers 3 --batch_size 16 --itr $itr --decomp_method $DECOMP_METHOD --final_fusion_method $FINAL_FUSION
python -u run_PEMS08.py --is_training 1 --task_name long_term_forecast --data ETTm1 --root_path ./dataset/ETT/ --data_path ETTm1.csv --model_id ETTm1_96_336 --enc_in 7 --dec_in 7 --c_out 7 --pred_len 336 --d_model 128 --e_layers 5 --asst_e_layers 3 --batch_size 16 --itr $itr --decomp_method $DECOMP_METHOD --final_fusion_method $FINAL_FUSION
python -u run_PEMS08.py --is_training 1 --task_name long_term_forecast --data ETTm1 --root_path ./dataset/ETT/ --data_path ETTm1.csv --model_id ETTm1_96_720 --enc_in 7 --dec_in 7 --c_out 7 --pred_len 720 --d_model 128 --e_layers 5 --asst_e_layers 3 --batch_size 16 --itr $itr --decomp_method $DECOMP_METHOD --final_fusion_method $FINAL_FUSION

# --- ETTm2 Dataset ---
echo "Running experiments for ETTm2 dataset..."
python -u run_PEMS08.py --is_training 1 --task_name long_term_forecast --data ETTm2 --root_path ./dataset/ETT/ --data_path ETTm2.csv --model_id ETTm2_96_96 --enc_in 7 --dec_in 7 --c_out 7 --pred_len 96 --d_model 128 --e_layers 5 --asst_e_layers 3 --batch_size 16 --itr $itr --decomp_method $DECOMP_METHOD --final_fusion_method $FINAL_FUSION
python -u run_PEMS08.py --is_training 1 --task_name long_term_forecast --data ETTm2 --root_path ./dataset/ETT/ --data_path ETTm2.csv --model_id ETTm2_96_192 --enc_in 7 --dec_in 7 --c_out 7 --pred_len 192 --d_model 128 --e_layers 5 --asst_e_layers 3 --batch_size 16 --itr $itr --decomp_method $DECOMP_METHOD --final_fusion_method $FINAL_FUSION
python -u run_PEMS08.py --is_training 1 --task_name long_term_forecast --data ETTm2 --root_path ./dataset/ETT/ --data_path ETTm2.csv --model_id ETTm2_96_336 --enc_in 7 --dec_in 7 --c_out 7 --pred_len 336 --d_model 128 --e_layers 5 --asst_e_layers 3 --batch_size 16 --itr $itr --decomp_method $DECOMP_METHOD --final_fusion_method $FINAL_FUSION
python -u run_PEMS08.py --is_training 1 --task_name long_term_forecast --data ETTm2 --root_path ./dataset/ETT/ --data_path ETTm2.csv --model_id ETTm2_96_720 --enc_in 7 --dec_in 7 --c_out 7 --pred_len 720 --d_model 128 --e_layers 5 --asst_e_layers 3 --batch_size 16 --itr $itr --decomp_method $DECOMP_METHOD --final_fusion_method $FINAL_FUSION

# --- Electricity Dataset ---
echo "Running experiments for Electricity dataset..."
python -u run_PEMS08.py --is_training 1 --task_name long_term_forecast --data custom --root_path ./dataset/electricity/ --data_path electricity.csv --model_id Electricity_96_96 --enc_in 321 --dec_in 321 --c_out 321 --pred_len 96 --d_model 128 --e_layers 5 --asst_e_layers 3 --batch_size 16 --itr $itr --decomp_method $DECOMP_METHOD --final_fusion_method $FINAL_FUSION
python -u run_PEMS08.py --is_training 1 --task_name long_term_forecast --data custom --root_path ./dataset/electricity/ --data_path electricity.csv --model_id Electricity_96_192 --enc_in 321 --dec_in 321 --c_out 321 --pred_len 192 --d_model 128 --e_layers 5 --asst_e_layers 3 --batch_size 16 --itr $itr --decomp_method $DECOMP_METHOD --final_fusion_method $FINAL_FUSION
python -u run_PEMS08.py --is_training 1 --task_name long_term_forecast --data custom --root_path ./dataset/electricity/ --data_path electricity.csv --model_id Electricity_96_336 --enc_in 321 --dec_in 321 --c_out 321 --pred_len 336 --d_model 128 --e_layers 5 --asst_e_layers 3 --batch_size 16 --itr $itr --decomp_method $DECOMP_METHOD --final_fusion_method $FINAL_FUSION
python -u run_PEMS08.py --is_training 1 --task_name long_term_forecast --data custom --root_path ./dataset/electricity/ --data_path electricity.csv --model_id Electricity_96_720 --enc_in 321 --dec_in 321 --c_out 321 --pred_len 720 --d_model 128 --e_layers 5 --asst_e_layers 3 --batch_size 16 --itr $itr --decomp_method $DECOMP_METHOD --final_fusion_method $FINAL_FUSION

# --- Traffic Dataset ---
echo "Running experiments for Traffic dataset..."
python -u run_PEMS08.py --is_training 1 --task_name long_term_forecast --data custom --root_path ./dataset/traffic/ --data_path traffic.csv --model_id Traffic_96_96 --enc_in 862 --dec_in 862 --c_out 862 --pred_len 96 --d_model 128 --e_layers 5 --asst_e_layers 3 --batch_size 16 --itr $itr --decomp_method $DECOMP_METHOD --final_fusion_method $FINAL_FUSION
python -u run_PEMS08.py --is_training 1 --task_name long_term_forecast --data custom --root_path ./dataset/traffic/ --data_path traffic.csv --model_id Traffic_96_192 --enc_in 862 --dec_in 862 --c_out 862 --pred_len 192 --d_model 128 --e_layers 5 --asst_e_layers 3 --batch_size 16 --itr $itr --decomp_method $DECOMP_METHOD --final_fusion_method $FINAL_FUSION
python -u run_PEMS08.py --is_training 1 --task_name long_term_forecast --data custom --root_path ./dataset/traffic/ --data_path traffic.csv --model_id Traffic_96_336 --enc_in 862 --dec_in 862 --c_out 862 --pred_len 336 --d_model 128 --e_layers 5 --asst_e_layers 3 --batch_size 16 --itr $itr --decomp_method $DECOMP_METHOD --final_fusion_method $FINAL_FUSION
python -u run_PEMS08.py --is_training 1 --task_name long_term_forecast --data custom --root_path ./dataset/traffic/ --data_path traffic.csv --model_id Traffic_96_720 --enc_in 862 --dec_in 862 --c_out 862 --pred_len 720 --d_model 128 --e_layers 5 --asst_e_layers 3 --batch_size 16 --itr $itr --decomp_method $DECOMP_METHOD --final_fusion_method $FINAL_FUSION

# --- Weather Dataset ---
echo "Running experiments for Weather dataset..."
python -u run_PEMS08.py --is_training 1 --task_name long_term_forecast --data custom --root_path ./dataset/weather/ --data_path weather.csv --model_id Weather_96_96 --enc_in 21 --dec_in 21 --c_out 21 --pred_len 96 --d_model 128 --e_layers 5 --asst_e_layers 3 --batch_size 16 --itr $itr --decomp_method $DECOMP_METHOD --final_fusion_method $FINAL_FUSION
python -u run_PEMS08.py --is_training 1 --task_name long_term_forecast --data custom --root_path ./dataset/weather/ --data_path weather.csv --model_id Weather_96_192 --enc_in 21 --dec_in 21 --c_out 21 --pred_len 192 --d_model 128 --e_layers 5 --asst_e_layers 3 --batch_size 16 --itr $itr --decomp_method $DECOMP_METHOD --final_fusion_method $FINAL_FUSION
python -u run_PEMS08.py --is_training 1 --task_name long_term_forecast --data custom --root_path ./dataset/weather/ --data_path weather.csv --model_id Weather_96_336 --enc_in 21 --dec_in 21 --c_out 21 --pred_len 336 --d_model 128 --e_layers 5 --asst_e_layers 3 --batch_size 16 --itr $itr --decomp_method $DECOMP_METHOD --final_fusion_method $FINAL_FUSION
python -u run_PEMS08.py --is_training 1 --task_name long_term_forecast --data custom --root_path ./dataset/weather/ --data_path weather.csv --model_id Weather_96_720 --enc_in 21 --dec_in 21 --c_out 21 --pred_len 720 --d_model 128 --e_layers 5 --asst_e_layers 3 --batch_size 16 --itr $itr --decomp_method $DECOMP_METHOD --final_fusion_method $FINAL_FUSION

# --- PEMS Datasets (Short-term Forecast) ---
echo "Running experiments for PEMS datasets..."
python -u run_PEMS08.py --is_training 1 --task_name short_term_forecast --data PEMS --root_path ./dataset/PEMS/ --data_path PEMS03.npz --model_id PEMS03_96_12 --enc_in 358 --dec_in 358 --c_out 358 --pred_len 12 --d_model 128 --e_layers 5 --asst_e_layers 3 --batch_size 16 --itr $itr --decomp_method $DECOMP_METHOD --final_fusion_method $FINAL_FUSION
python -u run_PEMS08.py --is_training 1 --task_name short_term_forecast --data PEMS --root_path ./dataset/PEMS/ --data_path PEMS03.npz --model_id PEMS03_96_24 --enc_in 358 --dec_in 358 --c_out 358 --pred_len 24 --d_model 128 --e_layers 5 --asst_e_layers 3 --batch_size 16 --itr $itr --decomp_method $DECOMP_METHOD --final_fusion_method $FINAL_FUSION
python -u run_PEMS08.py --is_training 1 --task_name short_term_forecast --data PEMS --root_path ./dataset/PEMS/ --data_path PEMS03.npz --model_id PEMS03_96_48 --enc_in 358 --dec_in 358 --c_out 358 --pred_len 48 --d_model 128 --e_layers 5 --asst_e_layers 3 --batch_size 16 --itr $itr --decomp_method $DECOMP_METHOD --final_fusion_method $FINAL_FUSION
python -u run_PEMS08.py --is_training 1 --task_name short_term_forecast --data PEMS --root_path ./dataset/PEMS/ --data_path PEMS03.npz --model_id PEMS03_96_96 --enc_in 358 --dec_in 358 --c_out 358 --pred_len 96 --d_model 128 --e_layers 5 --asst_e_layers 3 --batch_size 16 --itr $itr --decomp_method $DECOMP_METHOD --final_fusion_method $FINAL_FUSION
python -u run_PEMS08.py --is_training 1 --task_name short_term_forecast --data PEMS --root_path ./dataset/PEMS/ --data_path PEMS08.npz --model_id PEMS08_96_12 --enc_in 170 --dec_in 170 --c_out 170 --pred_len 12 --d_model 128 --e_layers 5 --asst_e_layers 3 --batch_size 16 --itr $itr --decomp_method $DECOMP_METHOD --final_fusion_method $FINAL_FUSION
python -u run_PEMS08.py --is_training 1 --task_name short_term_forecast --data PEMS --root_path ./dataset/PEMS/ --data_path PEMS08.npz --model_id PEMS08_96_24 --enc_in 170 --dec_in 170 --c_out 170 --pred_len 24 --d_model 128 --e_layers 5 --asst_e_layers 3 --batch_size 16 --itr $itr --decomp_method $DECOMP_METHOD --final_fusion_method $FINAL_FUSION
python -u run_PEMS08.py --is_training 1 --task_name short_term_forecast --data PEMS --root_path ./dataset/PEMS/ --data_path PEMS08.npz --model_id PEMS08_96_48 --enc_in 170 --dec_in 170 --c_out 170 --pred_len 48 --d_model 128 --e_layers 5 --asst_e_layers 3 --batch_size 16 --itr $itr --decomp_method $DECOMP_METHOD --final_fusion_method $FINAL_FUSION
python -u run_PEMS08.py --is_training 1 --task_name short_term_forecast --data PEMS --root_path ./dataset/PEMS/ --data_path PEMS08.npz --model_id PEMS08_96_96 --enc_in 170 --dec_in 170 --c_out 170 --pred_len 96 --d_model 128 --e_layers 5 --asst_e_layers 3 --batch_size 16 --itr $itr --decomp_method $DECOMP_METHOD --final_fusion_method $FINAL_FUSION

echo "All experiments completed."
