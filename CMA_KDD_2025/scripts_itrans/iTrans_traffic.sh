export CUDA_VISIBLE_DEVICES=1

model_name=iTransformer

for pred_len in 96 192 336 720
do
python -u run_itrans.py \
  --is_training 1 \
  --root_path Data/datasets/ \
  --data_path traffic.csv \
  --model_id traffic_96_$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len $pred_len \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1
done

