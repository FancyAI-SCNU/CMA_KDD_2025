# Requirement
conda create -n diffts python==3.10
conda activate diffts
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install einops ema-pytorch matplotlib scikit-learn scipy seaborn tqdm dm-control dm-env dm-tree mujoco gluonts
 

# Training CMA(Traffic Datasets)

Step1: Run the code of "Timexer" or "iTransformer" to get a pretrained TSP. Here we demonstrate iTransformer as a TSP.
bash scripts_itrans/iTrans_traffic.sh  
# After getting model checkpoints, modify the model path: ./Check_itrans/checkpoint_traffic.pth

Step2: Pretrain the initial CMA
python main.py --name traffic --config_file Config/traffic_dts.yaml --gpu=4 --train --milestone 1

Step3: Train the CMA with adaptation
python main.py --name traffic --config_file Config/traffic.yaml --gpu=4 --train --milestone 10 --pretrained

# Testing CMA(Traffic Datasets)
python main.py --name traffic --config_file Config/traffic.yaml --gpu 3 --sample 1  --milestone 20  --mode predict --pred_len 96