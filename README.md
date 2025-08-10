## _CMA: A Unified Contextual Meta-Adaptation Methodology for Time-Series Denoising and Prediction_ [[Paper]](https://dl.acm.org/doi/10.1145/3711896.3736881) 
### Accepted in KDD 2025
### Haiqi Jiang, Ying Ding, Chenjie Pan, Aimin Huang, Rui Chen, Chenyou Fan
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](./LICENSE) 


## Requirements and Installation
```
conda create -n cma python==3.10
conda activate cma
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install einops ema-pytorch matplotlib scikit-learn scipy seaborn tqdm dm-control dm-env dm-tree mujoco gluonts
```

## Datasets

Please download the datasets from 
<a href="https://drive.google.com/drive/folders/1ir73qd_Ej2Zppa9u4Nfr2Q-n4BDsxEKO" target="_blank" rel="noopener noreferrer">Drive</a>

Unzip them and place under ./Data/datasets, or change Config/ yaml for your customized locations.

## Training CMA

### Step 1: Run the code of "Timexer" or "iTransformer" to get a pretrained TSP.

Here we demonstrate iTransformer as a TSP backbone, and show example on training with Traffic data
```
bash scripts_itrans/iTrans_traffic.sh  
```

After getting model checkpoints, modify the model path: ./Check_itrans/checkpoint_traffic.pth

### Step 2: Pretrain the initial CMA
```
python main.py --name traffic --config_file Config/traffic_dts.yaml --gpu=4 --train --milestone 1
```

### Step 3: Train the CMA with adaptation
```
python main.py --name traffic --config_file Config/traffic.yaml --gpu=4 --train --milestone 10 --pretrained
```

## Testing CMA
```
python main.py --name traffic --config_file Config/traffic.yaml --gpu 3 --sample 1  --milestone 20  --mode predict --pred_len 96
```




## Citation
If you find this repo helpful in your research, please kindly cite us.

```bibtex
@inproceedings{jiang2025cma,
author = {Jiang, Haiqi and Ding, Ying and Pan, Chenjie and Huang, Aimin and Chen, Rui and Fan, Chenyou},
title = {CMA: A Unified Contextual Meta-Adaptation Methodology for Time-Series Denoising and Prediction},
year = {2025},
booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
}
```
