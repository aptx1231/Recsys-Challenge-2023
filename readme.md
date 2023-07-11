# Solutions for [RecSys Challenge 2023](http://recsyschallenge.com/2023)
## Team Info

Team name: BUAA_BIGSCity

Team member: Jiawei Jiang, Wang Bing.

Team adviser: Jingyuan Wang.

Rank: 6th in the academic track

## Train & Test

You can reproduce the results by following these steps:

**Step 1**: Download the [dataset](https://sharechat.com/recsys2023/dashboard) and place it in the ``./data/`` directory to get the following directory structure:

```shell
./data/test/*.csv
./data/train/*.csv
```

**Step 2**: Environment Configuration:

```shell
conda create -n recsys python=3.9.7
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install -r requirements.txt
```

**Step 3**: Fuse the training data set:

```shell
python merge_data.py
```

This command will generate the file ``./data/train.csv`.

**Step 4**: Run command to train three models:

```shell
python trainDeep.py --model xDeepFM --dense_bins 5 --sparse_dim 4 --batch_size 256 --epoch 50 --patience 10 --device 2 --exp_id 96499
python trainDeep.py --model xDeepFM --dense_bins 10 --sparse_dim 4 --batch_size 256 --epoch 50 --patience 10 --device 1 --exp_id 3957
python trainDeep.py --model xDeepFM --dense_bins 5 --sparse_dim 4 --batch_size 256 --epoch 50 --patience 10 --device 1 --dropout 0.2 --exp_id 1984
```

Each command generates one model, where the exp_id in the command represents the ID record of one training. 

Correspondingly, the trained models are stored in `./ckpt/exp_id_*.ckpt`, and the model prediction results are stored in `./output/exp_id_*.csv`, and the logs of the training process are stored in `./log/exp_id_*.log`.

**Step 5**: Merge the output of 3 models. [This step can be skipped because the predicted results are already included in the `./output` directory.]

 ```shell
 python result_merge.py
 ```

This command will fuse the output of the 3 models to get the file `avg_96499_3957_1984_1_1_2.csv`. This file has a final submission score of 6.282142, ranking 6th in the academic track.

## Method Introduction

The proposal will be presented in the paper submitted to the workshop.

## Acknowledgements

[DeepCTR-Torch](https://github.com/shenweichen/DeepCTR-Torch)
