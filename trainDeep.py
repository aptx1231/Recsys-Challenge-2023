import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch_local.models import xDeepFM, DeepFM, DCN, AutoInt, NFM, DCNMix, AFM
# TODO: DCN-M
from deepctr_torch_local.inputs import SparseFeat, DenseFeat, get_feature_names

from deepctr_torch_local.callbacks import EarlyStopping, ModelCheckpoint
import os
import torch
import random
import numpy as np 
import pandas as pd
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from utils import set_random_seed, get_logger, ensure_dir, str2bool, str2float

# TODO: sklearn 加seed

parser = argparse.ArgumentParser()
# 增加指定的参数
parser.add_argument('--model', type=str,
                        default='xDeepFM', help='the name of model')
parser.add_argument('--exp_id', type=str, default=None, help='id of experiment')
parser.add_argument('--seed', type=int, default=2023, help='random seed')
parser.add_argument('--dense_bins', type=int, default=10, help='dense_bins')
parser.add_argument('--sparse_dim', type=int, default=4, help='sparse_dim')
parser.add_argument('--add_dense', type=str2bool, default=False, help='add_dense')
parser.add_argument('--device', type=int, default=3, help='device')
parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
parser.add_argument('--epochs', type=int, default=50, help='epochs')
parser.add_argument('--patience', type=int, default=10, help='patience')
parser.add_argument('--cross_num', type=int, default=4, help='cross_num')
parser.add_argument('--heads', type=int, default=4, help='heads')
parser.add_argument('--strategy', type=str, default='kmeans', help='strategy')  # ["uniform"等宽, "quantile"等频, "kmeans"聚类]
parser.add_argument('--f1', type=int, default=256, help='f1')
parser.add_argument('--f2', type=int, default=256, help='f2')
parser.add_argument('--f3', type=int, default=256, help='f3')
parser.add_argument('--f4', type=int, default=138, help='f4')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
args = parser.parse_args()

# python trainDeep.py --model xDeepFM --dense_bins 5 --sparse_dim 4 --batch_size 256 --epoch 50 --patience 10 --device 2 --strategy uniform
# python trainDeep.py --model xDeepFM --dense_bins 5 --sparse_dim 4 --batch_size 256 --epoch 50 --patience 10 --device 3 --strategy quantile
# python trainDeep.py --model xDeepFM --dense_bins 5 --sparse_dim 8 --batch_size 256 --epoch 50 --patience 10 --device 3
# python trainDeep.py --model xDeepFM --dense_bins 5 --sparse_dim 16 --batch_size 256 --epoch 50 --patience 10 --device 3
# python trainDeep.py --model xDeepFM --dense_bins 10 --sparse_dim 4 --batch_size 256 --epoch 50 --patience 10 --device 1 --f1 128 --f2 128 --f3 256 --f4 128
# python trainDeep.py --model xDeepFM --dense_bins 10 --sparse_dim 4 --batch_size 256 --epoch 50 --patience 10 --device 1 --f1 256 --f2 256 --f3 128 --f4 64
# python trainDeep.py --model xDeepFM --dense_bins 5 --sparse_dim 4 --batch_size 256 --epoch 50 --patience 10 --device 2 --f1 128 --f2 128 --f3 256 --f4 128
# python trainDeep.py --model xDeepFM --dense_bins 10 --sparse_dim 4 --batch_size 256 --epoch 50 --patience 10 --device 1 --f1 256 --f2 256 --f3 256 --f4 256
# python trainDeep.py --model xDeepFM --dense_bins 5 --sparse_dim 4 --batch_size 256 --epoch 50 --patience 10 --device 2 --f1 256 --f2 256 --f3 256 --f4 256
# python trainDeep.py --model xDeepFM --dense_bins 10 --sparse_dim 4 --batch_size 256 --epoch 50 --patience 10 --device 3 --f1 128 --f2 128 --f3 128 --f4 128
# python trainDeep.py --model xDeepFM --dense_bins 5 --sparse_dim 4 --batch_size 256 --epoch 50 --patience 10 --device 3 --f1 128 --f2 128 --f3 128 --f4 128
# python trainDeep.py --model xDeepFM --dense_bins 5 --sparse_dim 4 --batch_size 256 --epoch 50 --patience 10 --device 2 --f1 256 --f2 256 --f3 128 --f4 64
# python trainDeep.py --model xDeepFM --dense_bins 10 --sparse_dim 4 --batch_size 256 --epoch 50 --patience 10 --device 1 --f1 256 --f2 256 --f3 256 --f4 128(默认)
# python trainDeep.py --model xDeepFM --dense_bins 5 --sparse_dim 4 --batch_size 256 --epoch 50 --patience 10 --device 3 --add_dense True
# python trainDeep.py --model DCN-M --dense_bins 10 --sparse_dim 4 --batch_size 256 --epoch 50 --patience 10 --device 2 --cross_num 6
# python trainDeep.py --model DCN-M --dense_bins 10 --sparse_dim 16 --batch_size 256 --epoch 50 --patience 10 --device 1 --cross_num 2
# python trainDeep.py --model DCN-Mix --dense_bins 10 --sparse_dim 4 --batch_size 256 --epoch 50 --patience 10 --device 1 --cross_num 4 --strategy quantile
# python trainDeep.py --model DCN-Mix --dense_bins 10 --sparse_dim 16 --batch_size 256 --epoch 50 --patience 10 --device 3 --cross_num 2
# python trainDeep.py --model DCN-M --dense_bins 10 --sparse_dim 4 --batch_size 256 --epoch 50 --patience 10 --device 1 --cross_num 4
# python trainDeep.py --model DCN-M --dense_bins 10 --sparse_dim 16 --batch_size 256 --epoch 50 --patience 10 --device 1 --cross_num 4
# python trainDeep.py --model DCN-Mix --dense_bins 10 --sparse_dim 4 --batch_size 256 --epoch 50 --patience 10 --device 3 --cross_num 4
# python trainDeep.py --model DCN-Mix --dense_bins 10 --sparse_dim 16 --batch_size 256 --epoch 50 --patience 10 --device 3 --cross_num 4
# python trainDeep.py --model AutoInt --dense_bins 10 --sparse_dim 4 --batch_size 256 --epoch 50 --patience 10 --device 2 --heads 4
# python trainDeep.py --model AutoInt --dense_bins 10 --sparse_dim 16 --batch_size 256 --epoch 50 --patience 10 --device 2 --heads 4
# python trainDeep.py --model xDeepFM --dense_bins 10 --sparse_dim 16 --batch_size 256 --epoch 50 --patience 10 --device 1
# python trainDeep.py --model xDeepFM --dense_bins 10 --sparse_dim 16 --batch_size 256 --epoch 50 --patience 10 --device 1

dense_bins = args.dense_bins
sparse_dim = args.sparse_dim
dense_dim = 1
strategy = args.strategy
add_dense = args.add_dense
dropout = args.dropout

device = 'cuda:{}'.format(args.device)
model_name = args.model
batch_size = args.batch_size
epochs = args.epochs
patience = args.patience
cross_num = args.cross_num  # for DCN-V, DCN-M, DCN-Mix
heads = args.heads # autoint
# lr = ?

seed = args.seed
set_random_seed(seed)

config = locals()

# 加载必要的数据

exp_id = config.get('exp_id', None)
if exp_id is None:
    exp_id = int(random.SystemRandom().random() * 100000)
    config['exp_id'] = exp_id

logger = get_logger(config)
logger.info('Exp_id {}'.format(exp_id))
logger.info(config)

logger.info('read data')

test_data = pd.read_csv('data/test/000000000000.csv', sep='\t')
logger.info('test_data: {}'.format(test_data.shape))

train_data = pd.read_csv('./data/train.csv', sep='\t')
logger.info('train_data: {}'.format(train_data.shape))

cat_features = ['f_{}'.format(i) for i in range(1, 42)]
bin_features = ['f_{}'.format(i) for i in range(33, 42)]
num_features = ['f_{}'.format(i) for i in range(42, 80)]
date_features = ['f_1']

sparse_features = cat_features
dense_features = num_features
logger.info('sparse_features: {}'.format(sparse_features))
logger.info('dense_features: {}'.format(dense_features))

target = ['is_installed']

data_path = 'data/data_dense_bins{}_{}.csv'.format(dense_bins, strategy)
logger.info('data path: {}'.format(data_path))
if os.path.exists(data_path):
    data = pd.read_csv(data_path)
    logger.info('Load data: {}'.format(data.shape))
    add_sparse_features = [fea + '_encode' for fea in dense_features]
else:
    data = pd.concat([train_data, test_data])
    logger.info('New data: {}'.format(data.shape))
    dense_mean = data[dense_features].mean()
    data[sparse_features] = data[sparse_features].fillna(-1, )
    # data[dense_features] = data[dense_features].fillna(0, )
    data[dense_features] = data[dense_features].fillna(dense_mean, )

    # 数据处理
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    add_sparse_features = []
    for fea in tqdm(dense_features, total=len(dense_features)):
        # TODO: random_state = seed 
        discretizer = KBinsDiscretizer(n_bins=dense_bins, encode='ordinal', strategy=strategy, random_state=seed)  # 等频quantile，等宽uniform
        data[fea + '_encode'] = discretizer.fit_transform(np.array(data[fea].tolist()).reshape(-1, 1))
        add_sparse_features.append(fea + '_encode')
    data.to_csv(data_path, index=False)
logger.info('add_sparse_features: {}'.format(add_sparse_features))

if add_dense:
    mms = MinMaxScaler(feature_range=(0,1))
    data[dense_features] = mms.fit_transform(data[dense_features])
    fixlen_feature_columns = \
        [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=sparse_dim) for i, feat in enumerate(sparse_features)] + \
        [SparseFeat(feat, vocabulary_size=dense_bins, embedding_dim=sparse_dim) for i, feat in enumerate(add_sparse_features)] + \
        [DenseFeat(feat, dense_dim) for feat in dense_features]
else:
    fixlen_feature_columns = \
    [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=sparse_dim) for i, feat in enumerate(sparse_features)] + \
    [SparseFeat(feat, vocabulary_size=dense_bins, embedding_dim=sparse_dim) for i, feat in enumerate(add_sparse_features)]

dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
logger.info('feature_names: {}'.format(feature_names))

# 数据
logger.info('Load Data')
train = data[~data['is_installed'].isna()]
test = data[data['is_installed'].isna()]

train_model_input = {name: train[name] for name in feature_names}
test_model_input = {name: test[name] for name in feature_names}

# 模型
logger.info('Create Model')
if model_name == 'xDeepFM':
    model = xDeepFM(linear_feature_columns, dnn_feature_columns, task='binary', device=device, seed=seed, dnn_dropout=dropout,
                    dnn_hidden_units=(args.f1, args.f2), cin_layer_size=(args.f3, args.f4))
elif model_name == 'DeepFM':
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary', device=device, seed=seed, dnn_dropout=dropout)
elif model_name == 'NFM':
    model = NFM(linear_feature_columns, dnn_feature_columns, task='binary', device=device, seed=seed, 
                biout_dropout=dropout, dnn_dropout=dropout, dnn_hidden_units=(256, 128))
elif model_name == 'AFM':
    model = AFM(linear_feature_columns, dnn_feature_columns, task='binary', device=device, seed=seed, afm_dropout=dropout)
elif model_name == 'DCN-V':
    model = DCN(linear_feature_columns, dnn_feature_columns, task='binary', device=device, seed=seed, dnn_dropout=dropout,
                dnn_hidden_units=(256, 128), cross_num=cross_num, cross_parameterization='vector')
elif model_name == 'DCN-M':
    model = DCN(linear_feature_columns, dnn_feature_columns, task='binary', device=device, seed=seed, dnn_dropout=dropout,
                dnn_hidden_units=(256, 128), cross_num=cross_num, cross_parameterization='matrix')
elif model_name == 'DCN-Mix':
    model = DCNMix(linear_feature_columns, dnn_feature_columns, task='binary', device=device, seed=seed, dnn_dropout=dropout,
                   dnn_hidden_units=(256, 128), cross_num=cross_num)
elif model_name == 'AutoInt':
    model = AutoInt(linear_feature_columns, dnn_feature_columns, task='binary', device=device, seed=seed, dnn_dropout=dropout,
                    att_head_num=heads)
else:
    raise ValueError('Error model name {}'.format(model_name))

model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'])
logger.info(model)

# 训练
logger.info('Training')
save_path = 'ckpt/{}_{}_{}_Batch{}_bins{}'.format(exp_id, model_name, sparse_dim, batch_size, dense_bins)
if not add_dense:
    save_path += '_only_sparse.ckpt'
else:
    save_path += '.ckpt'

es = EarlyStopping(monitor='val_binary_crossentropy', min_delta=0, verbose=1, patience=patience, mode='min')
mdckpt = ModelCheckpoint(filepath=save_path, monitor='val_binary_crossentropy', verbose=1, save_best_only=True, mode='min')
history = model.fit(train_model_input, train_data[target].values, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.2, callbacks=[es,mdckpt])

# 测试
# save_path = 'ckpt/xDeepFM_4_Batch256_bins10_onlysparse.ckpt'
model = torch.load(save_path)

pred_ans = model.predict(test_model_input, batch_size=batch_size)
logger.info('Pred Test Min {}, Max {}'.format(pred_ans.min(), pred_ans.max()))

pred_oof = model.predict(train_model_input, batch_size=batch_size)
logger.info('Pred Train Min {}, Max {}'.format(pred_oof.min(), pred_oof.max()))

logloss = metrics.log_loss(train_data[target], pred_oof)
acc = metrics.roc_auc_score(train_data[target], pred_oof)
precision = metrics.precision_score(train_data[target], [1 if i >= 0.5 else 0 for i in pred_oof])
recall = metrics.recall_score(train_data[target], [1 if i >= 0.5 else 0 for i in pred_oof])
f1 = metrics.f1_score(train_data[target], [1 if i >= 0.5 else 0 for i in pred_oof])

logger.info(f"Logloss: {logloss:.4f}, AUC: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

save_path_res = './output/{}_{}_{}_Batch{}_bins{}'.format(exp_id, model_name, sparse_dim, batch_size, dense_bins)
if not add_dense:
    save_path_res += '_only_sparse.csv'
else:
    save_path_res += '.csv'
logger.info('Save result to {}'.format(save_path_res))

submission = pd.DataFrame()
submission["RowId"] = test_data["f_0"]
submission["is_clicked"] = np.random.random((test_data.shape[0]))
submission["is_installed"] = pred_ans
submission.to_csv(save_path_res, index=False, sep='\t')

