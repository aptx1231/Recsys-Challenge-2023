import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch_local.models import *
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


ensure_dir('ckpt/')
ensure_dir('log/')
ensure_dir('output/')

parser = argparse.ArgumentParser()
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
parser.add_argument('--f4', type=int, default=128, help='f4')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
parser.add_argument('--lr', type=float, default=0.001, help='lr')
parser.add_argument('--load_path', type=str, default=None, help='load path')
parser.add_argument('--local_test', type=str2bool, default=False, help='local_test')
parser.add_argument('--binary_cross', type=str2bool, default=False, help='binary_cross')
parser.add_argument('--multi_cross', type=str2bool, default=False, help='multi_cross')
parser.add_argument('--num_cross', type=str2bool, default=False, help='num_cross')
parser.add_argument('--target_encode', type=str2bool, default=False, help='target_encode')
args = parser.parse_args()

dense_bins = args.dense_bins
sparse_dim = args.sparse_dim
dense_dim = 1
strategy = args.strategy
add_dense = args.add_dense
dropout = args.dropout
local_test = args.local_test
lr = args.lr
binary_cross = args.binary_cross
multi_cross = args.multi_cross
num_cross = args.num_cross
target_encode = args.target_encode

device = 'cuda:{}'.format(args.device)
model_name = args.model
batch_size = args.batch_size
epochs = args.epochs
patience = args.patience
cross_num = args.cross_num  # for DCN-V, DCN-M, DCN-Mix
heads = args.heads # autoint DIFM

seed = args.seed
set_random_seed(seed)

config = locals()

# 加载必要的数据

exp_id = args.exp_id
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

cat_features = ['f_{}'.format(i) for i in range(1, 42) if i != 30 and i != 31]
bin_features = ['f_{}'.format(i) for i in range(33, 42)]
num_features = ['f_{}'.format(i) for i in range(42, 80)]
date_features = ['f_1']

sparse_features = cat_features
dense_features = num_features
logger.info('sparse_features: {}'.format(sparse_features))
logger.info('dense_features: {}'.format(dense_features))

target = ['is_installed']

# if not target_encode:
#     data_path = 'data/data_dense_bins{}_{}_updatef1.csv'.format(dense_bins, strategy)
# else:
#     data_path = 'data/data_dense_bins{}_{}_updatef1_target_encode.csv'.format(dense_bins, strategy)
if not target_encode:
    data_path = 'data/data_dense_bins{}_{}.csv'.format(dense_bins, strategy)
else:
    data_path = 'data/data_dense_bins{}_{}_target_encode.csv'.format(dense_bins, strategy)
logger.info('data path: {}'.format(data_path))
if os.path.exists(data_path):
    data = pd.read_csv(data_path)
    logger.info('Load data: {}'.format(data.shape))
    add_sparse_features = [fea + '_encode' for fea in dense_features]
    len_feas = {}
    for feat in sparse_features:
        len_feas[feat] = data[feat].nunique()
else:
    data = pd.concat([train_data, test_data])
    len_feas = {}
    for feat in sparse_features:
        len_feas[feat] = data[feat].nunique()
    logger.info('New data: {}'.format(data.shape))
    dense_mean = data[dense_features].mean()
    # data[sparse_features] = data[sparse_features].fillna(-1, )
    data[dense_features] = data[dense_features].fillna(dense_mean, )

    # 数据处理
    for feat in sparse_features:
        if feat == 'f_1':
            data[feat] = data[feat] - 45  # 45是最小的
        else:
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

if binary_cross:
    binary_cross_features = []
    for i in tqdm(range(len(bin_features)), total=len(bin_features), desc='binary_cross'):
        for j in range(i + 1, len(bin_features)):
            data[f'{bin_features[i]}and{bin_features[j]}'] = data[bin_features[i]] & data[bin_features[j]]
            data[f'{bin_features[i]}or{bin_features[j]}'] = data[bin_features[i]] | data[bin_features[j]]
            data[f'{bin_features[i]}xor{bin_features[j]}'] = data[bin_features[i]] ^ data[bin_features[j]]
            binary_cross_features.append(f'{bin_features[i]}and{bin_features[j]}')
            binary_cross_features.append(f'{bin_features[i]}or{bin_features[j]}')
            binary_cross_features.append(f'{bin_features[i]}xor{bin_features[j]}')
    logger.info('binary_cross_features: {}'.format(binary_cross_features))
    logger.info('binary_cross data: {}'.format(data.shape))
    for feat in binary_cross_features:
        len_feas[feat] = data[feat].nunique()

if multi_cross:
    multi_cross_features = []
    cat_features_crossuse = ['f_3', 'f_5', 'f_7', 'f_8', 'f_9', 'f_10', 'f_11', 
                             'f_12', 'f_14', 'f_16', 'f_17', 'f_19', 'f_20', 'f_21', 
                             'f_22', 'f_23', 'f_24', 'f_25', 'f_26', 'f_27', 'f_28', 'f_29', 'f_32', 'f_33', 
                             'f_34', 'f_35', 'f_36', 'f_37', 'f_38', 'f_39', 'f_40', 'f_41']
    for i in tqdm(range(len(cat_features_crossuse)), total=len(cat_features_crossuse), desc='multi_cross'):
        for j in range(i + 1, len(cat_features_crossuse)):
            data[f'{cat_features_crossuse[i]}+{cat_features_crossuse[j]}'] = data[cat_features_crossuse[i]] + data[cat_features_crossuse[j]]
            # data[f'{cat_features_crossuse[i]}-{cat_features_crossuse[j]}'] = data[cat_features_crossuse[i]] - data[cat_features_crossuse[j]]
            data[f'{cat_features_crossuse[i]}*{cat_features_crossuse[j]}'] = data[cat_features_crossuse[i]] * data[cat_features_crossuse[j]]
            # data[f'{cat_features_crossuse[i]}/{cat_features_crossuse[j]}'] = data[cat_features_crossuse[i]] / (data[cat_features_crossuse[j]] + 1e-7)
            multi_cross_features.append(f'{cat_features_crossuse[i]}+{cat_features_crossuse[j]}')
            multi_cross_features.append(f'{cat_features_crossuse[i]}*{cat_features_crossuse[j]}')
    # logger.info('multi_cross_features: {}'.format(multi_cross_features))
    logger.info('multi_cross data: {}'.format(data.shape))
    for feat in multi_cross_features:
        len_feas[feat] = int(data[feat].max() + 1)

if num_cross:
    num_cross_features = []
    for i in tqdm(range(len(add_sparse_features)), total=len(add_sparse_features), desc='num_cross'):
        for j in range(i + 1, len(add_sparse_features)):
            data[f'{add_sparse_features[i]}+{add_sparse_features[j]}'] = data[add_sparse_features[i]] + data[add_sparse_features[j]]
            # data[f'{add_sparse_features[i]}-{add_sparse_features[j]}'] = data[add_sparse_features[i]] - data[add_sparse_features[j]]
            data[f'{add_sparse_features[i]}*{add_sparse_features[j]}'] = data[add_sparse_features[i]] * data[add_sparse_features[j]]
            # data[f'{add_sparse_features[i]}/{add_sparse_features[j]}'] = data[add_sparse_features[i]] / (data[add_sparse_features[j]] + 1e-7)
            num_cross_features.append(f'{add_sparse_features[i]}+{add_sparse_features[j]}')
            num_cross_features.append(f'{add_sparse_features[i]}*{add_sparse_features[j]}')
    # logger.info('num_cross_features: {}'.format(num_cross_features))
    logger.info('num_cross data: {}'.format(data.shape))
    for feat in num_cross_features:
        len_feas[feat] = int(data[feat].max() + 1)
        print(feat, data[feat].max() + 1)

if local_test:
    logger.info('Test local using f_1 = 66!')
    local_valid_data = data[data['f_1'] == 21]  # 21 + 45 = 66
    data = data[data['f_1'] != 21]
    logger.info('local_valid_data: {}'.format(local_valid_data.shape))
    logger.info('Other data: {}'.format(data.shape))      

if add_dense:
    mms = MinMaxScaler(feature_range=(0,1))
    data[dense_features] = mms.fit_transform(data[dense_features])
    fixlen_feature_columns = \
        [SparseFeat(feat, vocabulary_size=len_feas[feat], embedding_dim=sparse_dim) for i, feat in enumerate(sparse_features)] + \
        [SparseFeat(feat, vocabulary_size=dense_bins, embedding_dim=sparse_dim) for i, feat in enumerate(add_sparse_features)] + \
        [DenseFeat(feat, dense_dim) for feat in dense_features]
    features = sparse_features + add_sparse_features + dense_features
else:
    fixlen_feature_columns = \
    [SparseFeat(feat, vocabulary_size=len_feas[feat], embedding_dim=sparse_dim) for i, feat in enumerate(sparse_features)] + \
    [SparseFeat(feat, vocabulary_size=dense_bins, embedding_dim=sparse_dim) for i, feat in enumerate(add_sparse_features)]
    features = sparse_features + add_sparse_features

if target_encode:
    target_fea = [f for f in data.columns if 'is_installed_mean' in f]
    fixlen_feature_columns += [DenseFeat(feat, dense_dim) for feat in target_fea]
    features += target_fea
    print(len(features))
    dense_mean_target_fea = data[target_fea].mean()
    data[target_fea] = data[target_fea].fillna(dense_mean_target_fea)

if binary_cross:
    fixlen_feature_columns += [SparseFeat(feat, vocabulary_size=len_feas[feat], embedding_dim=sparse_dim) for i, feat in enumerate(binary_cross_features)]
    features += binary_cross_features

if multi_cross:
    fixlen_feature_columns += [SparseFeat(feat, vocabulary_size=len_feas[feat], embedding_dim=sparse_dim) for i, feat in enumerate(multi_cross_features)]
    features += multi_cross_features

if num_cross:
    fixlen_feature_columns += [SparseFeat(feat, vocabulary_size=len_feas[feat], embedding_dim=sparse_dim) for i, feat in enumerate(num_cross_features)]
    features += num_cross_features

dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
# logger.info('feature_names: {}'.format(feature_names))

# 数据
logger.info('Load Data')
train = data[~data['is_installed'].isna()][features]
test = data[data['is_installed'].isna()][features]
y = data[~data['is_installed'].isna()][target]
logger.info('train shape {}'.format(train.shape))
logger.info('test shape {}'.format(test.shape))
logger.info('y shape {}'.format(y.shape))

if local_test:
    local_valid_data_y = local_valid_data[target]
    local_valid_data = local_valid_data[features]

train_model_input = {name: train[name] for name in feature_names}
test_model_input = {name: test[name] for name in feature_names}

# 模型
logger.info('Create Model')
if model_name == 'xDeepFM':
    model = xDeepFM(linear_feature_columns, dnn_feature_columns, task='binary', device=device, seed=seed, dnn_dropout=dropout,
                    dnn_hidden_units=(args.f1, args.f2), cin_layer_size=(args.f3, args.f4))
elif model_name == 'DeepFM':
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary', device=device, seed=seed, dnn_dropout=dropout,
                   dnn_hidden_units=(args.f1, args.f2))
elif model_name == 'NFM':
    model = NFM(linear_feature_columns, dnn_feature_columns, task='binary', device=device, seed=seed, 
                bi_dropout=dropout, dnn_dropout=dropout, dnn_hidden_units=(args.f1, args.f2))
elif model_name == 'AFM':
    model = AFM(linear_feature_columns, dnn_feature_columns, task='binary', device=device, seed=seed, afm_dropout=dropout)
elif model_name == 'DCN-V':
    model = DCN(linear_feature_columns, dnn_feature_columns, task='binary', device=device, seed=seed, dnn_dropout=dropout,
                dnn_hidden_units=(args.f1, args.f2), cross_num=cross_num, cross_parameterization='vector')
elif model_name == 'DCN-M':
    model = DCN(linear_feature_columns, dnn_feature_columns, task='binary', device=device, seed=seed, dnn_dropout=dropout,
                dnn_hidden_units=(args.f1, args.f2), cross_num=cross_num, cross_parameterization='matrix')
elif model_name == 'DCN-Mix':
    model = DCNMix(linear_feature_columns, dnn_feature_columns, task='binary', device=device, seed=seed, dnn_dropout=dropout,
                   dnn_hidden_units=(args.f1, args.f2), cross_num=cross_num)
elif model_name == 'AutoInt':
    model = AutoInt(linear_feature_columns, dnn_feature_columns, task='binary', device=device, seed=seed, dnn_dropout=dropout,
                    dnn_hidden_units=(args.f1, args.f2), att_head_num=heads)
elif model_name == 'WDL':
    model = WDL(linear_feature_columns, dnn_feature_columns, task='binary', device=device, seed=seed, dnn_dropout=dropout,
                dnn_hidden_units=(args.f1, args.f2))
elif model_name == 'DIFM':
    model = DIFM(linear_feature_columns, dnn_feature_columns, task='binary', device=device, seed=seed, dnn_dropout=dropout,
                 att_head_num=heads, dnn_hidden_units=(args.f1, args.f2))
elif model_name == 'IFM':
    model = IFM(linear_feature_columns, dnn_feature_columns, task='binary', device=device, seed=seed, dnn_dropout=dropout,
                dnn_hidden_units=(args.f1, args.f2))
elif model_name == 'FiBiNET':
    model = FiBiNET(linear_feature_columns, dnn_feature_columns, task='binary', device=device, seed=seed, dnn_dropout=dropout,
                    dnn_hidden_units=(args.f1, args.f2))
elif model_name == 'ONN':
    model = ONN(linear_feature_columns, dnn_feature_columns, task='binary', device=device, seed=seed, dnn_dropout=dropout,
                dnn_hidden_units=(args.f1, args.f2))
elif model_name == 'PNN':
    model = PNN(dnn_feature_columns, task='binary', device=device, seed=seed, dnn_dropout=dropout,
                dnn_hidden_units=(args.f1, args.f2))
elif model_name == 'AFN':
    model = AFN(linear_feature_columns, dnn_feature_columns, task='binary', device=device, seed=seed, dnn_dropout=dropout,
                afn_dnn_hidden_units=(args.f1, args.f2))
elif model_name == 'CCPM':
    model = CCPM(linear_feature_columns, dnn_feature_columns, task='binary', device=device, seed=seed, dnn_dropout=dropout,
                dnn_hidden_units=(args.f1, args.f2))
else:
    raise ValueError('Error model name {}'.format(model_name))

model.compile(torch.optim.Adam(model.parameters(), lr=lr), "binary_crossentropy", metrics=['binary_crossentropy'])
logger.info(model)

if args.load_path is None:
    # 训练
    logger.info('Training')
    save_path = 'ckpt/{}_{}_{}_Batch{}_bins{}'.format(exp_id, model_name, sparse_dim, batch_size, dense_bins)
    if not add_dense:
        save_path += '_only_sparse.ckpt'
    else:
        save_path += '.ckpt'

    es = EarlyStopping(monitor='val_binary_crossentropy', min_delta=0, verbose=1, patience=patience, mode='min')
    mdckpt = ModelCheckpoint(filepath=save_path, monitor='val_binary_crossentropy', verbose=1, save_best_only=True, mode='min')
    history = model.fit(train_model_input, y.values, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.2, callbacks=[es,mdckpt])

    # 测试
    model = torch.load(save_path)
else:
    logger.info('Loading {}'.format(args.load_path))
    model = torch.load(args.load_path)

logger.info('Testing')
pred_ans = model.predict(test_model_input, batch_size=batch_size)
logger.info('Pred Test Min {}, Max {}, Mean {}'.format(pred_ans.min(), pred_ans.max(), pred_ans.mean()))

pred_oof = model.predict(train_model_input, batch_size=batch_size)
logger.info('Pred Train Min {}, Max {}, Mean {}'.format(pred_oof.min(), pred_oof.max(), pred_oof.mean()))

logloss = metrics.log_loss(y, pred_oof)
acc = metrics.roc_auc_score(y, pred_oof)
precision = metrics.precision_score(y, [1 if i >= 0.5 else 0 for i in pred_oof])
recall = metrics.recall_score(y, [1 if i >= 0.5 else 0 for i in pred_oof])
f1 = metrics.f1_score(y, [1 if i >= 0.5 else 0 for i in pred_oof])

logger.info(f"Train Data Logloss: {logloss:.4f}, AUC: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

if local_test:
    logger.info('Local Testing')
    logger.info('Local test data shape: {}'.format(local_valid_data.shape))
    local_valid_data_input = {name: local_valid_data[name] for name in feature_names}

    pred_ans_valid = model.predict(local_valid_data_input, batch_size=batch_size)
    logger.info('Pred Test Min {}, Max {}, Mean {}'.format(pred_ans_valid.min(), pred_ans_valid.max(), pred_ans_valid.mean()))

    logloss = metrics.log_loss(local_valid_data_y, pred_ans_valid)
    acc = metrics.roc_auc_score(local_valid_data_y, pred_ans_valid)
    precision = metrics.precision_score(local_valid_data_y, [1 if i >= 0.5 else 0 for i in pred_ans_valid])
    recall = metrics.recall_score(local_valid_data_y, [1 if i >= 0.5 else 0 for i in pred_ans_valid])
    f1 = metrics.f1_score(local_valid_data_y, [1 if i >= 0.5 else 0 for i in pred_ans_valid])

    logger.info(f"Local Test Logloss: {logloss:.4f}, AUC: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

save_path_res = './output/{}_{}_{}_Batch{}_bins{}'.format(exp_id, model_name, sparse_dim, batch_size, dense_bins)
if not add_dense:
    save_path_res += '_only_sparse'
logger.info('Save result to {}'.format(save_path_res + '.csv'))

submission = pd.DataFrame()
submission["RowId"] = test_data["f_0"]
submission["is_clicked"] = np.random.random((test_data.shape[0]))
submission["is_installed"] = pred_ans
submission.to_csv(save_path_res + '.csv', index=False, sep='\t')

