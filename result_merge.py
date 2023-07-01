import pandas as pd

a = pd.read_csv('./output/96499_xDeepFM_4_Batch256_bins5_only_sparse.csv', sep='\t')
b = pd.read_csv('./output/3957_xDeepFM_4_Batch256_bins10_only_sparse.csv', sep='\t')
a['is_installed'] = (a['is_installed'] + b['is_installed']) / 2
a['is_clicked'] = (a['is_clicked'] + b['is_clicked']) / 2
a.to_csv('./output/avg_96499_3957.csv', index=False, sep='\t')


a = pd.read_csv('./output/avg_96499_3957.csv', sep='\t')
b = pd.read_csv('./output/1984_xDeepFM_4_Batch256_bins5_only_sparse.csv', sep='\t')
a['is_installed'] = (a['is_installed'] + b['is_installed']) / 2
a['is_clicked'] = (a['is_clicked'] + b['is_clicked']) / 2
a.to_csv('./output/avg_96499_3957_1984_1_1_2.csv', index=False, sep='\t')
