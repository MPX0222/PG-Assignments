import torch
import numpy as np
import pandas as pd
from train_deepnet_demo import Net
from torch.utils.data import DataLoader, TensorDataset

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import collections

data_path_1 = r'data/test/lh.ThickAvg- 389.csv'
data_path_2 = r'data/test/rh.ThickAvg- 389.csv'
data_path_3 = r'data/test/lh.GausCurv- 389.csv'
data_path_4 = r'data/test/rh.GausCurv- 389.csv'
data_path_5 = r'data/test/lh.MeanCurv- 389.csv'
data_path_6 = r'data/test/rh.MeanCurv- 389.csv'
data_path_7 = r'data/test/lh.SurfArea - 389.csv'
data_path_8 = r'data/test/rh.SurfArea - 389.csv'
data_path_9 = r'data/test/lh.GrayVol - 389.csv'
data_path_10 = r'data/test/rh.GrayVol- 389.csv'

label_path = r'subject_info - 389.csv'
ori_label_path = r'data/train/subject_info - 1600.csv'

# 处理测试集数据
original_x1, original_x2, original_x3, original_x4, original_x5, original_x6, original_x7, original_x8, original_x9, original_x10 = pd.read_csv(data_path_1), pd.read_csv(data_path_2), pd.read_csv(data_path_3).dropna(), pd.read_csv(data_path_4), pd.read_csv(data_path_5), pd.read_csv(data_path_6), pd.read_csv(data_path_7), pd.read_csv(data_path_8), pd.read_csv(data_path_9), pd.read_csv(data_path_10)

print(original_x1.shape, original_x2.shape, original_x3.shape, original_x4.shape, original_x5.shape, original_x6.shape, original_x7.shape, original_x8.shape, original_x9.shape, original_x10.shape)


original_x1, original_x2, original_x3, original_x4 = original_x1.iloc[:, 1:], original_x2.iloc[:, 1:], original_x3.iloc[:, 1:], original_x4.iloc[:, 1:]
original_x5, original_x6, original_x7, original_x8 = original_x5.iloc[:, 1:], original_x6.iloc[:, 1:], original_x7.iloc[:, 1:], original_x8.iloc[:, 1:]
original_x9, original_x10 = original_x9.iloc[:, 1:], original_x10.iloc[:, 1:]

original_x1, original_x2, original_x3, original_x4 = np.array(original_x1), np.array(original_x2), np.array(original_x3), np.array(original_x4)
original_x5, original_x6, original_x7, original_x8 = np.array(original_x5), np.array(original_x6), np.array(original_x7), np.array(original_x8)
original_x9, original_x10 = np.array(original_x9), np.array(original_x10)

print(original_x1.shape, original_x2.shape, original_x3.shape, original_x4.shape, original_x5.shape, original_x6.shape, original_x7.shape, original_x8.shape, original_x9.shape, original_x10.shape)

original_x_thick = (original_x1 + original_x2) * 0.5
original_x_gauscurv = (original_x3 + original_x4) * 0.5
original_x_meancurv = (original_x5 + original_x6) * 0.5
original_x_surfarea = (original_x7 + original_x8) * 0.5
original_x_grayvol = (original_x9 + original_x10) * 0.5
   
# 可视化原始年龄分布
# label = pd.read_csv(ori_label_path)
# label = label.iloc[:, 3]
# value_counts = label.value_counts()
# print('ORI Label Value Counts:\n')
# print(value_counts)

# 计算输入
original_x = np.hstack([original_x_thick, original_x_gauscurv, original_x_meancurv, original_x_surfarea])

original_x = torch.tensor(original_x, dtype=torch.float32)

model_1 = torch.load('5_fc_layer_model-x5x7-avg0_2-epoch200-lr0_00005.pkl')
model_2 = torch.load('5_fc_layer_model-x5x7-avg0_3-epoch200-lr0_00005.pkl')
model_3 = torch.load('5_fc_layer_model-x5x7-avg0_4-epoch200-lr0_00005.pkl')
model_4 = torch.load('5_fc_layer_model-x5x7-avg0_5-epoch200-lr0_00005.pkl')
model_5 = torch.load('5_fc_layer_model-x5x7-avg0_6-epoch200-lr0_00005.pkl')
model_6 = torch.load('5_fc_layer_model-x5x7-avg0_7-epoch200-lr0_00005.pkl')
model_7 = torch.load('5_fc_layer_model-x5x7-avg0_8-epoch200-lr0_00005.pkl')

model_8 = torch.load('5_fc_layer_model-test1.pkl')
model_9 = torch.load('5_fc_layer_model-ensemble_thick_gaus_mean_surf.pkl')

# predict_data_1, predict_data_2, predict_data_3, predict_data_4, predict_data_5, predict_data_6, predict_data_7 = model_1(original_x), model_2(original_x), model_3(original_x), model_4(original_x), model_5(original_x), model_6(original_x), model_7(original_x)

predict_data_9 = model_9(original_x)

# predict_data = (predict_data_1 * 0.2 + predict_data_2 * 0.1 + predict_data_3 * 0.1 + predict_data_4 * 0.2 + predict_data_5 * 0.1 + predict_data_6 * 0.1 + predict_data_7 * 0.2) / 1

predict_data = predict_data_9



predict_data = list(int(i) for i in predict_data)

# predict_data = list(int(i) + int(10*(i-int(i)))*2 if int(10*(i-int(i))) >= 5 else int(i) - int(10*(i-int(i)))*2 for i in predict_data)

print(predict_data)
print('\nVariance {}'.format(np.var(predict_data)))


# 统计性分析
counter = collections.Counter(predict_data)
# 遍历Counter对象的键值对，打印出每个元素及其出现次数
print('\nCount Values')
for item, count in counter.items():
    print(f"{item}: {count}")
print('\n')

# 存进csv输出      
predict_df = pd.read_csv(label_path, encoding='gbk')
predict_df['年龄'] = predict_data
print(predict_df.head(10))
predict_df.to_csv(label_path)

