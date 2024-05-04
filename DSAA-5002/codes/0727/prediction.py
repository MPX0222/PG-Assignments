import torch
import numpy as np
import pandas as pd
from train_deepnet_demo import Net
from torch.utils.data import DataLoader, TensorDataset

from sklearn.decomposition import PCA

data_path_1 = r'data/test/lh.ThickAvg- 389.csv'
data_path_2 = r'data/test/rh.ThickAvg- 389.csv'
label_path = r'subject_info - 389.csv'

# 处理测试集数据
data_x_1 = pd.read_csv(data_path_1)
data_x_2 = pd.read_csv(data_path_2)
original_x1, original_x2 = data_x_1.iloc[:, 1:], data_x_2.iloc[:, 1:]
original_x1, original_x2 = np.array(original_x1), np.array(original_x2)

original_x = (original_x1 + original_x2) * 0.5

# pca_fitter = PCA(n_components=10)
# original_x = pca_fitter.fit_transform(original_x)

original_x = torch.tensor(original_x, dtype=torch.float32)

model_1 = torch.load('5_fc_layer_model-x5x7-avg0_4-epoch200-lr0_00005.pkl')
model_2 = torch.load('5_fc_layer_model-x5x7-avg0_5-epoch200-lr0_00005.pkl')
model_3 = torch.load('5_fc_layer_model-x5x7-avg0_6-epoch200-lr0_00005.pkl')
model_4 = torch.load('5_fc_layer_model-x5x7-avg0_3-epoch200-lr0_00005.pkl')
model_5 = torch.load('5_fc_layer_model-x5x7-avg0_8-epoch200-lr0_00005.pkl')
predict_data_1, predict_data_2, predict_data_3, predict_data_4, predict_data_5 = model_1(original_x), model_2(original_x), model_3(original_x), model_4(original_x), model_5(original_x)
predict_data = (predict_data_1 * 0.2 + predict_data_2 * 0.2 + predict_data_3 * 0.2 + predict_data_4 * 0.2 + predict_data_5 * 0.2) / 1
print(predict_data)

predict_data = list(int(i) for i in predict_data)

# predict_data = list(int(i) + int(10*(i-int(i)))*2 if int(10*(i-int(i))) >= 5 else int(i) - int(10*(i-int(i)))*2 for i in predict_data)

print(predict_data)

predict_df = pd.read_csv(label_path)
predict_df['年龄'] = predict_data
print(predict_df.head(10))
predict_df.to_csv(label_path)

