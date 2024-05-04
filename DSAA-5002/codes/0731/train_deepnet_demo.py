import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler


# 构建深度网络模型
class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

def data_preprocess():
    data_path_1 = r'data/train/lh.MeanCurv - 1600.csv'
    data_path_2 = r'data/train/lh.GausCurv - 1600.csv'
    data_path_3 = r'data/train/rh.MeanCurv- 1600.csv'
    data_path_4 = r'data/train/rh.GausCurv- 1600.csv'

    data_path_5 = r'data/train/lh.ThickAvg - 1600.csv'
    data_path_6 = r'data/train/lh.SurfArea - 1600.csv'
    data_path_7 = r'data/train/rh.ThickAvg- 1600.csv'
    data_path_8 = r'data/train/rh.SurfArea - 1600.csv'

    label_path = r'data/train/subject_info - 1600.csv'

    original_x1, original_x2, original_x3, original_x4, original_y = pd.read_csv(data_path_1), pd.read_csv(
        data_path_2), pd.read_csv(data_path_3), pd.read_csv(data_path_4), pd.read_csv(label_path)
    original_x5, original_x6, original_x7, original_x8 = pd.read_csv(data_path_5), pd.read_csv(
        data_path_6), pd.read_csv(data_path_7), pd.read_csv(data_path_8)

    original_x1, original_x2, original_x3, original_x4 = original_x1.iloc[:, 1:], original_x2.iloc[:, 1:], original_x3.iloc[:, 1:], original_x4.iloc[:, 1:]
    original_x5, original_x6, original_x7, original_x8 = original_x5.iloc[:, 1:], original_x6.iloc[:, 1:], original_x7.iloc[:, 1:], original_x8.iloc[:, 1:]
    original_y = original_y.iloc[:, 3]

    original_x1 = (original_x1 - original_x1.min()) / (original_x1.max() - original_x1.min())
    original_x2 = (original_x2 - original_x2.min()) / (original_x2.max() - original_x2.min())
    original_x3 = (original_x3 - original_x3.min()) / (original_x3.max() - original_x3.min())
    original_x4 = (original_x4 - original_x4.min()) / (original_x4.max() - original_x4.min())
    # original_x5 = (original_x5 - original_x5.min()) / (original_x5.max() - original_x5.min())
    # original_x6 = (original_x6 - original_x6.min()) / (original_x6.max() - original_x6.min())
    # original_x7 = (original_x7 - original_x7.min()) / (original_x7.max() - original_x7.min())
    # original_x8 = (original_x8 - original_x8.min()) / (original_x8.max() - original_x8.min())

    # # PCA
    # pca_fitter = PCA(n_components=10)
    # original_x1, original_x2, original_x3, original_x4 = pca_fitter.fit_transform(original_x1), pca_fitter.fit_transform(original_x2), pca_fitter.fit_transform(original_x3), pca_fitter.fit_transform(original_x4)
    # original_x5, original_x6, original_x7, original_x8 = pca_fitter.fit_transform(
    #     original_x5), pca_fitter.fit_transform(original_x6), pca_fitter.fit_transform(
    #     original_x7), pca_fitter.fit_transform(original_x8)

    #
    # # SELECT-K-BEST
    # k = 10  # 选择最相关的特征数量
    # selector = SelectKBest(f_classif, k=k)
    # original_x1, original_x2, original_x3, original_x4 = selector.fit_transform(original_x1, original_y), selector.fit_transform(original_x2, original_y), selector.fit_transform(original_x3, original_y), selector.fit_transform(original_x4, original_y)


    original_x1, original_x2, original_x3, original_x4 = np.array(original_x1), np.array(original_x2), np.array(original_x3), np.array(original_x4)
    original_x5, original_x6, original_x7, original_x8 = np.array(original_x5), np.array(original_x6), np.array(original_x7), np.array(original_x8)
    original_y = np.array(original_y)

    # original_x = np.concatenate([original_x1 + original_x3, original_x2 + original_x4], axis=1)
    original_x = (original_x5 + original_x7) * 0.7

    return original_x, original_y


def main():
    original_x, original_y = data_preprocess()
    print(original_x.shape, original_y.shape)

    # 将数据集分为训练集和测试集
    X_train, X_valid, y_train, y_valid = train_test_split(original_x, original_y, test_size=0.3, random_state=0)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_valid = torch.tensor(X_train, dtype=torch.float32)
    y_valid = torch.tensor(y_train, dtype=torch.float32)
    train_dataset = TensorDataset(X_train, y_train)
    valid_dataset = TensorDataset(X_valid, y_valid)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=True)

    # 定义输入数据的维度
    input_dim = X_train.shape[1]
    total_epoch = 200
    model = Net(input_dim)
    criterion = nn.L1Loss()  # MAE Loss
    optimizer = optim.SGD(model.parameters(), lr=0.00005, momentum=0.95)  # 使用Adam优化器进行参数更新

    # 对训练数据进行拟合
    for epoch in range(total_epoch):  # 迭代次数为100轮
        train_loss = 0
        for i, (inputs, labels) in enumerate(train_dataloader):  # 遍历每个batch的数据和标签
            optimizer.zero_grad()  # 清空梯度缓存区以便进行反向传播计算梯度。同时需要注意的是，在实际应用中需要对训练数据进行相同的预处理操作。
            outputs = model(inputs)  # 通过模型进行预测得到输出结果
            loss = criterion(outputs, labels)  # 计算预测结果与真实结果之间的均方误差损失函数值。同时需要注意的是，在实际应用中需要对训练数据进行相同的预处理操作。
            loss.backward()  # 对损失函数进行反向传播计算梯度。同时需要注意的是，在实际应用中需要对训练数据进行相同的预处理操作。
            optimizer.step()  # 根据梯度下降算法更新参数。同时需要注意的是，在实际应用中需要对训练数据进行相同的预处理操作。
            train_loss += loss.item()

        # if epoch > 15:
        #     print(outputs)
        print('Epoch [{}/{}], Train Loss: {:.4f}'.format(epoch + 1, total_epoch, train_loss / (i + 1)))  # 这里采用了print函数来打印损失函数值。同时需要注意的是，在实际应用中需要对训练数据进行相同的预处理操作。

        model.eval()
        with torch.no_grad():
            valid_loss = 0
            for i, (valid_inputs, valid_labels) in enumerate(valid_dataloader):
                valid_outputs = model(valid_inputs)
                valid_loss_item = criterion(valid_outputs, valid_labels)
                valid_loss += valid_loss_item

            print('Epoch [{}/{}], Valid Loss: {:.4f}'.format(epoch + 1, total_epoch, valid_loss / (i + 1)))

        # 保存模型
        torch.save(model, '5_fc_layer_model-x5x7-avg0_7-epoch200-lr0_00005.pkl')


# main()