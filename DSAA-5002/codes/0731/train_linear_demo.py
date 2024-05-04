import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


data_path = r'data/train/wmparc - 1600.csv'
label_path = r'data/train/subject_info - 1600.csv'
original_x, original_y = pd.read_csv(data_path), pd.read_csv(label_path)
original_x = original_x.iloc[:, 1:]
original_y = original_y.iloc[:, 3]

original_x = np.array(original_x)
original_y = np.array(original_y)
print(original_x.shape, original_y.shape)

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(original_x, original_y, test_size=0.2, random_state=0)

# 创建线性回归模型并拟合训练数据
model = LinearRegression()
model.fit(X_train, y_train)

# 对测试集进行预测并计算均方误差
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)