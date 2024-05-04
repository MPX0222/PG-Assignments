import numpy as np
from transformers import BertModel, BertTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd

import torch
import torch.nn as nn

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print('Using CPU')

class BertRegressor(nn.Module):
    def __init__(self, num_labels, pretrained_model_name='bert-base-cased'):
        super(BertRegressor, self).__init__()
        # 加载预训练的BERT模型和分词器
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        # 设置输出层为线性回归层，输入维度为768(BERT隐藏层大小),输出维度为num_labels
        self.fc = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        # 将输入数据传入BERT模型，得到每个token的嵌入表示
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 取最后一层的隐藏状态作为整个序列的表示
        last_hidden_state = outputs[0]
        # 通过全连接层得到最终的回归结果
        logits = self.fc(last_hidden_state)
        return logits

def train_bert():
    # 加载数据集
    data_path = r'data/train/rh.GausCurv- 1600.csv'
    label_path = r'data/train/subject_info - 1600.csv'

    original_x, original_y = pd.read_csv(data_path), pd.read_csv(label_path)
    original_x = original_x.iloc[:, 1:]
    original_y = original_y.iloc[:, 3]

    original_x = original_x.values.tolist()
    original_y = original_y.values.tolist()

    # original_x = np.array(original_x)
    # original_y = np.array(original_y)
    # original_x = torch.from_numpy(original_x)
    # original_y = torch.from_numpy(original_y)
    #
    # original_x = original_x.to(device)
    # original_y = original_y.to(device)
    print(original_x, original_y)

    # 将数据集分为训练集和测试集
    X_train, X_valid, y_train, y_valid = train_test_split(original_x, original_y, test_size=0.2, random_state=0)


    # 初始化tokenizer和模型
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = BertRegressor(num_labels=1) # 假设输出为一个数值型结果，因此 num_labels=1
    model = model.to(device)

    # 将文本转换成BERT输入格式
    X_train = [tokenizer.encode(text, add_special_tokens=True) for text in X_train]
    X_test = [tokenizer.encode(text, add_special_tokens=True) for text in X_valid]

    # 将数据传入模型进行训练
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    for epoch in range(100): # 训练3个epochs
        optimizer.zero_grad()
        input_data = torch.tensor(X_train).to(device)
        logits = model(input_data, input_data) # 传入训练集数据作为input和attention mask参数
        loss = nn.MSELoss()(logits, torch.tensor(y_train)) # 使用均方误差损失函数计算损失值
        loss.backward() # 反向传播计算梯度
        optimizer.step() # 更新模型参数
        print('Epoch {}: Loss={}'.format(epoch+1, loss.item())) # 打印当前epoch和损失值

train_bert()