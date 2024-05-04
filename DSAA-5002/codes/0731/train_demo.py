import warnings
import pandas as pd
import torchmetrics

from torch import optim
from tqdm import tqdm
from colorama import Fore
from CamvidDataset.dataset import *
from Models.UnetPlus import *

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.manual_seed(17)
device = torch.device("cuda")


def train(model, epoch_total, train_loader, valid_loader, loss_func, optimizer, lr_scheduler, device):
    train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=33).to(device)
    valid_acc = torchmetrics.Accuracy(task='multiclass', num_classes=33).to(device)

    for epoch in range(epoch_total):
        train_loss_list = []
        valid_loss_list = []

        train_batch_acc = 0
        train_batch_loss = 0
        valid_batch_acc = 0
        valid_batch_loss = 0

        train_bar = tqdm(train_loader, desc='Train Epoch: ' + str(epoch + 1) + ', Batch Acc: ' + str(
            train_batch_acc) + ', Batch Loss: ' + str(train_batch_loss),
                         bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.GREEN, Fore.RESET))
        model.train()
        for train_data, train_label in train_bar:
            train_data, train_label = train_data.to(device), train_label.to(torch.int64).to(device)
            optimizer.zero_grad()
            train_pred_label = model(train_data)
            train_batch_loss = loss_func(train_pred_label, train_label)
            train_pred_label = train_pred_label.argmax(dim=1)
            train_batch_acc = train_acc(train_pred_label, train_label)
            train_loss_list.append(train_batch_loss.item())
            train_batch_loss.backward()
            optimizer.step()
            tqdm.set_description(train_bar, desc='Train Epoch: ' + str(epoch + 1) + ', Batch Acc: ' + str(
                float('%.5f' % train_batch_acc)) + ', Batch Loss: ' + str(float('%.5f' % train_batch_loss)))

        valid_bar = tqdm(valid_loader, desc='Valid Epoch: ' + str(epoch + 1) + ', Batch Acc: ' + str(
            valid_batch_acc) + ', Batch Loss: ' + str(valid_batch_loss),
                         bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.GREEN, Fore.RESET))
        model.eval()
        with torch.no_grad():
            for valid_data, valid_label in valid_bar:
                valid_data, valid_label = valid_data.to(device), valid_label.to(torch.int64).to(device)
                valid_pred_label = model(valid_data)
                valid_batch_loss = loss_func(valid_pred_label, valid_label)
                valid_pred_label = valid_pred_label.argmax(dim=1)
                valid_batch_acc = valid_acc(valid_pred_label, valid_label)
                valid_loss_list.append(valid_batch_loss.item())
                tqdm.set_description(valid_bar, desc='Valid Epoch: ' + str(epoch + 1) + ', Batch Acc: ' + str(
                    float('%.5f' % valid_batch_acc)) + ', Batch Loss: ' + str(float('%.5f' % valid_batch_loss)))

        train_accuracy = train_acc.compute()
        valid_accuracy = valid_acc.compute()
        train_loss = np.average(train_loss_list)
        valid_loss = np.average(valid_loss_list)

        print('Epoch: ' + str(epoch + 1) + ', Train Acc: ' + str(
            float('%.5f' % train_accuracy) * 100) + ', Train Loss: ' + str(float('%.5f' % train_loss))
              + ', Valid Acc: ' + str(valid_accuracy) + ', Valid Loss: ' + str(valid_loss))

        train_acc.reset()
        valid_acc.reset()

        lr_scheduler.step()

epoch = 20
batch_size = 1
DATA_DIR = 'CamvidDataset'

train_loader, valid_loader = load_dataset(DATA_DIR, batch_size)
model = UnetPlusPlus(num_classes=33).to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1, last_epoch=-1)

train(model, epoch, train_loader, valid_loader, loss_func, optimizer, scheduler, device)
