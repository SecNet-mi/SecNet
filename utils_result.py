import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch


sys.path.append('.')


from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from DataLoader.LoadData import get_test_EEG_data
from Models import utils


class Exporter:
    def __init__(self, net_name, randomFolder):
        self.file_path = os.path.join(os.path.abspath(os.path.join(randomFolder, '..')), 'info.txt')
        self.figure_path = os.path.join(os.path.abspath(os.path.join(randomFolder, '..')), 'figures')
        if not os.path.exists(self.figure_path):
            os.makedirs(self.figure_path)

    def export2txt(self, content):
        f = open(self.file_path, mode='a')
        f.write(content)
        f.write('\n')
        f.close()

    def plot_confusion_matrix(self, conf_matrix, subNo):
        plt.figure(figsize=(6, 5), dpi=125)
        # Plot confusion matrix
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=72)
        plt.title('Confusion Matrix')
        plt.colorbar()

        # Add axis labels
        classes = ['left', 'right', 'foot', 'tongue']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, fontsize=12)
        plt.yticks(tick_marks, classes, fontsize=12)

        # Add data labels
        thresh = conf_matrix.max() / 2.
        for i, j in np.ndindex(conf_matrix.shape):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > thresh else "black", fontsize=12)

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        # Save confusion matrix image
        plt.savefig(self.figure_path + '/sub_{}_confusion_matrix.png'.format(subNo))
        plt.close()


def get_results_info_with_dataloader(Net, loader, criterion=torch.nn.CrossEntropyLoss(reduction='sum'), device = 'cuda'):
    cor_num = 0.0
    datalen=0
    loaderlen=0
    all_loss = 0.0
    Net.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            y = y.flatten()
            predict = Net(x)[0]
            loss = criterion(predict, y)
            loss /=x.shape[0]

            cor_num += torch.sum(torch.argmax(predict, dim=1) == y, dtype=torch.float32).item()
            datalen += len(y)
            loaderlen += 1
            all_loss += loss.item()

        mean_loss = all_loss / loaderlen
        mean_accu = cor_num / datalen
    return mean_loss, mean_accu


def get_confusion_matrix(Net, test_dataset,device):
    Net.to(device)
    Net.eval()
    loader = DataLoader(test_dataset)
    ypred = []
    ytrue = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        predict = Net(x)[0]
        predict = predict.detach().cpu().numpy()
        ypred.append(np.argmax(predict, 1))
        ytrue.append(y.flatten().detach().cpu().numpy())
    cm = confusion_matrix(ytrue, ypred)
    return cm


def ho_test(sub: int, data_dir: str, net_name: str, device, Net, model_state_dict_path='Saved_files\\trained_model\\HoldOut'):
    test_dataset = get_test_EEG_data(sub, data_dir)
    path=os.path.join(os.path.dirname(__file__), model_state_dict_path, net_name, net_name + '_sub' + str(sub) + '.pth')
    Net.load_state_dict(torch.load(path,weights_only=True))
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    Net.to(device)
    Net.eval()
    test_dataloader = DataLoader(test_dataset)
    test_loss, test_acc = get_results_info_with_dataloader(Net, test_dataloader, criterion, device)
    cm=get_confusion_matrix(Net,test_dataset,device)
    print(f'Loss:{test_loss},Accuracy:{test_acc}')
    return test_acc,cm
def ho_test_withdataset(sub: int, test_dataset, net_name: str, device, Net, model_state_dict_path='Saved_files\\trained_model\\HoldOut'):
    path=os.path.join(os.path.dirname( __file__ ), model_state_dict_path, net_name,
                                         net_name + '_sub' + str(sub) + '.pth')
    Net.load_state_dict(torch.load(path))
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    Net.to(device)
    Net.eval()

    test_dataloader = DataLoader(test_dataset)
    test_loss, test_acc = get_results_info_with_dataloader(Net, test_dataloader, criterion, device)
    cm=get_confusion_matrix(Net,test_dataset,device)
    print(f'Loss:{test_loss},Accuracy:{test_acc}')
    return test_acc,cm

def ho_test_FB(sub: int, data_dir: str, net_name: str, device, Net, model_state_dict_path='Saved_files\\trained_model\\HoldOut'):
    test_dataset = get_test_EEG_data(sub, data_dir)
    fbf=utils.filterBank()
    test_dataset.data_tensor=fbf(test_dataset.data_tensor)
    path=os.path.join(os.path.dirname( __file__ ), model_state_dict_path, net_name,
                                         net_name + '_sub' + str(sub) + '.pth')
    Net.load_state_dict(torch.load(path))
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    Net.to(device)
    Net.eval()

    test_dataloader = DataLoader(test_dataset)
    test_loss, test_acc = get_results_info_with_dataloader(Net, test_dataloader, criterion, device)
    cm=get_confusion_matrix(Net,test_dataset,device)
    print(f'Loss:{test_loss},Accuracy:{test_acc}')
    return test_acc,cm


def train_one_epoch(Net, dataloader, criterion, optimizer, device):
    Net.train()
    train_loss_sum = 0
    train_acc_sum = 0
    data_len = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        predict = Net(x)[0]
        loss = criterion(predict, y)
        loss = loss / x.shape[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * x.shape[0]
        train_acc_sum += torch.sum(torch.argmax(predict, dim=1) == y, dtype=torch.float32).item()
        data_len += len(y)

    train_loss = train_loss_sum / data_len
    train_acc = train_acc_sum / data_len
    return train_loss, train_acc

