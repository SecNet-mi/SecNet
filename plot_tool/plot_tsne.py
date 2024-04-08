import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import os
import torch
import sys
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from DataLoader.LoadData import get_test_EEG_data
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Models import Nets as Network
from matplotlib.patches import Ellipse
from lightning.pytorch import seed_everything
device='cpu'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
seed_everything(42, workers=True)
torch.use_deterministic_algorithms(True)


def draw_ellipse(position, covariance, ax=None, color=None,**kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    # for nsig in range(1, 4):
    nsig=2
    ax.add_patch(Ellipse(position, width=nsig * width, height=nsig * height,
                         angle=angle,alpha=0.2,color=color))

def plot_tsne_(Net, dataset,subNo, save_path):
    Net.to('cpu')
    Net.eval()
    output_features = []
    output_y = []
    loader=DataLoader(dataset,batch_size=1)
    for x, y in loader:
        features = Net(x)[1]
        output_features.append(features.flatten(1).clone().detach().numpy())
        output_y.append(y.clone().detach().numpy().reshape(-1))

    feature_dim = features.size(-1)

    X = np.array(output_features).reshape(-1, feature_dim)
    y = np.array(output_y).reshape(-1)

    tsne = TSNE(n_components=2, random_state=2)
    X_tsne = tsne.fit_transform(X)

    # 获取不同类别的索引
    unique_classes = np.unique(y)
    label_names = {0: 'Left', 1: 'Right', 2: 'Foot', 3: 'Tongue'}
    # 绘制t-SNE降维后的结果，不同类别使用不同颜色
    plt.figure(figsize=(6, 4))
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    # colors = cm.rainbow(np.linspace(0, 1, len(label_names)))
    colors = [cm.Set1(i) for i in range(4)]
    # colors = cm.Paired(np.linspace(0, 1, len(label_names)))
    # colors = ['#FF0000', '#00FF00', '#0000FF', '#EFD500']
    for cls, color in zip(unique_classes, colors):
        idx = np.where(y == cls)
        plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=label_names[cls],color=color, s=50,alpha=0.9,marker='o')
        class_data = X_tsne[y == cls]
        mean = np.mean(class_data, axis=0)
        cov = np.cov(class_data, rowvar=False)
        # draw_ellipse(mean, cov, alpha=0.1, color=color)

    plt.xticks([])
    plt.yticks([])

    plt.title('t-SNE of Subject {}'.format(subNo), fontsize=26)
    plt.subplots_adjust(top=0.88, bottom=0.05, left=0.05, right=0.80, hspace=0.1, wspace=0.1)
    plt.grid(False)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = save_path + '/tsne_org_s{}.png'
    plt.savefig(save_path.format(subNo), dpi=400)
    plt.close()


class visualization:
    def __init__(self,args):
        self.root_path=os.path.dirname(os.path.dirname(__file__))
        self.legend=args.legend


    def turn_on_plot_legend(self):
        self.legend=True
    def ouput_features_for_subject(self, sub:int):
        test_dataset = get_test_EEG_data(sub,data_path=os.path.join(self.root_path, 'Data',args.data_name))
        path = os.path.join(self.root_path, 'Saved_files/save/models', args.net_dirname,
                            args.net_name + '_sub' + str(sub) + '.pth')
        module = Network.__dict__[args.net_name]
        Net = module(configs).to('cpu')
        Net.load_state_dict(torch.load(path))
        Net.to('cpu')
        Net.eval()
        loader = DataLoader(test_dataset, batch_size=1)
        output_features=[]
        output_y=[]
        for x, y in loader:
            _, features = Net(x)
            output_features.append(features.flatten(1).clone().detach().numpy())
            output_y.append(y.clone().detach().numpy().reshape(-1))

        feature_dim=features.size(-1)
        output_x=np.array(output_features).reshape(-1,feature_dim)
        output_y=np.array(output_y).reshape(-1)
        return output_x, output_y


    def plot_tsne_one_subject(self,subNo:int=1):
        X,y=self.ouput_features_for_subject(subNo)
        tsne = TSNE(n_components=2, random_state=42,perplexity=20,max_iter=3000)
        X_tsne = tsne.fit_transform(X)

        # 获取不同类别的索引
        unique_classes = np.unique(y)
        label_names={0:'Left', 1:'Right', 2:'Foot', 3:'Tongue'}
        # 绘制t-SNE降维后的结果，不同类别使用不同颜色
        plt.figure(figsize=(6, 4))
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        # colors = cm.Set1(np.linspace(0, 1, len(label_names)))
        # colors = cm.Paired(np.linspace(0, 1, len(label_names)))
        # colors = ['#FF0000','#00FF00','#0000FF','#EFD500']
        colors=[cm.Set1(i) for i in range(4)]
        for cls, color in zip(unique_classes,colors):
            idx = np.where(y == cls)
            plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=label_names[cls],color=color, s=50,alpha=0.9,marker='o')
            class_data = X_tsne[y == cls]
            mean = np.mean(class_data, axis=0)
            cov = np.cov(class_data, rowvar=False)
            # draw_ellipse(mean, cov, alpha=0.1, color=color)

        plt.xticks([])
        plt.yticks([])
        # plt.xlabel('t-SNE Feature 1',fontsize=20)
        # plt.ylabel('t-SNE Feature 2',fontsize=20)
        plt.title(args.title.format(subNo),fontsize=26)
        plt.subplots_adjust(top=0.88,bottom=0.05,left=0.05,right=0.80,hspace=0.1,wspace=0.1)
        plt.grid(False)
        if self.legend:
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1),fontsize=12)

        save_path = os.path.join(os.path.dirname(__file__),'save_figs', 'TSNE')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_path=save_path+ os.path.join(os.path.dirname(__file__),'tsne_org_s{}.png')
        plt.savefig(save_path.format(subNo), dpi=400)
        plt.close()

    def plot_tsne_for_all_subjects(self,sublist):
        X=None
        y=None
        for i in sublist:
            if X is None:
                X, y = self.ouput_features_for_subject(i)
            else:
                X=np.concatenate((X,X),axis=0)
                y=np.concatenate((y,y),axis=0)
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X)

        # 获取不同类别的索引
        unique_classes = np.unique(y)
        label_names={0:'Left', 1:'Right', 2:'Foot', 3:'Tongue'}
        # 绘制t-SNE降维后的结果，不同类别使用不同颜色
        plt.figure(figsize=(8, 4))
        for cls in unique_classes:
            idx = np.where(y == cls)
            plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=label_names[cls])
        plt.xticks([])
        plt.yticks([])
        # plt.xlabel('t-SNE Feature 1',fontsize=20)
        # plt.ylabel('t-SNE Feature 2',fontsize=20)
        plt.title('t-SNE Visualization of Features',fontsize=24)
        plt.grid(False)
        if self.legend:
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    def plot_SPD_matrix(self,sub_No:int=1):
        size_n=100
        X, y = self.ouput_features_for_subject(sub_No)
        temp_xi=np.zeros((size_n,size_n))
        row_idx, col_idx = np.triu_indices(size_n)
        classSPD=np.zeros((4,100,100))
        for i, xi in enumerate(X):
            temp_xi[row_idx,col_idx]=xi
            diag_of_X=np.diag(temp_xi)
            xi_e=temp_xi+temp_xi.T-np.diag(diag_of_X)
            classSPD[y[i]]+=xi_e
        num=220
        for i in range(4):
            plt.subplot(221+i)
            plt.imshow(classSPD[i])
        plt.show()





if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--savefig',type=bool, default=True)
    parser.add_argument('--title',type=str, default='t-SNE of Subject {}' )
    parser.add_argument('--net_name',type=str, default='basemodel')
    parser.add_argument('--net_dirname',type=str, default='basemodel')
    parser.add_argument('--data_name',type=str, default='BCIC_2a')
    parser.add_argument('--legend',type=bool, default=False)
    parser.add_argument('--width',type=int, default=350)
    parser.add_argument("--in_channel", type=int, default=100)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--fusionMode", type=str, default='concat')
    args=parser.parse_args()
    configs=vars(args)


    visual=visualization(args)
    for sn in range(1,10):
        # if sn==9:
        visual.turn_on_plot_legend()

        visual.plot_tsne_one_subject(sn)
