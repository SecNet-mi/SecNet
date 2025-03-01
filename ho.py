import os.path
import random
import argparse

import time,copy

from DataLoader.LoadData import *
from Models import utils
from DataLoader.LoadData import get_EEG_Hold_Out_Scheme
from Models import Nets
from Models.optimizer.modified_optimizer import customAdam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.tensorboard import SummaryWriter
from plot_tool.plot_tsne import plot_tsne_
from utils_result import  get_results_info_with_dataloader, Exporter, ho_test_FB, ho_test_withdataset,train_one_epoch
from lightning.pytorch import seed_everything

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
seed_everything(42, workers=True)
torch.use_deterministic_algorithms(True)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(42)

def verify_model(Net,test_dataloader,criterion,device,epoch,train_loss,train_acc,save_logs,writer):
    test_loss, test_acc = get_results_info_with_dataloader(Net, test_dataloader, criterion, device=device)
    print(f'Epoch:{epoch:d}__train_loss:{train_loss:.5f}__train_accracy:{train_acc:.5f}__'
          f'test_accracy:{test_acc:.5f}')
    Net.eval()
    if save_logs:
        writer.add_scalars('train_loss', {'train_loss': train_loss}, epoch)
        writer.add_scalars('test acc', {'test_acc': test_acc}, epoch)

def get_combinations(modules=[]):
    def drawback(startIndex,path,mods,results):
        results.append(path.copy())
        for i in range(startIndex,len(mods)):
            path[mods[i]]=True
            drawback(i+1,path,mods,results)
            path[mods[i]]=False

    path={module: False for module in modules}
    results=[]
    drawback(0,path,modules,results)
    return results

class runner:
    def __init__(self,randomFolder, configs):
        self.randomFolder=randomFolder
        self.net_name=configs['net_name']
        self.data_dir_name=configs['data_dir_name']
        self.data_dir_path = os.path.join(os.path.dirname(__file__),'Data', self.data_dir_name)
        self.lr=configs['lr']
        self.device=configs['device']
        self.beta2=configs['betatwo']
        self.scheduler=configs['scheduler']
        self.Net = None
        self.NumOfEpoch=configs['NumOfEpoch']
        self.batch_size=configs['batchsize']
        self.earlyStopStrategy=configs['earlystop']
        self.save_logs=configs['save_logs']
        self.fs=configs['fs']
        self.lambdavalue=configs['lambdavalue']
        self.test_acc_vs_epoch=[]

    def trans(self,data):

        if self.net_name=='FBCNet' or self.net_name=='FBMSNet':
            process=utils.filterBank([[4,8],[8,12],[12,16],[16,20],[20,24],[24,28],[28,32],[32,36],[36,40]],fs=self.fs,axis=-1)
            return process(data)
        else:
            return data

    def ho_train(self, sub):


        data_seed = 20230101

        save_path = os.path.join(os.path.dirname(__file__),
                                 'Saved_files', 'trained_model', 'HoldOut', self.net_name)

        file_name = f'{self.net_name}_sub{sub}.pth'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        train_dataset, split_train_dataset, split_validation_dataset, test_dataset = (
            get_EEG_Hold_Out_Scheme(sub,self.data_dir_path,configs['validation_ratio'], data_seed))
        train_dataset.data_tensor=self.trans(train_dataset.data_tensor)
        split_train_dataset.data_tensor=self.trans(split_train_dataset.data_tensor)
        split_validation_dataset.data_tensor=self.trans(split_validation_dataset.data_tensor)
        test_dataset.data_tensor=self.trans(test_dataset.data_tensor)
        self.test_dataset=test_dataset

        train_len, split_train_len, split_validation_len, test_dataset_len = len(train_dataset), len(
            split_train_dataset), len(split_validation_dataset), len(test_dataset)

        print(train_len, split_train_len, split_validation_len, test_dataset_len)
        start_time=time.time()
        split_train_dataloader = DataLoader(split_train_dataset, batch_size=self.batch_size, shuffle=True,pin_memory=True,worker_init_fn=seed_worker)
        split_validation_dataloader = DataLoader(split_validation_dataset, batch_size=self.batch_size, shuffle=True,pin_memory=True,worker_init_fn=seed_worker)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,pin_memory=True,worker_init_fn=seed_worker)
        test_dataloader = DataLoader(test_dataset, batch_size=10000, shuffle=True,pin_memory=True,worker_init_fn=seed_worker)
        criterion = torch.nn.CrossEntropyLoss(reduction='sum').to(self.device)  # mean or sum  no effect too much

        Net = Nets.__dict__[self.net_name](configs=configs).to(self.device)
        print(Net)
        optimizer = customAdam(Net.parameters(), lr=self.lr)


        if self.scheduler:
            scheduler = CyclicLR(optimizer, self.lr*0.8, self.lr * 3, 10, 10, cycle_momentum=False)
        else:
            print('No Scheduler!')


        if self.save_logs:
            writer = SummaryWriter(self.randomFolder + str(sub))

        remaining_epoch = self.NumOfEpoch
        best_accu = 0
        best_validation_loss=float('inf')
        mini_loss = None
        if self.earlyStopStrategy:
            # First step
            print(f'Sub {sub} strats trainning!')
            for epoch in range(1500):
                split_train_loss, split_train_acc = train_one_epoch(Net, split_train_dataloader, criterion, optimizer, self.device)

                if self.scheduler:
                    scheduler.step()

                Net.eval()


                split_validation_loss, split_validation_accu = get_results_info_with_dataloader(Net, split_validation_dataloader,
                                                                                criterion,device=self.device)
                test_loss, test_acc = get_results_info_with_dataloader(Net, test_dataloader, criterion,device=self.device)
                remaining_epoch = remaining_epoch - 1

                print(
                    f'Epoch:{epoch}__train_loss:{split_train_loss}__train_accracy:{split_train_acc}__validation_loss:{split_validation_loss}__validation_accuracy:{split_validation_accu}__best_accuracy:{best_accu}__remaining_epoch:{remaining_epoch}__test_accracy:{test_acc}'
                )

                if remaining_epoch <= 0:
                    break
                if mini_loss is None or split_train_loss < mini_loss:
                    mini_loss = split_train_loss

                if split_validation_accu > best_accu:
                    best_Net = copy.deepcopy(Net.state_dict())
                    optimizer_state = copy.deepcopy(optimizer.state_dict())
                    remaining_epoch = self.NumOfEpoch
                    best_accu = split_validation_accu

            print('Earyly stopping,and retrain the Net using both the training data and validation data.')

            Net.load_state_dict(best_Net)
            optimizer.load_state_dict(optimizer_state)

            if self.scheduler:
                scheduler=CyclicLR(optimizer,self.lr*0.8,self.lr*3,10,10,cycle_momentum=False)
            else:
                print('No Scheduler!')

            for epoch in range(1000):
                train_loss, train_acc = train_one_epoch(Net, split_train_dataloader, criterion, optimizer, self.device)
                split_validation_loss, split_validation_accu = train_one_epoch(Net, split_validation_dataloader, criterion, optimizer, self.device)

                if self.scheduler:
                    scheduler.step()
                    print(scheduler.get_last_lr())

                Net.eval()


                test_loss, test_acc = get_results_info_with_dataloader(Net, test_dataloader, criterion,device=self.device)

                print(f'Epoch:{epoch}__train_loss:{train_loss}__train_accracy:{train_acc}__validation_loss:{split_validation_loss}__'+
                      f'validation_accuracy:{split_validation_accu}__test_accracy:{test_acc}')

                if split_validation_loss < mini_loss:
                    break
                if self.save_logs:
                    writer.add_scalars('train_loss', {'train_loss': train_loss}, epoch)
                    writer.add_scalars('test acc', {'test_acc': test_acc}, epoch)
                    # Run in the test data.
            Net.eval()
            test_loss, test_acc = get_results_info_with_dataloader(Net, test_dataloader, criterion,device=self.device)

            print(f'sub:{sub}--loss:{test_loss}--acc:{test_acc}')

            # Save the model.
            print(file_name)
            torch.save(Net.state_dict(), os.path.join(save_path, file_name))
            print('The model was saved successfully!')
        else:
            epoch = 0
            end_epoch = self.NumOfEpoch
            while True:
                train_loss, train_acc = train_one_epoch(Net, train_dataloader, criterion, optimizer, self.device)
                if self.scheduler:
                    scheduler.step()
                    print(scheduler.get_last_lr())
                verify_model(Net, test_dataloader, criterion, self.device, epoch, train_loss, train_acc, self.save_logs, writer)
                epoch += 1
                if epoch >= end_epoch:
                    break

            #  ------------Run in the test data. ----------------
            Net.eval()
            test_loss, test_acc = get_results_info_with_dataloader(Net, test_dataloader, criterion,device=self.device)

            print(f'sub:{sub}--loss:{test_loss}--acc:{test_acc}')

        # Save the model.
        self.Net = Net
        print(file_name)
        torch.save(Net.state_dict(), os.path.join(save_path, file_name))
        print('The model was saved successfully!')
        del train_dataset
        consume_time=time.time()-start_time
        print(f'training consumes time is:{consume_time}')

        return test_acc


    def run(self):
        exporter=Exporter(self.net_name,self.randomFolder)

        all_accu = []
        subjectNum = {
            'BCIC_2a': 9,
            'OpenBMI_cross_session': 54,
        }.get(self.data_dir_name, 0)

        print('net_name:\t'+self.net_name)
        print(f'lr:{self.lr}')
        print('data_dir_name: '+ self.data_dir_name)
        print(subjectNum)

        for i in range(1,subjectNum+1):
            print(f'subject: {i}')
            self.ho_train(sub=i)
            if self.net_name=='FBCNet':
                acc, cm = ho_test_FB(i, self.data_dir_path, self.net_name, self.device, self.Net)
            else:
                acc, cm= ho_test_withdataset(i, self.test_dataset, self.net_name,self.device, self.Net)
            all_accu.append(acc)

            exporter.plot_confusion_matrix(conf_matrix=cm,subNo=i)
            exporter.export2txt(f'subject{i}, Average accuracy:{acc}')
            exporter.export2txt(f'subject{i}, confusion matrix:\n{cm} \n')
            plot_tsne_(self.Net, self.test_dataset, i, os.path.dirname(self.randomFolder)+'/t-sne-figs')

        title= 'results-'+self.data_dir_name
        content =f'All_accu:{all_accu}\n Average accuracy:{sum(all_accu)/len(all_accu)}'
        print(content)
        exporter.export2txt(content)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--dataNo", type=int, default=1)
    parser.add_argument("--scheduler", type=bool, default=False)
    parser.add_argument("--earlystop", type=bool, default=False)
    parser.add_argument("--NumOfEpoch", type=int, default=250)
    parser.add_argument("--batchsize", default=16)
    parser.add_argument("--betatwo", type=float, default=0.999)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--width", type=int, default=300)
    parser.add_argument("--in_channel", type=int, default=100)
    parser.add_argument("--save_logs", type=bool, default=True)
    parser.add_argument("--fs", type=int, default=250)
    parser.add_argument("--window_len", type=int, default=100)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--net_name", type=str, default='bciiv2a')
    parser.add_argument("--identityflag", type=bool, default=True)
    parser.add_argument("--validation_ratio", type=float, default=0.2) # we did not use this setting
    parser.add_argument("--lambdavalue", type=float, default=0.0005)
    parser.add_argument("--drop_att", type=float, default=0.2)
    parser.add_argument("--data_dir_name", type=str, default='BCIC_2a')
    parser.add_argument("--branch", type=int, default=3)
    parser.add_argument("--isAttention", type=bool, default=True)
    parser.add_argument("--isGaussianNoise", type=bool, default=False)
    parser.add_argument("--isNormal", type=bool, default=True)
    parser.add_argument("--isSum", type=bool, default=False)
    parser.add_argument("--issplitQKV", type=bool, default=True)
    parser.add_argument("--lam", type=float, default=0.001)
    parser.add_argument("--ksize", type=int, default=5)
    parser.add_argument("--scmdim", type=int, default=100)
    parser.add_argument("--p", type=int, default=3)
    args = parser.parse_args()
    configs=vars(args)
    configs['device']= 'cuda' if torch.cuda.is_available() else 'cpu'
    subtitle = 'SecNet'
    randomFoldert = '_' + str(time.strftime("%Y-%m-%d--%H-%M", time.localtime())) + '-' + str(random.randint(1, 1000))
    randomFolder = os.path.join(os.path.dirname(__file__), 'Saved_files', 'LOG',
                                subtitle + configs['net_name'] + randomFoldert + '/log_s')
    runner_ = runner(randomFolder, configs)
    runner_.run()


