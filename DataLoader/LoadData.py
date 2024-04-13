import os

import mne
import numpy as np
import scipy
import scipy.io as sio
import torch
from scipy import signal
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class my_dataset(Dataset):
    def __init__(self,data_tensor,label_tensor):
        assert data_tensor.size(0)==label_tensor.size(0)
        self.data_tensor=data_tensor
        self.label_tensor=label_tensor
    def __getitem__(self, index):
        return self.data_tensor[index],self.label_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)

def get_test_EEG_data(sub,data_path):

    test_path = os.path.join(data_path,r'S{}test'.format(sub),'data.mat')
    test_data = sio.loadmat(test_path)
    test_x = test_data['x_data']
    test_y = test_data['y_data']
    test_x,test_y = torch.FloatTensor(test_x),torch.LongTensor(test_y).reshape(-1)
    test_dataset = my_dataset(test_x,test_y)
    return test_dataset

def get_EEG_Hold_Out_Scheme(sub, data_path, validation_size=0.2, data_seed=20140101):
    train_path = os.path.join(data_path, r'S{}train'.format(sub),'data.mat')

    train_data = sio.loadmat(train_path)
    train_x = train_data['x_data']
    train_y = train_data['y_data'].reshape(-1)
    print(train_x.shape, train_y.shape)
    split_train_x, split_validation_x, split_train_y, split_validation_y = train_test_split(train_x, train_y,
                                                                                            test_size=validation_size,
                                                                                            random_state=data_seed,
                                                                                            stratify=train_y)
    label_unique = np.unique(train_y)
    # log the
    for label in label_unique:
        index = (train_y == label)
        label_num = np.sum(index)
        print("class-{}:{}".format(label, label_num))

    train_x, train_y = torch.FloatTensor(train_x), torch.LongTensor(train_y).reshape(-1)
    split_train_x, split_train_y = torch.FloatTensor(split_train_x), torch.LongTensor(split_train_y).reshape(-1)
    split_validation_x, split_validation_y = torch.FloatTensor(split_validation_x), torch.LongTensor(
        split_validation_y).reshape(-1)

    train_dataset = my_dataset(train_x, train_y)
    split_train_dataset = my_dataset(split_train_x, split_train_y)
    split_validation_dataset = my_dataset(split_validation_x, split_validation_y)
    test_dataset = get_test_EEG_data(sub, data_path)
    del train_x, train_y
    return train_dataset, split_train_dataset, split_validation_dataset, test_dataset

def get_other_Subject_EEG_data(sub, data_path, all_session=True, num_of_subject=9):
    '''
        return: two seesions data of one subject
    '''
    first_=True
    for i in range(1,num_of_subject+1):
        if i != sub:
            path = os.path.join(data_path, 'S{}train'.format(i), 'data.mat')
            data = sio.loadmat(path)
            print('loading----train_sub_{}'.format(i))
            if first_:
                first_=False
                data_x = data['x_data']
                data_y = data['y_data'].reshape(-1)
                if all_session:
                    session_2_path = os.path.join(data_path, r'S{}test'.format(i), 'data.mat')
                    session_2_data = sio.loadmat(session_2_path)
                    session_2_x = session_2_data['x_data']
                    session_2_y = session_2_data['y_data'].reshape(-1)
                    data_x = np.concatenate((data_x, session_2_x))
                    data_y = np.concatenate((data_y, session_2_y))


            else:
                data_x_temp = data['x_data']
                data_y_temp = data['y_data'].reshape(-1)
                if all_session:
                    session_2_path = os.path.join(data_path, r'S{}test'.format(i), 'data.mat')
                    session_2_data = sio.loadmat(session_2_path)
                    session_2_x = session_2_data['x_data']
                    session_2_y = session_2_data['y_data'].reshape(-1)
                    data_x_temp = np.concatenate((data_x_temp, session_2_x))
                    data_y_temp = np.concatenate((data_y_temp, session_2_y))
                data_x=np.concatenate((data_x,data_x_temp))
                data_y=np.concatenate((data_y,data_y_temp))

    print(data_x.shape)
    data_x, data_y = torch.FloatTensor(data_x), torch.LongTensor(data_y).reshape(-1)
    return my_dataset(data_x, data_y)

def get_one_Subject_EEG_test_session(sub, data_path):
    '''
        return: get one subject's test session data
    '''


    session_2_path = os.path.join(data_path, r'S{}test'.format(sub), 'data.mat')
    session_2_data = sio.loadmat(session_2_path)
    session_2_x = session_2_data['x_data']
    session_2_y = session_2_data['y_data'].reshape(-1)

    data_x, data_y = torch.FloatTensor(session_2_x), torch.LongTensor(session_2_y).reshape(-1)
    print('test_sub')
    print(data_x.shape)
    return my_dataset(data_x, data_y)


def get_EEG_K_Fold_Scheme(sub, data_path, k=10, validation_size=0.2, data_seed=20240101, all_session=False):
    path = os.path.join(data_path, 'S{}train'.format(sub), 'data.mat')
    data = sio.loadmat(path)

    data_x = data['x_data']
    data_y = data['y_data'].reshape(-1)

    if all_session:
        session_2_path = os.path.join(data_path, r'S{}test'.format(sub), 'data.mat')
        session_2_data = sio.loadmat(session_2_path)
        session_2_x = session_2_data['x_data']
        session_2_y = session_2_data['y_data'].reshape(-1)

        data_x = np.concatenate((data_x, session_2_x))
        data_y = np.concatenate((data_y, session_2_y))

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=data_seed)
    returnlist=[]
    for train_index, test_index in skf.split(data_x, data_y):
        train_x = data_x[train_index]
        train_y = data_y[train_index]
        test_x = data_x[test_index]
        test_y = data_y[test_index]


        split_train_x, split_validation_x, split_train_y, split_validation_y = (
            train_test_split(train_x, train_y,
                             test_size=validation_size,
                             random_state=data_seed,
                             stratify=train_y))

        train_x = torch.FloatTensor(train_x)
        train_y = torch.LongTensor(train_y)
        test_x = torch.FloatTensor(test_x)
        test_y = torch.LongTensor(test_y)
        split_train_x = torch.FloatTensor(split_train_x)
        split_train_y = torch.LongTensor(split_train_y)
        split_validation_x = torch.FloatTensor(split_validation_x)
        split_validation_y = torch.LongTensor(split_validation_y)

        returnlist.append([my_dataset(train_x, train_y),
                my_dataset(split_train_x, split_train_y),
                my_dataset(split_validation_x,split_validation_y),
                my_dataset(test_x, test_y)])

    return returnlist

class Load_IV2a:
    def __init__(self,  *args):
        self.file_to_load = os.path.join(os.path.dirname(os.path.dirname(__file__)),'RawData','BCICIV_2a')
        self.channels_to_remove = ['EOG-left', 'EOG-central', 'EOG-right']
        self.fs = None
        super(Load_IV2a, self).__init__(*args)

    def get_epochs(self, tmin=0, tmax=4.0, baseline=None,reject = False, file_path=None):
        raw_data = self.raw_eeg_subject = mne.io.read_raw_gdf(file_path)
        events, event_ids = mne.events_from_annotations(raw_data)
        self.fs = raw_data.info.get('sfreq')
        if reject == True:
            reject_events = mne.pick_events(events,[1])
            reject_oneset = reject_events[:,0]/self.fs
            duration = [4]*len(reject_events)
            descriptions = ['bad trial']*len(reject_events)
            blink_annot = mne.Annotations(reject_oneset,duration,descriptions)
            raw_data.set_annotations(blink_annot)
        
        stims =[value for key, value in event_ids.items() if key in ['769', '770', '771', '772']]
        epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=tmin, tmax=tmax, event_repeated='drop',
                            baseline=baseline, preload=True, proj=False, reject_by_annotation=True)
        epochs = epochs.drop_channels(self.channels_to_remove)
        self.y_labels = epochs.events[:, -1] - min(epochs.events[:, -1])
        self.x_data = epochs.get_data()*1e6
        # length = len(self.x_data)
        eeg_data={'x_data': self.x_data,
                  'y_labels': self.y_labels,
                  'fs': self.fs
                  }
        return eeg_data

    def get_session2(self, tmin=0, tmax=4.0, baseline=None,file_path=None, lable_path=None):
        raw_data = mne.io.read_raw_gdf(file_path)
        self.fs = raw_data.info.get('sfreq')
        events, event_ids = mne.events_from_annotations(raw_data)
        stims =[value for key, value in event_ids.items() if key in ['783']]
        epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=tmin, tmax=tmax, event_repeated='drop',
                            baseline=baseline, preload=True, proj=False, reject_by_annotation=False)
        epochs = epochs.drop_channels(self.channels_to_remove)
        label_info  = scipy.io.loadmat(lable_path)
        #label_info shape:(288, 1)
        self.y_labels = label_info['classlabel'].reshape(-1) -1
        # print(self.y_labels)
        self.x_data = epochs.get_data()*1e6
        eeg_data={'x_data': self.x_data,
                  'y_labels': self.y_labels,
                  'fs': self.fs}
        return eeg_data

    def build_dataset(self):
        for sub in range(1, 10):
            data_name = r'A0{}T.gdf'.format(sub)
            data = self.get_epochs(tmin=0.0, tmax=4.0,file_path=os.path.join(self.file_to_load, data_name))
            train_x = np.array(data['x_data'])[:, :, :1000]
            train_y = np.array(data['y_labels'])

            data_name = r'A0{}E.gdf'.format(sub)
            label_name = r'A0{}E.mat'.format(sub)
            data = self.get_session2(tmin=0.0, tmax=4.0, file_path=os.path.join(self.file_to_load, data_name),lable_path=os.path.join(self.file_to_load,label_name))
            test_x = np.array(data['x_data'])[:, :, :1000]
            test_y = data['y_labels']

            train_x = np.array(train_x)
            train_y = np.array(train_y).reshape(-1)

            test_x = np.array(test_x)
            test_y = np.array(test_y).reshape(-1)

            print('trian_x:', train_x.shape)
            print('train_y:', train_y.shape)

            print('test_x:', test_x.shape)
            print('test_y:', test_y.shape)

            SAVE_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),'Data','BCIC_2a')

            if not os.path.exists(SAVE_path):
                os.makedirs(SAVE_path)

            SAVE_test = os.path.join(SAVE_path, r'S{}test'.format(sub))
            SAVE_train = os.path.join(SAVE_path, 'S{}train'.format(sub))

            if not os.path.exists(SAVE_test):
                os.makedirs(SAVE_test)
            if not os.path.exists(SAVE_train):
                os.makedirs(SAVE_train)

            scipy.io.savemat(os.path.join(SAVE_train, "data.mat"), {'x_data': train_x, 'y_data': train_y})
            scipy.io.savemat(os.path.join(SAVE_test, "data.mat"), {'x_data': test_x, 'y_data': test_y})
            print('Built successfully!')

class Load_OpenBMI:
    def __init__(self, path: str='None'):
        self.path = path
        self.rootpath = os.path.join(os.path.dirname(os.path.dirname(__file__)),'RawData', 'OpenBMI')
        self.channel_selection=[7, 32, 8, 9, 33, 10, 34, 12, 35, 13, 36, 14, 37, 17, 38, 18, 39, 19, 40, 20]

    def get_session_data(self, session_No: int,subject:int):
        session_name = 'session' + str(session_No)
        print(session_name)
        if subject < 10:
            data_path = os.path.join(self.rootpath, session_name,
                                     'sess0{}_subj0{}_EEG_MI.mat'.format(session_No, subject))
        else:
            data_path = os.path.join(self.rootpath, session_name,
                                     'sess0{}_subj{}_EEG_MI.mat'.format(session_No, subject))

        raw_data = scipy.io.loadmat(data_path)
        x = np.concatenate((raw_data['EEG_MI_train'][0, 0]['smt'], raw_data['EEG_MI_test'][0, 0]['smt']),
                           axis=1).astype(np.float32)
        x = signal.resample(x, 1000)*0.1
        x = x.transpose([1, 2, 0])
        x = x[:,self.channel_selection,:]
        y = np.concatenate(
            (raw_data['EEG_MI_train'][0, 0]['y_dec'].squeeze(), raw_data['EEG_MI_test'][0, 0]['y_dec'].squeeze()),
            axis=0).astype(int) - 1

        del raw_data
        return x, y

    def get_train_in_session_data(self, session_No: int,subject:int):
        session_name = 'session' + str(session_No)
        print(session_name)
        if subject < 10:
            data_path = os.path.join(self.rootpath, session_name,
                                     'sess0{}_subj0{}_EEG_MI.mat'.format(session_No, subject))
        else:
            data_path = os.path.join(self.rootpath, session_name,
                                     'sess0{}_subj{}_EEG_MI.mat'.format(session_No, subject))

        raw_data = scipy.io.loadmat(data_path)
        x = raw_data['EEG_MI_train'][0, 0]['smt'].astype(np.float32)
        x = signal.resample(x, 1000)*0.1
        x = x.transpose([1, 2, 0])
        x = x[:, self.channel_selection, :]
        y = raw_data['EEG_MI_train'][0, 0]['y_dec'].squeeze().astype(int) - 1
        del raw_data
        return x, y

    def get_test_in_session_data(self, session_No: int, subject:int):
        session_name = 'session' + str(session_No)
        print(session_name)
        if subject < 10:
            data_path = os.path.join(self.rootpath, session_name,
                                     'sess0{}_subj0{}_EEG_MI.mat'.format(session_No, subject))
        else:
            data_path = os.path.join(self.rootpath, session_name,
                                     'sess0{}_subj{}_EEG_MI.mat'.format(session_No, subject))

        raw_data = scipy.io.loadmat(data_path)
        x = raw_data['EEG_MI_test'][0, 0]['smt'].astype(np.float32)
        x = signal.resample(x, 1000)*0.1
        x = x.transpose([1, 2, 0])
        x = x[:, self.channel_selection, :]
        y = raw_data['EEG_MI_test'][0, 0]['y_dec'].squeeze().astype(int) - 1
        # c = np.array([m.item() for m in raw_data['EEG_MI_train'][0, 0]['chan'].squeeze().tolist()])
        # s = raw_data['EEG_MI_train'][0, 0]['fs'].squeeze().item()
        del raw_data
        return x, y

    def apply_filters(self, x, fs):
        f0 = 60
        q = 30
        b, a = signal.butter(5, [0.5, 100], btype='bandpass', fs=fs)
        b_notch, a_notch = signal.iirnotch(f0, q, fs=fs)
        x = signal.filtfilt(b, a, x, axis=2)
        x = signal.filtfilt(b_notch, a_notch, x, axis=2)
        return x

    def build_dataset(self):
        fs = 250
        save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),'Data','OpenBMI_cross_session')
        for sub in range(1, 55):
            print('sub{}:begin.'.format(sub))
            save_train_path = os.path.join(save_path, 'S{}train'.format(sub))
            save_test_path = os.path.join(save_path, 'S{}test'.format(sub))

            if not os.path.exists(save_train_path):
                os.makedirs(save_train_path)
            if not os.path.exists(save_test_path):
                os.makedirs(save_test_path)

            train_x, train_y = self.get_session_data(1,sub)
            train_x = self.apply_filters(train_x, fs)
            print(train_x.shape)
            scipy.io.savemat(os.path.join(save_train_path, 'data.mat'), {'x_data': train_x, 'y_data': train_y})
            test_x, test_y = self.get_session_data(2,sub)
            test_x = self.apply_filters(test_x, fs)
            scipy.io.savemat(os.path.join(save_test_path, 'data.mat'), {'x_data': test_x, 'y_data': test_y})
            print(test_x.shape)

        print('Built successfully!')

if __name__ == '__main__':
    loader=Load_IV2a()
    loader.build_dataset()
    loader=Load_OpenBMI()
    loader.build_dataset()