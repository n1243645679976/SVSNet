import torch
import os
FEAT_PATH = '../../feature/sim_net/wav'
LIST_PATH = '../../feature/sim_net/'
MODEL_PATH = './logdir/'

testing_list = 'sim_list_test.txt'
training_list = 'sim_list_train.txt'
# FEAT_PATH = '../../feature/VCC20/VCC2020-listeningtest/'
# LIST_PATH = '../../dataset/VCC20/VCC2020-listeningtest-info/VCC202-listeningtest-scores/vcc20_all.txt'

class Dataset(torch.utils.data.Dataset):
    def __init__(self, testing, feat_path, test_list, train_list, device):
        self.X = {}
        self.device = device
        self.feat_path = feat_path

        if not testing:
            self.train = True
            self.train_list = self.load_list(train_list)
            self.test_list = self.load_list(test_list)
        else:
            self.train = False
            self.test_list = self.load_list(test_list)

    def __len__(self):
        if self.train:
            return len(self.train_list)
        else:
            return len(self.test_list)

    def load_list(self, list_name):
        with open(os.path.join(list_name)) as f:
            file_list = f.read().splitlines()
            for i in range(len(file_list)):
                x1, x2, _, _ = file_list[i].split(',')
                if x1.split('.')[0] not in self.X:
                    self.X[x1.split('.')[0]] = torch.load(os.path.join(self.feat_path, x1.split('.')[0]+'.wav.pt'))['wav'].to(self.device).reshape(1,-1)
                if x2.split('.')[0] not in self.X:
                    self.X[x2.split('.')[0]] = torch.load(os.path.join(self.feat_path, x2.split('.')[0]+'.wav.pt'))['wav'].to(self.device).reshape(1,-1)

    def change_mode(self, str, ep):
        if str == 'train':
            self.train=True
        elif str == 'test':
            self.train=False

    def __getitem__(self, index):
        if self.train:
            x1, x2, _, similarity = self.train_list[index].split(',')
        else:
            x1, x2, _, similarity = self.test_list[index].split(',')
        
        X1 = self.X[x1.split('.')[0]].to(self.device)
        X2 = self.X[x2.split('.')[0]].to(self.device)
        # if 'S00' in x1 or 'T00' in x1:
        #     r = x2 + ' ' + x1
        # else:
        #     r = x1 + ' ' + x2
        return (X1, X2), torch.tensor([int(similarity)-1]).to(self.device), x1 + ' ' + x2
