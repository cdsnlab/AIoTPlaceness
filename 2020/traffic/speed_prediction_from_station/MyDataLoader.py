import pandas as pd
import numpy as np
import torch
import torch.utils.data as data_utils 
import torch.utils.data as Data
import datetime

def MyDataLoader(dataset):
    if dataset == 'S':
        return MyDataLoaderS()
    if dataset == 'G':
        return MyDataLoaderG()
    return None


class MyDataLoaderS():
    def __init__(self, filename='../../dataprocess/sungsu_dataset_t16_s106_20210328.npz',
                        train_start='2015-01-01',
                        test_start='2019-01-01', 
                        test_end='2020-01-01',
                        shuffle_mode=False,
                        random_seed=42,
                        val_ratio=0.1): 
        self.data = np.load(filename)
        print('sungsu_dataset_loaded')
        
        fdf =pd.read_csv("../../dependency/node2vec-GMAN/SE_sungsu_directed_100_80_64.txt", sep=' ', skiprows=1, header=None)
        self.N2V = fdf.sort_values(0).set_index(0).values
        
        self.DAs = self.data['DAs']
        self.Ms = np.load('../../dataprocess/Mstatic_sungsu.npy')
        self.Mv = np.load('../../dataprocess/nM50_sungsu_road_features.npy')

        self.holiday_df = pd.read_csv('../../holiday/national_holidays.csv', index_col=0)
        
        # tr: train set, 
        # te: test set
        self.X = self.data['X']
        self.C = self.data['C'] #/ self.data['C'].max()
        self.C = (self.C - self.C.mean(axis=0)) / self.C.std(axis=0)
        self.Ac = self.data['Ac']
        self.As = self.data['As']
        #self.P = self.data['P']
        self.D = self.data['D']
        #self.M = self.data['MV']
        #self.nM = self.data['nMV']
        self.nM = self.get_Mfeatures(self.D, self.Mv)
        self.P = self.get_holidays(self.D, self.holiday_df)
        self.K = np.eye(7)[[dt.astype(datetime.datetime).weekday() for dt in self.D]]
        self.PK = np.hstack((self.K, self.P[:, np.newaxis]))


        
        self.Ns, self.Nt, self.Nc  = self.X.shape[1], self.C.shape[2], self.C.shape[1]
        
        np.random.seed(0)
        #self.idx = (self.D >= np.datetime64(train_start)) & (self.D < np.datetime64(test_start))
        #mlist = np.arange(self.idx.shape[0])[self.idx]
        mlist = np.arange(len(self.D))
        np.random.shuffle(mlist)

        #self.tr_idx = mlist[int(mlist.shape[0]*val_ratio):]
        #self.va_idx = mlist[:int(mlist.shape[0]*val_ratio)]
        #self.te_idx = (self.D >= np.datetime64(test_start)) & (self.D < np.datetime64(test_end))
        
        if shuffle_mode:
            np.random.seed(0)
            mlist = np.arange(len(self.D))
            np.random.shuffle(mlist)
            self.tr_idx = mlist[:int(len(self.D)*0.7)]
            self.va_idx = mlist[int(len(self.D)*0.7):int(len(self.D)*0.8)]
            self.te_idx = mlist[int(len(self.D)*0.8):]
        else:
            np.random.seed(0)
            idx = (self.D >= np.datetime64(train_start)) & (self.D < np.datetime64(test_start))
            mlist = np.arange(idx.shape[0])[idx]
            self.tr_idx = mlist[int(mlist.shape[0]*val_ratio):]
            self.va_idx = mlist[:int(mlist.shape[0]*val_ratio)]
            self.te_idx = (self.D >= np.datetime64(test_start)) & (self.D < np.datetime64(test_end))

        

        self.tr_X = torch.tensor(self.X[self.tr_idx]).float()
        self.tr_C = torch.tensor(self.C[self.tr_idx]).float()
        self.tr_M = torch.tensor(self.nM[self.tr_idx]).float()
        self.tr_D = self.D[self.tr_idx]
        #self.tr_P = torch.tensor(self.P[self.tr_idx]).long()
        #self.tr_K = torch.tensor(self.K[self.tr_idx]).long()
        self.tr_PK = torch.tensor(self.PK[self.tr_idx]).long()
        

        self.va_X = torch.tensor(self.X[self.va_idx]).float()
        self.va_C = torch.tensor(self.C[self.va_idx]).float()
        self.va_M = torch.tensor(self.nM[self.va_idx]).float()
        self.va_D = self.D[self.va_idx]
        #self.va_P = torch.tensor(self.P[self.va_idx]).long()
        #self.va_K = torch.tensor(self.K[self.va_idx]).long()
        self.va_PK = torch.tensor(self.PK[self.va_idx]).long()

        self.te_X = torch.tensor(self.X[self.te_idx]).float()
        self.te_C = torch.tensor(self.C[self.te_idx]).float()
        self.te_M = torch.tensor(self.nM[self.te_idx]).float()
        self.te_D = self.D[self.te_idx]
        #self.te_P = torch.tensor(self.P[self.te_idx]).long()
        #self.te_K = torch.tensor(self.K[self.te_idx]).long()
        self.te_PK = torch.tensor(self.PK[self.te_idx]).long()
        
        #self.a_Ac = torch.tensor(self.data['Ac'])
        #self.a_As = torch.tensor(self.data['As'])
        
        #self.n_Ac = torch.tensor(self.Ac / self.Ac.sum(axis=0)[np.newaxis, :])
        #self.n_As = torch.tensor(self.As / self.As.sum(axis=0)[np.newaxis, :])
        
        self.n_Ac = self.Ac / self.Ac.sum(axis=0)[np.newaxis, :]
        self.n_As = self.As / self.As.sum(axis=0)[np.newaxis, :]
        
        
    def get_dataloaders(self):
        train_dataset = Data.TensorDataset(self.tr_X, self.tr_M, self.tr_C, self.tr_PK)
        train_loader = Data.DataLoader(dataset=train_dataset, batch_size=48, shuffle=True)
        val_dataset = Data.TensorDataset(self.va_X, self.va_M, self.va_C, self.va_PK)
        val_loader = Data.DataLoader(dataset=val_dataset, batch_size=48, shuffle=False)
        test_dataset = Data.TensorDataset(self.te_X, self.te_M, self.te_C, self.te_PK)
        test_loader = Data.DataLoader(dataset=test_dataset, batch_size=48, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def get_holidays(self, D, holiday_df):
        P = []
        for dt in D:
            dt = dt.astype(datetime.datetime)
            ymd = dt.year*10000+dt.month*100+dt.day
            if ymd in holiday_df['locdate'].tolist():
                P.append(1)
            else:
                P.append(0)
        return np.array(P)

    def get_Mfeatures(self, D, Mv):
        didx = []
        for d in D:
            didx.append(int((d - D[0]) / np.timedelta64(1, 'D')))
        return Mv[didx, :, :]
        
    def denormalize(self, x_pred):
        return x_pred
    
class MyDataLoaderG():
    def __init__(self, filename='../../dataprocess/gangnam_dataset_final_20210324.npz',
                        train_start='2015-01-01',
                        test_start='2019-01-01', 
                        test_end='2020-01-01',
                        shuffle_mode=False,
                        random_seed=42,
                        val_ratio=0.1): 
        print('gangnam_dataset_loaded')
        self.data = np.load(filename)
        
        fdf =pd.read_csv("../../dependency/node2vec-GMAN/SE_gangnam_directed_100_80_64.txt", sep=' ', skiprows=1, header=None)
        self.N2V = fdf.sort_values(0).set_index(0).values
        
        self.DAs = np.load('../../dataprocess/DAs.npy')
        self.Ms = np.load('../../dataprocess/Mstatic_gangnam.npy')
        self.Mv = np.load('../../dataprocess/nM50_gangnam_road_features.npy')

        self.holiday_df = pd.read_csv('../../holiday/national_holidays.csv', index_col=0)
        
        # tr: train set, 
        # te: test set
        self.X = self.data['X']
        self.C = self.data['C'] #/ self.data['C'].max()
        self.C = (self.C - self.C.mean(axis=0)) / self.C.std(axis=0)
        self.Ac = self.data['Ac']
        self.As = self.data['As']
        #self.P = self.data['P']
        self.D = self.data['D']
        #self.M = self.data['MV']
        #self.nM = self.data['nMV']
        self.nM = self.get_Mfeatures(self.D, self.Mv)
        self.P = self.get_holidays(self.D, self.holiday_df)
        self.K = np.eye(7)[[dt.astype(datetime.datetime).weekday() for dt in self.D]]
        self.PK = np.hstack((self.K, self.P[:, np.newaxis]))


        
        self.Ns, self.Nt, self.Nc  = self.X.shape[1], self.C.shape[2], self.C.shape[1]
        
        np.random.seed(0)
        #self.idx = (self.D >= np.datetime64(train_start)) & (self.D < np.datetime64(test_start))
        #mlist = np.arange(self.idx.shape[0])[self.idx]
        mlist = np.arange(len(self.D))
        np.random.shuffle(mlist)

        #self.tr_idx = mlist[int(mlist.shape[0]*val_ratio):]
        #self.va_idx = mlist[:int(mlist.shape[0]*val_ratio)]
        #self.te_idx = (self.D >= np.datetime64(test_start)) & (self.D < np.datetime64(test_end))
        
        if shuffle_mode:
            np.random.seed(0)
            mlist = np.arange(len(self.D))
            np.random.shuffle(mlist)
            self.tr_idx = mlist[:int(len(self.D)*0.7)]
            self.va_idx = mlist[int(len(self.D)*0.7):int(len(self.D)*0.8)]
            self.te_idx = mlist[int(len(self.D)*0.8):]
        else:
            np.random.seed(0)
            idx = (self.D >= np.datetime64(train_start)) & (self.D < np.datetime64(test_start))
            mlist = np.arange(idx.shape[0])[idx]
            self.tr_idx = mlist[int(mlist.shape[0]*val_ratio):]
            self.va_idx = mlist[:int(mlist.shape[0]*val_ratio)]
            self.te_idx = (self.D >= np.datetime64(test_start)) & (self.D < np.datetime64(test_end))

        

        self.tr_X = torch.tensor(self.X[self.tr_idx]).float()
        self.tr_C = torch.tensor(self.C[self.tr_idx]).float()
        self.tr_M = torch.tensor(self.nM[self.tr_idx]).float()
        self.tr_D = self.D[self.tr_idx]
        #self.tr_P = torch.tensor(self.P[self.tr_idx]).long()
        #self.tr_K = torch.tensor(self.K[self.tr_idx]).long()
        self.tr_PK = torch.tensor(self.PK[self.tr_idx]).long()
        

        self.va_X = torch.tensor(self.X[self.va_idx]).float()
        self.va_C = torch.tensor(self.C[self.va_idx]).float()
        self.va_M = torch.tensor(self.nM[self.va_idx]).float()
        self.va_D = self.D[self.va_idx]
        #self.va_P = torch.tensor(self.P[self.va_idx]).long()
        #self.va_K = torch.tensor(self.K[self.va_idx]).long()
        self.va_PK = torch.tensor(self.PK[self.va_idx]).long()

        self.te_X = torch.tensor(self.X[self.te_idx]).float()
        self.te_C = torch.tensor(self.C[self.te_idx]).float()
        self.te_M = torch.tensor(self.nM[self.te_idx]).float()
        self.te_D = self.D[self.te_idx]
        #self.te_P = torch.tensor(self.P[self.te_idx]).long()
        #self.te_K = torch.tensor(self.K[self.te_idx]).long()
        self.te_PK = torch.tensor(self.PK[self.te_idx]).long()
        
        #self.a_Ac = torch.tensor(self.data['Ac'])
        #self.a_As = torch.tensor(self.data['As'])
        
        #self.n_Ac = torch.tensor(self.Ac / self.Ac.sum(axis=0)[np.newaxis, :])
        #self.n_As = torch.tensor(self.As / self.As.sum(axis=0)[np.newaxis, :])
        
        self.n_Ac = self.Ac / self.Ac.sum(axis=0)[np.newaxis, :]
        self.n_As = self.As / self.As.sum(axis=0)[np.newaxis, :]
        
        
    def get_dataloaders(self):
        train_dataset = Data.TensorDataset(self.tr_X, self.tr_M, self.tr_C, self.tr_PK)
        train_loader = Data.DataLoader(dataset=train_dataset, batch_size=48, shuffle=True)
        val_dataset = Data.TensorDataset(self.va_X, self.va_M, self.va_C, self.va_PK)
        val_loader = Data.DataLoader(dataset=val_dataset, batch_size=48, shuffle=False)
        test_dataset = Data.TensorDataset(self.te_X, self.te_M, self.te_C, self.te_PK)
        test_loader = Data.DataLoader(dataset=test_dataset, batch_size=48, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def get_holidays(self, D, holiday_df):
        P = []
        for dt in D:
            dt = dt.astype(datetime.datetime)
            ymd = dt.year*10000+dt.month*100+dt.day
            if ymd in holiday_df['locdate'].tolist():
                P.append(1)
            else:
                P.append(0)
        return np.array(P)
        
    def get_Mfeatures(self, D, Mv):
        didx = []
        for d in D:
            didx.append(int((d - D[0]) / np.timedelta64(1, 'D')))
        return Mv[didx, :, :]

    def denormalize(self, x_pred):
        return x_pred