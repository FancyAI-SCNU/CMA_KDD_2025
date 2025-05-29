import os
import torch
import numpy as np
import pandas as pd
import time
import pickle

from scipy import io
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from torch.utils.data import Dataset
from Models.interpretable_diffusion.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from Utils.masking_utils import noise_mask
from Utils.Data_utils.timefeatures import time_features

class CustomDataset(Dataset):
    def __init__(
        self, 
        name,
        data_root, 
        window=64,
        feat_len=30, 
        proportion=0.8, 
        save2npy=True, 
        neg_one_to_one=True,
        seed=123,
        period='train',
        output_dir='./OUTPUT',
        predict_length=None,
        missing_ratio=None,
        style='separate', 
        distribution='geometric', 
        mean_mask_length=3
    ):
        super(CustomDataset, self).__init__()
        assert period in ['train', 'test'], 'period must be train or test.'
        if period == 'train':
            assert ~(predict_length is not None or missing_ratio is not None), ''
        self.name, self.pred_len, self.missing_ratio = name, predict_length, missing_ratio
        self.style, self.distribution, self.mean_mask_length = style, distribution, mean_mask_length
        self.rawdata, self.scaler,self.data_stamp = self.read_data(data_root, self.name)
        self.dir = os.path.join(output_dir, 'samples')
        os.makedirs(self.dir, exist_ok=True)
        
        self.window, self.period = window, period
        self.len, self.var_num = self.rawdata.shape[0], self.rawdata.shape[-1]
        self.sample_num_total = max(self.len - self.window + 1, 0)
        self.save2npy = save2npy
        self.auto_norm = neg_one_to_one

        self.data = self.__normalize(self.rawdata)
        print(self.data.shape[0])
        train_len = int(self.data.shape[0]*0.7)
        test_len = int(self.data.shape[0]*0.2)
        val_len = self.len-train_len-test_len
        
        # self.train_data = self.data[:12 * 30 * 24 * 4,:]
        # self.val_data = self.data[12 * 30 * 24 * 4 -feat_len:12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,:]
        # self.test_data = self.data[12 * 30 * 24 * 4 + 4 * 30 * 24 * 4-feat_len:12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,:]
        # print(self.train_data.shape)
        # self.train_data_stamp = self.data_stamp[:12 * 30 * 24 * 4,:]
        # self.val_data_stamp = self.data_stamp[12 * 30 * 24 * 4 -feat_len:12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,:]
        # self.test_data_stamp = self.data_stamp[12 * 30 * 24 * 4 + 4 * 30 * 24 * 4-feat_len:12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,:]
        
        # self.sample_num_train = max(self.train_data.shape[0] - self.window + 1,0)
        # self.sample_num_val = max(self.val_data.shape[0] - self.window + 1,0)
        # self.sample_num_test = max(self.test_data.shape[0] - self.window + 1,0)

        self.train_data = self.data[:train_len,:]
        self.val_data = self.data[train_len-feat_len:val_len+train_len,:]
        self.test_data = self.data[self.len-test_len-feat_len:,:]
        print(self.train_data.shape,"1111")
        self.train_data_stamp = self.data_stamp[:train_len,:]
        self.val_data_stamp = self.data_stamp[train_len-feat_len:val_len+train_len,:]
        self.test_data_stamp = self.data_stamp[self.len-test_len-feat_len:,:]

        self.sample_num_train = max(self.train_data.shape[0] - self.window + 1,0)
        self.sample_num_val = max(self.val_data.shape[0] - self.window + 1,0)
        self.sample_num_test = max(self.test_data.shape[0] - self.window + 1,0)
        print(self.sample_num_train,"222")
        
        #train, inference = self.__getsamples(self.data, proportion, seed)
        train,train_mark = self.__get_train_samples(self.train_data,self.train_data_stamp)
        inference,inference_mark = self.__get_test_samples(self.test_data,self.test_data_stamp)

        
        self.samples = train if period == 'train' else inference
        self.samples_mark = train_mark if period == 'train' else inference_mark

        if period == 'test':
            if missing_ratio is not None:
                self.masking = self.mask_data(seed)
            elif predict_length is not None:
                masks = np.ones(self.samples.shape)
                masks[:, -predict_length:, :] = 0
                self.masking = masks.astype(bool)
            else:
                raise NotImplementedError()
        self.sample_num = self.samples.shape[0]

    def __getsamples(self, data, proportion, seed):
        
        x = np.zeros((self.sample_num_total, self.window, self.var_num))
        
        
        for i in range(self.sample_num_total):
            start = i
            end = i + self.window
            x[i, :, :] = data[start:end, :]
        
        train_data, test_data = self.divide(x, proportion, seed)
        # 读取数据集的时候,真实值已经恢复了原本的scaler
        if self.save2npy:
            if 1 - proportion > 0:
                np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_test.npy"), self.unnormalize(test_data))
            np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_train.npy"), self.unnormalize(train_data))
            if self.auto_norm:
                if 1 - proportion > 0:
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), unnormalize_to_zero_to_one(test_data))
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), unnormalize_to_zero_to_one(train_data))
            else:
                if 1 - proportion > 0:
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), test_data)
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), train_data)

        return train_data, test_data
    
    def __get_train_samples(self, data,data_mark):
        
        x = np.zeros((self.sample_num_train, self.window, self.var_num))
        x_mark = np.zeros((self.sample_num_train, self.window, data_mark.shape[1]))
        
        for i in range(self.sample_num_train):
            start = i
            end = i + self.window
            x[i, :, :] = data[start:end, :]
            x_mark[i, :, :] = data_mark[start:end, :]
        
        train_data = x
        train_mark = x_mark
        # 读取数据集的时候,真实值已经恢复了原本的scaler
        # if self.save2npy:
            
        #     np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_train.npy"), self.unnormalize(train_data))
        #     if self.auto_norm:
                
        #         np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), unnormalize_to_zero_to_one(train_data))
        #     else:
                
        #         np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), train_data)

        return train_data,train_mark
    

    def __get_test_samples(self, data,data_mark):
        
        x = np.zeros((self.sample_num_test, self.window, self.var_num))
        x_mark = np.zeros((self.sample_num_train, self.window, data_mark.shape[1]))
        
        for i in range(self.sample_num_test):
            start = i
            end = i + self.window
            x[i, :, :] = data[start:end, :]
            x_mark[i, :, :] = data_mark[start:end, :]
        
        test_data = x
        test_mark = x_mark
        # 读取数据集的时候,真实值已经恢复了原本的scaler
        if self.save2npy:
            
            np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_test.npy"), self.unnormalize(test_data))
            
            
            if self.auto_norm:
               
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), unnormalize_to_zero_to_one(test_data))
                
            else:
                
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), test_data)
                
        return test_data,test_mark
    
    def normalize(self, sq):
        d = sq.reshape(-1, self.var_num)
        d = self.scaler.transform(d)
        
        if self.auto_norm:
            d = normalize_to_neg_one_to_one(d)
        return d.reshape(-1, self.window, self.var_num)

    def unnormalize(self, sq):
        d = self.__unnormalize(sq.reshape(-1, self.var_num))
        return d.reshape(-1, self.window, self.var_num)
    
    def __normalize(self, rawdata):
        data = self.scaler.transform(rawdata)
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(data)
        return data

    def __unnormalize(self, data):
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        x = data
        return self.scaler.inverse_transform(x)
    
    @staticmethod
    def divide(data, ratio, seed=2023):
        size = data.shape[0]
        
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)
        
        regular_train_num = int(np.ceil(size * ratio))
        id_rdm = np.random.permutation(size)
        regular_train_id = id_rdm[:regular_train_num]
        irregular_train_id = id_rdm[regular_train_num:]
        print(regular_train_id,irregular_train_id)
        regular_data = data[regular_train_id, :]
        irregular_data = data[irregular_train_id, :]
        
        # Restore RNG.
        np.random.set_state(st0)
        
        return regular_data, irregular_data

    @staticmethod
    def read_data(filepath, name=''):
        """Reads a single .csv
        """
        df = pd.read_csv(filepath, header=0)
        df_stamp = df[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
        # df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        # df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        # df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        # df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        # df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
        # df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
        # data_stamp = df_stamp.drop(['date'], axis=1).values

        freq ="h"
        data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=freq)
        data_stamp = data_stamp.transpose(1, 0)
        
        # if name == 'etth':
        df.drop(df.columns[0], axis=1, inplace=True)
        data = df.values
        
        scaler = StandardScaler()
        scaler = scaler.fit(data)
        file = "scaler_"+name+".sav"
        pickle.dump(scaler,open(file,'wb'))
        print("done")
        
        return data, scaler ,data_stamp
    
    def mask_data(self, seed=2023):
        
        masks = np.ones_like(self.samples)
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        for idx in range(self.samples.shape[0]):
            x = self.samples[idx, :, :]  # (seq_length, feat_dim) array
            mask = noise_mask(x, self.missing_ratio, self.mean_mask_length, self.style,
                              self.distribution)  # (seq_length, feat_dim) boolean array
            masks[idx, :, :] = mask

        if self.save2npy:
            np.save(os.path.join(self.dir, f"{self.name}_masking_{self.window}.npy"), masks)

        # Restore RNG.
        np.random.set_state(st0)
        return masks.astype(bool)

    def __getitem__(self, ind):
        if self.period == 'test':
            x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
            m = self.masking[ind, :, :]  # (seq_length, feat_dim) boolean array
            x_mark = self.samples_mark[ind,:,:]
            
            return torch.from_numpy(x).float(), torch.from_numpy(m),torch.from_numpy(x_mark)
        x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
        x_mark = self.samples_mark[ind,:,:]
        
        return torch.from_numpy(x).float(),torch.from_numpy(x_mark).float()

    def __len__(self):
        return self.sample_num
    

class fMRIDataset(CustomDataset):
    def __init__(
        self, 
        proportion=1., 
        **kwargs
    ):
        super().__init__(proportion=proportion, **kwargs)

    @staticmethod
    def read_data(filepath, name=''):
        """Reads a single .csv
        """
        data = io.loadmat(filepath + '/sim4.mat')['ts']
        scaler = StandardScaler()
        scaler = scaler.fit(data)
        return data, scaler
