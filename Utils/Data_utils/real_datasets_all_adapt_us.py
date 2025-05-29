from sklearn.preprocessing import MinMaxScaler, StandardScaler
from Utils.Data_utils.timefeatures import time_features
from Utils.masking_utils import noise_mask
from Models.interpretable_diffusion.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from torch.utils.data import Dataset
from scipy import io
import os
import torch
import numpy as np
import pandas as pd
import time
import pickle
import random

np.random.seed(11)
torch.manual_seed(11)
torch.cuda.manual_seed(11)


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
        adapt=None,
        adapt_h=False,
        adapt_num_step=5,
        mean_mask_length=3,
        seq_len=30
    ):
        super(CustomDataset, self).__init__()
        # assert period in ['train', 'test'], 'period must be train or test.'
        if period == 'train':
            assert ~(predict_length is not None or missing_ratio is not None), ''

        self.name, self.pred_len, self.missing_ratio = name, predict_length, missing_ratio
        self.adapt = adapt
        self.feat_len = feat_len
        folder_path = "new_us_1h"
        self.style, self.distribution, self.mean_mask_length = style, distribution, mean_mask_length

        self.dir = os.path.join('./meta_output/', 'samples')
        os.makedirs(self.dir, exist_ok=True)

        samples = []
        samples_mark = []
        samples_incontext = []
        samples_incontext_mark = []
        gt = []
        ix = 0
        for filename in os.listdir(folder_path):
            ix += 1
            print(ix)
            if ix == 40:
                break
            file_path = os.path.join(folder_path, filename)
            self.rawdata, self.data_stamp = self.read_data(
                file_path, self.name)

            if self.adapt != 0:
                self.real_pred_len = self.pred_len*(self.adapt)
            else:
                self.real_pred_len = seq_len

            # predict_length = self.pred_len
            self.adapt_h = adapt_h
            self.window, self.period = window, period
            self.adapt_num_step = adapt_num_step
            self.his_interval = 5
            # self.window_adapt =   self.real_pred_len + self.adapt_num_step + self.pred_len + self.feat_len
            self.window_adapt = self.real_pred_len + \
                self.feat_len + self.adapt_num_step*self.his_interval
            # if self.adapt_h:
            #     self.window_adapt = adapt_his_len + self.real_pred_len

            self.len, self.var_num = self.rawdata.shape[0], self.rawdata.shape[-1]
            self.sample_num_total = max(self.len - self.window_adapt + 1, 0)
            self.save2npy = save2npy
            self.auto_norm = neg_one_to_one
            self.auto_norm = False

            print(self.rawdata.shape[0])
            train_len = int(self.rawdata.shape[0]*0.7)
            test_len = int(self.rawdata.shape[0]*0.2)
            val_len = self.len-train_len-test_len

            # self.num_step = 20
            # if self.window-self.feat_len<30:
            #     self.choice_number = self.feat_len + 30
            # else:
            #     self.choice_number = self.window
            self.choice_number = self.real_pred_len
            self.num_step = adapt_num_step
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

            if ix == 1:
                scaler = StandardScaler()
                self.scaler = scaler.fit(self.rawdata)
                file = "ecl_scaler.sav"
                pickle.dump(scaler, open(file, 'wb'))
            self.rawdata = self.__normalize(self.rawdata)
            self.train_data = self.rawdata[:train_len, :]
            self.val_data = self.rawdata[train_len -
                                         feat_len:val_len+train_len, :]
            self.test_data = self.rawdata[self.len-test_len-feat_len:, :]

            self.train_data_stamp = self.data_stamp[:train_len, :]
            self.val_data_stamp = self.data_stamp[train_len -
                                                  feat_len:val_len+train_len, :]
            self.test_data_stamp = self.data_stamp[self.len -
                                                   test_len-feat_len:, :]

            self.sample_num_train = max(
                self.train_data.shape[0] - self.window_adapt, 0)
            self.sample_num_val = max(
                self.val_data.shape[0] - self.window_adapt, 0)
            self.sample_num_test = max(
                self.test_data.shape[0] - self.window_adapt, 0)

            # train, inference = self.__getsamples(self.data, proportion, seed)
            train, train_mark, train_incontext, train_incontext_mark = \
                self.__get_train_samples(
                    self.train_data, self.train_data_stamp)
            inference_val, inference_mark_val, inference_incontext_val, inference_incontext_mark_val, x_gt = \
                self.__get_val_samples(self.val_data, self.val_data_stamp)
            inference, inference_mark, inference_incontext, inference_incontext_mark, x_gt_test = \
                self.__get_val_samples(self.test_data, self.test_data_stamp)
            # print(period)
            self.samples = train if period == 'train' else inference
            self.samples_mark = train_mark if period == 'train' else inference_mark
            # print(inference.shape)
            self.samples_incontext = train_incontext if period == 'train' else inference_incontext
            self.samples_incontext_mark = train_incontext_mark if period == 'train' else inference_incontext_mark

            if period == 'vali':
                self.samples = inference_val
                self.samples_mark = inference_mark_val
                self.samples_incontext = inference_incontext_val
                self.samples_incontext_mark = inference_incontext_mark_val

            # print(self.samples.shape, self.samples_incontext.shape)
            # exit()
            # print(len(samples))
            # print(len(samples_incontext))

            samples.append(self.samples)
            samples_mark.append(self.samples_mark)

            samples_incontext.append(self.samples_incontext)
            samples_incontext_mark.append(
                self.samples_incontext_mark)

            gt.append(x_gt_test)

        self.samples_all = np.concatenate(
            samples, axis=0).astype(np.float32)
        self.samples_mark_all = np.concatenate(
            samples_mark, axis=0).astype(np.float32)
        self.samples_incontext_all = np.concatenate(
            samples_incontext, axis=0).astype(np.float32)
        self.samples_incontext_mark_all = np.concatenate(
            samples_incontext_mark, axis=0).astype(np.float32)
        gt = np.concatenate(
            gt, axis=0).astype(np.float32)
        # print(self.samples_all.shape)
        # print(self.samples_incontext_all.shape)
        # exit()
        self.sample_num = self.samples_all.shape[0]

        if period == 'test':
            if missing_ratio is not None:
                self.masking = self.mask_data(seed)
            elif predict_length is not None:
                masks = np.ones(self.samples_all.shape)
                masks[:, -predict_length:, :] = 0
                self.masking = masks.astype(bool)
            else:
                raise NotImplementedError()

            if self.save2npy:

                np.save(os.path.join(
                    self.dir, f"{self.name}_ground_truth_{self.window}_test.npy"), self.unnormalize(self.samples_all))

                if self.auto_norm:

                    np.save(os.path.join(
                        self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), unnormalize_to_zero_to_one(self.samples_all))

                else:

                    np.save(os.path.join(
                        self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), self.samples_all)

            if self.adapt != 0:

                np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_test_adapt.npy"), self.unnormalize_in(
                    gt, self.real_pred_len+self.feat_len))

                if self.auto_norm:
                    np.save(os.path.join(
                        self.dir, f"{self.name}_norm_truth_{self.window}_test_adapt.npy"), unnormalize_to_zero_to_one(gt))

                else:
                    np.save(os.path.join(
                        self.dir, f"{self.name}_norm_truth_{self.window}_test_adapt.npy"), gt)

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
                np.save(os.path.join(
                    self.dir, f"{self.name}_ground_truth_{self.window}_test.npy"), self.unnormalize(test_data))
            np.save(os.path.join(
                self.dir, f"{self.name}_ground_truth_{self.window}_train.npy"), self.unnormalize(train_data))
            if self.auto_norm:
                if 1 - proportion > 0:
                    np.save(os.path.join(
                        self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), unnormalize_to_zero_to_one(test_data))
                np.save(os.path.join(
                    self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), unnormalize_to_zero_to_one(train_data))
            else:
                if 1 - proportion > 0:
                    np.save(os.path.join(
                        self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), test_data)
                np.save(os.path.join(
                    self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), train_data)

        return train_data, test_data

    def __get_train_samples(self, data, data_mark):

        x = np.zeros((data.shape[0]-self.real_pred_len-self.feat_len,
                      self.window, self.var_num))
        x_mark = np.zeros((data.shape[0]-self.real_pred_len-self.feat_len,
                           self.window, data_mark.shape[1]))

        for i in range(data.shape[0]-self.real_pred_len-self.feat_len):
            start = i
            end = i + self.window
            if end > data.shape[0]-self.real_pred_len-self.feat_len:
                break

            x[i, :, :] = data[start:end, :]
            x_mark[i, :, :] = data_mark[start:end, :]

            # if self.adapt_h:
            #     start = self.adapt_his_len - self.feat_len + i
            #     end = start + self.window

            #     x[i, :, :] = data[start:end, :]
            #     x_mark[i, :, :] = data_mark[start:end, :]

        train_data = x
        train_mark = x_mark
        incontextnum = self.sample_num_train  # -self.adapt_num_step*self.his_interval

        train_incontxt_data = np.zeros(
            (incontextnum, self.num_step, self.window, self.var_num))
        train_incontxt_mark = np.zeros(
            (incontextnum, self.num_step, self.window, data_mark.shape[1]))

        for i in range(incontextnum):
            choice = list(range(i, i+self.adapt_num_step *
                          self.his_interval, self.his_interval))

            train_incontxt_data[i, :, :, :] = train_data[choice, :, :]
            train_incontxt_mark[i, :, :, :] = train_mark[choice, :, :]
        # print(train_data.shape)
        # print(train_incontxt_data.shape)
        # exit()
        # train_incontxt_data = train_incontxt_data[:,:,:,:]
        # train_incontxt_mark = train_incontxt_mark[:,:,:,:]
        train_data = train_data[self.adapt_num_step*self.his_interval:, :, :]
        train_mark = train_mark[self.adapt_num_step*self.his_interval:, :, :]

        # train_incontxt_data = np.zeros((train_data.shape[0]-self.adapt_his_len+self.feat_len,\
        #                                 self.adapt_his_len-self.feat_len,self.feat_len,self.var_num))
        # train_incontxt_mark = np.zeros((train_data.shape[0]-self.adapt_his_len+self.feat_len,\
        #                                 self.adapt_his_len-self.feat_len,self.feat_len,data_mark.shape[1]))
        # if self.adapt_h:
        #     for i in range(self.adapt_his_len-self.feat_len,train_data.shape[0]):
        #         choice = list(range(i-self.adapt_his_len+self.feat_len,i))

        #         train_incontxt_data[i-self.adapt_his_len+self.feat_len,:,:,:] = train_data[choice,:self.feat_len,:]
        #         train_incontxt_mark[i-self.adapt_his_len+self.feat_len,:,:,:] = train_mark[choice,:self.feat_len,:]

        #     train_data = train_data[self.adapt_his_len-self.feat_len:,:,:]
        #     train_mark = train_mark[self.adapt_his_len-self.feat_len:,:,:]

        return train_data, train_mark, train_incontxt_data, train_incontxt_mark

    def __get_test_samples(self, data, data_mark):

        x = np.zeros((self.sample_num_test, self.window, self.var_num))
        x_mark = np.zeros(
            (self.sample_num_test, self.window, data_mark.shape[1]))

        for i in range(self.sample_num_test):
            start = i
            end = i + self.window

            x[i, :, :] = data[start:end, :]
            x_mark[i, :, :] = data_mark[start:end, :]

        test_data = x
        test_mark = x_mark
        # 读取数据集的时候,真实值已经恢复了原本的scaler
        if self.save2npy:

            np.save(os.path.join(
                self.dir, f"{self.name}_ground_truth_{self.window}_test.npy"), self.unnormalize(test_data))

            if self.auto_norm:

                np.save(os.path.join(
                    self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), unnormalize_to_zero_to_one(test_data))

            else:

                np.save(os.path.join(
                    self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), test_data)

        return test_data, test_mark

    def __get_val_samples(self, data, data_mark):

        x = np.zeros((data.shape[0]-self.real_pred_len-self.feat_len, self.window,
                      self.var_num))
        x_mark = np.zeros((data.shape[0]-self.real_pred_len-self.feat_len, self.window,
                           data_mark.shape[1]))

        # if self.adapt==0:
        #     adapt_num = 0
        # else:
        #     adapt_num = self.adapt - 1
        # 真实值 !!

        x_incontex = np.zeros(
            (data.shape[0]-self.real_pred_len-self.feat_len, self.real_pred_len+self.feat_len, self.var_num))
        x_mark_incontext = np.zeros(
            (data.shape[0]-self.real_pred_len-self.feat_len, self.real_pred_len+self.feat_len, data_mark.shape[1]))

        for i in range(data.shape[0]-self.real_pred_len-self.feat_len):
            start = i
            end = i + self.window
            # if end>data.shape[0]-self.real_pred_len-self.feat_len:
            #     break
            x[i, :, :] = data[start:end, :]
            x_mark[i, :, :] = data_mark[start:end, :]

            end_incontext = i + self.feat_len + self.real_pred_len

            x_incontex[i, :, :] = data[start:end_incontext, :]
            x_mark_incontext[i, :, :] = data_mark[start:end_incontext, :]

        test_data = x
        test_mark = x_mark

        # if self.adapt_h:
        x_incontex = x_incontex[self.adapt_num_step*self.his_interval:, :, :]
        x_mark_incontext = x_mark_incontext[self.adapt_num_step *
                                            self.his_interval:, :, :]
        # 读取数据集的时候,真实值已经恢复了原本的scaler

        # if self.adapt!=0:

        #     np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_test_adapt.npy"), self.unnormalize_in(x_incontex,self.real_pred_len+self.feat_len))

        #     if self.auto_norm:
        #         np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test_adapt.npy"), unnormalize_to_zero_to_one(x_incontex))

        #     else:
        #         np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test_adapt.npy"), x_incontex)

        # -self.adapt_num_step*self.his_interval
        incontextnum = data.shape[0]-self.real_pred_len - \
            self.feat_len-self.adapt_num_step*self.his_interval
        test_incontxt_data = np.zeros(
            (incontextnum, self.num_step, self.window, self.var_num))
        test_incontxt_mark = np.zeros(
            (incontextnum, self.num_step, self.window, data_mark.shape[1]))
        for i in range(incontextnum):
            choice = list(range(i, i+self.adapt_num_step *
                          self.his_interval, self.his_interval))

            # print(choice)
            # exit()
            test_incontxt_data[i, :, :, :] = test_data[choice, :, :]
            test_incontxt_mark[i, :, :, :] = test_mark[choice, :, :]

        test_data = test_data[self.adapt_num_step*self.his_interval:, :, :]
        test_mark = test_mark[self.adapt_num_step*self.his_interval:, :, :]

        # if self.save2npy:

        #     np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_test.npy"), self.unnormalize(test_data))

        #     if self.auto_norm:

        #         np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), unnormalize_to_zero_to_one(test_data))

        #     else:

        #         np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), test_data)

        # 这个incontext是history的

        # test_incontxt_data = np.zeros((test_data.shape[0]-self.adapt_his_len+self.feat_len,\
        #                                 self.adapt_his_len-self.feat_len,self.feat_len,self.var_num))
        # test_incontxt_mark = np.zeros((test_data.shape[0]-self.adapt_his_len+self.feat_len,\
        #                                 self.adapt_his_len-self.feat_len,self.feat_len,data_mark.shape[1]))
        # if self.adapt_h:
        #     for i in range(self.adapt_his_len-self.feat_len,test_data.shape[0]):
        #         choice = list(range(i-self.adapt_his_len+self.feat_len,i))

        #         test_incontxt_data[i-self.adapt_his_len+self.feat_len,:,:,:] = test_data[choice,:self.feat_len,:]
        #         test_incontxt_mark[i-self.adapt_his_len+self.feat_len,:,:,:] = test_mark[choice,:self.feat_len,:]

        #     test_data = test_data[self.adapt_his_len-self.feat_len:,:,:]
        #     test_mark = test_mark[self.adapt_his_len-self.feat_len:,:,:]

        return test_data, test_mark, test_incontxt_data, test_incontxt_mark, x_incontex

    def normalize(self, sq):
        d = sq.reshape(-1, self.var_num)
        d = self.scaler.transform(d)

        if self.auto_norm:
            d = normalize_to_neg_one_to_one(d)
        return d.reshape(-1, self.window, self.var_num)

    def unnormalize(self, sq):
        d = self.__unnormalize(sq.reshape(-1, self.var_num))
        return d.reshape(-1, self.window, self.var_num)

    def unnormalize_in(self, sq, len_):
        d = self.__unnormalize(sq.reshape(-1, self.var_num))

        return d.reshape(-1, len_, self.var_num)

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
        print(regular_train_id, irregular_train_id)
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

        train_len = int(df_stamp.shape[0]*0.7)
        test_len = int(df_stamp.shape[0]*0.2)
        val_len = df_stamp.shape[0]-train_len-test_len

        # feat_len = 96
        # train_data = df_stamp['date'][:train_len]
        # test_data = df_stamp['date'][df_stamp.shape[0]-test_len-feat_len:]
        # print(train_data)
        # print("=======test=======")
        # print(test_data)
        # exit()
        # df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        # df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        # df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        # df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        # df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
        # df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
        # data_stamp = df_stamp.drop(['date'], axis=1).values

        freq = "h"
        data_stamp = time_features(pd.to_datetime(
            df_stamp['date'].values), freq=freq)
        data_stamp = data_stamp.transpose(1, 0)

        # if name == 'etth':
        df.drop(df.columns[0], axis=1, inplace=True)
        data = df.values

        # scaler = StandardScaler()
        # scaler = scaler.fit(data)
        # file = "scaler_"+name+".sav"
        # pickle.dump(scaler,open(file,'wb'))
        # print("done")

        return data, data_stamp

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
            np.save(os.path.join(
                self.dir, f"{self.name}_masking_{self.window}.npy"), masks)

        # Restore RNG.
        np.random.set_state(st0)
        return masks.astype(bool)

    def __getitem__(self, ind):
        if self.period == 'test' or self.period == 'vali':

            x = self.samples_all[ind, :, :]  # (seq_length, feat_dim) array
            m = self.masking[ind, :, :]  # (seq_length, feat_dim) boolean array
            x_mark = self.samples_mark_all[ind, :, :]

            # x_incontext = self.samples_incontext[ind,:,:,:]
            # x_incontext_mark = self.samples_incontext_mark[ind,:,:,:]
            # x_incontext = self.samples[ind,:,:]
            # x_incontext_mark = self.samples_mark[ind,:,:]
            # if self.adapt_h:

            # x_incontext = self.samples_incontext_all[ind, :, :, :]
            # x_incontext_mark = self.samples_incontext_mark_all[ind, :, :, :]
            # return torch.from_numpy(x).float(), torch.from_numpy(m), torch.from_numpy(x_mark), \
            #     torch.from_numpy(x_incontext).float(), torch.from_numpy(
            #     x_incontext_mark).float()

            x_ = x[:self.feat_len, :]
            y_ = x[self.feat_len//2:, :]
            x_mark_ = x_mark[:self.feat_len, :]
            y_mark_ = x_mark[self.feat_len//2:, :]
            return torch.from_numpy(x_).float(), torch.from_numpy(y_).float(), \
                torch.from_numpy(x_mark_).float(
            ), torch.from_numpy(y_mark_).float()

        x = self.samples_all[ind, :, :]  # (seq_length, feat_dim) array
        x_mark = self.samples_mark_all[ind, :, :]
        # x_incontext = self.samples_incontext[ind,:,:,:]
        # x_incontext_mark = self.samples_incontext_mark[ind,:,:,:]
        # x_incontext = self.samples[ind,:,:]
        # x_incontext_mark = self.samples_mark[ind,:,:]
        # if self.adapt_h:

        # x_incontext = self.samples_incontext_all[ind, :, :, :]
        # x_incontext_mark = self.samples_incontext_mark_all[ind, :, :, :]
        # return torch.from_numpy(x).float(), torch.from_numpy(x_mark).float(), \
        #     torch.from_numpy(x_incontext).float(), torch.from_numpy(
        #         x_incontext_mark).float()

        x_ = x[:self.feat_len, :]
        y_ = x[self.feat_len//2:, :]
        x_mark_ = x_mark[:self.feat_len, :]
        y_mark_ = x_mark[self.feat_len//2:, :]
        return torch.from_numpy(x_).float(), torch.from_numpy(y_).float(), \
            torch.from_numpy(x_mark_).float(
        ), torch.from_numpy(y_mark_).float()

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
