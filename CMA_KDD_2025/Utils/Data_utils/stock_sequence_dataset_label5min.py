import os
import numpy as np
import pandas as pd
import datetime
import pickle as pkl
import torch
from torch.utils import data
from shannon_store.shannon_store import DataStorageBase

from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import MinMaxScaler, StandardScaler
np.random.seed(42)
# cfg = Config.fromfile('configs/setting.py')

# vals = cfg.quantile_val
# mid_ind = cfg.mid_ind
# future_lag_minute = cfg.future_lag_minute

cache = {}


# def get_lb(x):
#     y = -1
#     if x > vals[4]:
#         y = 2
#     elif x < vals[2]:
#         y = 0
#     else:
#         y = 1
#     return y


# def get_qt(x):
#     if x in cache:
#         return cache[x]

#     y = -1
#     if x >= 0:
#         for i, val in enumerate(vals[mid_ind + 1:]):
#             if x < val:
#                 y = mid_ind + i
#                 break
#         if y < 0:
#             y = len(vals) - 1
#     else:
#         for i, val in enumerate(vals[:mid_ind]):
#             if x < val:
#                 y = i
#                 break
#         if y < 0:
#             y = mid_ind
#     if -100 <= x <= 100:
#         cache[x] = y
#     # print(x,y)
#     return y


# def get_quantile_class(x, d3=False):
#     ys = np.zeros_like(x).astype(np.int32)
#     x0 = np.round(x, 1)

#     if not d3:
#         for i in range(x0.shape[0]):
#             y = get_qt(x0[i])
#             ys[i] = y
#     else:
#         for i in range(x0.shape[0]):
#             for j in range(x0.shape[1]):
#                 y = get_qt(x0[i, j])
#                 ys[i, j] = y
#     # print(ys)
#     # print(x, x0, ys)
#     return ys


# def get_lb_class(x, d3=False):
#     # xs = x0 * self.price_quantile_interval_inverse
#     ys = np.zeros_like(x).astype(np.int32)
#     # x0 = np.round(x).astype(np.int32)
#     if not d3:
#         for i in range(x.shape[0]):
#             y = get_lb(x[i])
#             ys[i] = y
#     else:
#         for i in range(x.shape[0]):
#             for j in range(x.shape[1]):
#                 y = get_lb(x[i, j])
#                 ys[i, j] = y
#     # print(ys)
#     # print(x, x0, ys)
#     return ys


class public_stock_dataset_base(data.Dataset):
    # public_data_path = './data/public'
    # public_datasets = ['acl18', 'kdd17']
    TRAIN_FLAG, VAL_FLAG, TEST_FLAG = 0, 1, 2

    feat_gain_index = 4

    def __init__(self,

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
                 seq_len=30):
        # super(public_stock_dataset_base, self).__init__()
        hist_len = feat_len

        if period == 'train':
            assert ~(predict_length is not None or missing_ratio is not None), ''
        self.name, self.pred_len, self.missing_ratio = name, predict_length, missing_ratio
        self.adapt = adapt
        self.feat_len = feat_len

        self.style, self.distribution, self.mean_mask_length = style, distribution, mean_mask_length

        self.dir = os.path.join('./meta_output/', 'samples')
        os.makedirs(self.dir, exist_ok=True)
        if self.adapt != 0:

            self.real_pred_len = self.pred_len*(self.adapt)
        else:

            self.real_pred_len = seq_len

        with open('/home/haiqi/finrl_new_v1/data/trade_date_roll_1.pkl', 'rb') as f:
            self.train_set, self.val_set, self.test_set = pkl.load(f)

        with open('/home/haiqi/finrl_new_v1/data/stock_name_id.pkl', 'rb') as f:
            name2id, id2name = pkl.load(f)

        self.data_storage = DataStorageBase(
            '/home/haiqi/finrl_new_v1/InnovationFactors')
        self.zero_control = 10.0

        # prediction variables

        self.total_len = self.feat_len + self.pred_len

        if period == 'train':
            dset = self.train_set
        elif period == 'test':
            dset = self.test_set
        else:
            dset = self.val_set
        self.total_trade_date = sorted(list(dset))
        print(len(self.total_trade_date), '111111')

        assert self.pred_len >= 1
        self.num_stock = len(self.data_storage.get_all_ids(
            self.total_trade_date[0], 'Label5min'))
        self.minute_tick = 20  # 20
        self.future_lag = self.minute_tick * 5  # 20 * 5

        self.name2id = name2id
        self.id2name = id2name

        self.data_index = {}
        for i in range(0, len(self.total_trade_date)):
            date_ids = self.data_storage.get_all_ids(
                self.total_trade_date[i], 'Label5min')
            date_ids = list(filter(self.is_kcb, date_ids))
            for j in range(0, len(date_ids)):
                self.data_index.setdefault(
                    self.total_trade_date[i], []).append(date_ids[j])

        # hard-coding 3-second data for a day
        self.trading_hours = [0, 1, 2, 3]
        self.trading_hour_tick = 1200

        self.indices = ['IF.CFE', 'IC.CFE']
        self.pub_feat_keys = ['TickRtn', 'TickRtn_3', 'trade', 'trade_ratio', 'private_feat_0',
                              'mid_diff', 'im_vol1', 'im_vol2', 'im_vol3', 'im_vol4', 'im_vol5',
                              'spread', 'private_feat_1', 'private_feat_2', 'private_feat_3',
                              'private_feat_4', 'private_feat_5', 'private_feat_6']

        self.adapt_h = adapt_h
        self.window, self.period = window, period
        self.adapt_num_step = adapt_num_step
        self.his_interval = 5
        # self.window_adapt =   self.real_pred_len + self.adapt_num_step + self.pred_len + self.feat_len
        self.window_adapt = self.real_pred_len + \
            self.feat_len + self.adapt_num_step*self.his_interval

        # build additional attributes
        # https://datascienceparichay.com/article/pandas-get-day-of-week-from-date
        self.total_trade_info = {}

        for dt_s in dset:
            # print(dt_s)
            st = pd.to_datetime(dt_s, format='%Y%m%d', errors='ignore')
            # weekday_id = datetime.date(year, month, day).weekday()
            wk = st.weekday()
            mt = st.month - 1
            qt = st.quarter - 1
            day1 = st.day - 1
            # self.total_trade_info[dt_s] = [wk, mt, qt, day1]
            self.total_trade_info[dt_s] = [wk, mt, qt, day1]
            # day2 = st.days_in_month # how many days in this month
            # calendar_id = (mt) * 31 + (day - 1)
            # print(st,mt,wk, qt, day1)
            # exit()
        # print('haha')
        # exit()

    def is_kcb(self, stock_name):
        if stock_name.startswith('688') or stock_name.startswith('689'):
            return True
        return False


class public_stock_dataset(data.Dataset):
    """
    1) choose a segment in random for training, or
    """

    def __init__(self, name,
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
                 seq_len=30):
        super(public_stock_dataset, self).__init__()
        hist_len = feat_len

        assert period in ['train', 'test'], 'period must be train or test.'
        if period == 'train':
            assert ~(predict_length is not None or missing_ratio is not None), ''
        self.name, self.pred_len, self.missing_ratio = name, predict_length, missing_ratio
        self.adapt = adapt
        self.feat_len = feat_len

        self.style, self.distribution, self.mean_mask_length = style, distribution, mean_mask_length

        self.dir = os.path.join('./meta_output/', 'samples')
        os.makedirs(self.dir, exist_ok=True)
        if self.adapt != 0:

            self.real_pred_len = self.pred_len*(self.adapt)
        else:

            self.real_pred_len = seq_len

        with open('/home/haiqi/finrl_new_v1/data/trade_date_roll_1.pkl', 'rb') as f:
            self.train_set, self.val_set, self.test_set = pkl.load(f)
        print(self.test_set)
        exit()
        with open('/home/haiqi/finrl_new_v1/data/stock_name_id.pkl', 'rb') as f:
            name2id, id2name = pkl.load(f)

        self.data_storage = DataStorageBase(
            '/home/haiqi/finrl_new_v1/InnovationFactors')
        self.zero_control = 10.0

        # prediction variables

        self.total_len = self.feat_len + self.pred_len

        if period == 'train':
            dset = self.train_set
        else:
            dset = self.test_set
        self.total_trade_date = sorted(list(dset))

        assert self.pred_len >= 1
        self.num_stock = len(self.data_storage.get_all_ids(
            self.total_trade_date[0], 'Label5min'))
        self.minute_tick = 20  # 20
        self.future_lag = self.minute_tick * 5  # 20 * 5

        self.name2id = name2id
        self.id2name = id2name

        self.data_index = {}
        for i in range(0, len(self.total_trade_date)):
            date_ids = self.data_storage.get_all_ids(
                self.total_trade_date[i], 'Label5min')
            date_ids = list(filter(self.is_kcb, date_ids))
            for j in range(0, len(date_ids)):
                self.data_index.setdefault(
                    self.total_trade_date[i], []).append(date_ids[j])

        # hard-coding 3-second data for a day
        self.trading_hours = [0, 1, 2, 3]
        self.trading_hour_tick = 1200

        self.indices = ['IF.CFE', 'IC.CFE']
        self.pub_feat_keys = ['TickRtn', 'TickRtn_3', 'trade', 'trade_ratio', 'private_feat_0',
                              'mid_diff', 'im_vol1', 'im_vol2', 'im_vol3', 'im_vol4', 'im_vol5',
                              'spread', 'private_feat_1', 'private_feat_2', 'private_feat_3',
                              'private_feat_4', 'private_feat_5', 'private_feat_6']

        self.adapt_h = adapt_h
        self.window, self.period = window, period
        self.adapt_num_step = adapt_num_step
        self.his_interval = 5
        # self.window_adapt =   self.real_pred_len + self.adapt_num_step + self.pred_len + self.feat_len
        self.window_adapt = self.real_pred_len + \
            self.feat_len + self.adapt_num_step*self.his_interval

        # build additional attributes
        # https://datascienceparichay.com/article/pandas-get-day-of-week-from-date
        self.total_trade_info = {}

        for dt_s in dset:
            # print(dt_s)
            st = pd.to_datetime(dt_s, format='%Y%m%d', errors='ignore')
            # weekday_id = datetime.date(year, month, day).weekday()
            wk = st.weekday()
            mt = st.month - 1
            qt = st.quarter - 1
            day1 = st.day - 1
            # self.total_trade_info[dt_s] = [wk, mt, qt, day1]
            self.total_trade_info[dt_s] = [wk, mt, qt, day1]
            # day2 = st.days_in_month # how many days in this month
            # calendar_id = (mt) * 31 + (day - 1)
            # print(st,mt,wk, qt, day1)
            # exit()
        # print('haha')
        # exit()

    def is_kcb(self, stock_name):
        if stock_name.startswith('688') or stock_name.startswith('689'):
            return True
        return False

    def __len__(self):
        """__len__"""
        return len(self.total_trade_date)

    def __getitem__(self, index: int):
        my_date = self.total_trade_date[index]
        my_date_info = self.total_trade_info[my_date]

        # get random trading time
        all_slice_data = []
        all_label_data = []
        all_mid_data = []
        all_qt_label = []
        all_stock_id = []
        all_three_data = []
        # print(all_ids)
        slice_data_y = []
        slice_data_y_diff = []
        future_qt_label = []
        slice_data_mid = []

        temp = self.data_index[my_date]

        slice_np_in = []
        slice_y_in = []
        # stock_id_ = np.random.randint(0,len(temp)-1)
        # stock_id = temp[stock_id_]
        while True:
            print("111111111")
            stock_id_ = np.random.randint(0, len(temp)-1)
            stock_id = temp[stock_id_]
            period = torch.randint(low=0, high=4, size=(1,)).item()
            st_0 = self.feat_len + self.adapt_num_step*self.his_interval
            ed_0 = self.trading_hour_tick - \
                (self.real_pred_len + self.future_lag)

            if period == 0:
                st_0 = st_0 + self.feat_len  # + self.hist_len
            elif period == 3:
                ed_0 = ed_0 - self.minute_tick * 5

            mid_rel = torch.randint(low=st_0, high=ed_0, size=(1,)).item()
            # mid_rel = mid_rel // 10 * 10  # round to every 30 seconds
            # print(st, mid_rel, mid_rel2)
            mid = self.trading_hour_tick * period + mid_rel
            # print(mid.item())
            st_mid = mid - self.feat_len
            mid_ed = mid + self.pred_len

            slice_data = self.data_storage.get_slice_data(
                stock_id, my_date, 'PubFeature', beg=st_mid, end=mid_ed)

            if slice_data is None:
                # print(stock_id, my_date, st_mid, mid_ed)
                continue
            slice_data_np = slice_data.to_numpy()
            slice_data_np[:, [2, 4]] = np.log10(
                np.abs(slice_data_np[:, [2, 4]]) + 1)

            if np.any(np.isnan(slice_data_np)):
                continue

            slice_data_y = self.data_storage.get_slice_data(
                stock_id, my_date, 'Label5min', beg=st_mid, end=mid_ed)

            if slice_data_y is None:
                # # print('Bad data - 2')
                continue

            slice_data_y_np = slice_data_y.to_numpy()  # (_,4) 5mins _   _ log_mid_1min
            if np.any(np.isnan(slice_data_y_np)):
                continue

            slice_data_y_diff = slice_data_y_np[:, 0]

            slice_diff = slice_data_y_diff * 1000

            # slice_data_mid = slice_data_y_np[:, -1]
            # print(slice_data_mid.shape,'===')

            # future_qt_label = get_quantile_class(slice_diff, d3=False)
            # future_lb_label = get_lb_class(slice_diff, d3=False)

            if np.any(np.isnan(slice_data_y_diff)):
                continue

            # use to be 8, now set to 6 when classification
            # print(slice_data_np.shape[0], self.zero_control, '===')

            if np.count_nonzero(slice_data_np[:, 0]) < slice_data_np.shape[0] // self.zero_control:
                # print(slice_data_np[:, 0], 'skip')
                continue
            ###############################
            for st_0_incontext in range(mid_rel-self.adapt_num_step*self.his_interval, mid_rel, self.his_interval):
                # st_0_incontext = mid_rel-self.adapt_num_step*self.his_interval
                st_mid_in = st_0_incontext - self.feat_len
                mid_ed_in = st_0_incontext + self.pred_len

                slice_data_in = self.data_storage.get_slice_data(
                    stock_id, my_date, 'PubFeature', beg=st_mid_in, end=mid_ed_in)

                slice_data_np_in = slice_data_in.to_numpy()
                slice_data_np_in[:, [2, 4]] = np.log10(
                    np.abs(slice_data_np_in[:, [2, 4]]) + 1)

                slice_data_y_in = self.data_storage.get_slice_data(
                    stock_id, my_date, 'Label5min', beg=st_mid_in, end=mid_ed_in)

                slice_data_y_np_in = slice_data_y_in.to_numpy()  # (_,4) 5mins _   _ log_mid_1min

                slice_data_y_diff_in = slice_data_y_np_in[:, 0]

                slice_diff_in = slice_data_y_diff_in * 1000

                slice_np_in.append(slice_data_np_in)
                slice_y_in.append(slice_diff_in)
            # else:
            # print(np.count_nonzero(log_mid))
            break
            # all_stock_id.append(self.name2id[stock_id])

            # all_slice_data.append(slice_data_np[np.newaxis, ...])
            # all_label_data.append(slice_diff[np.newaxis, ...])
            # all_qt_label.append(future_qt_label[np.newaxis, ...])
            # all_mid_data.append(slice_data_mid[np.newaxis, ...])
            # all_three_data.append(future_lb_label[np.newaxis, ...])

        if len(all_slice_data) == 0:

            return 0

        # [wk, mt, qt, day1, period=1, mid_ind=1644]
        my_date_info.extend([period, mid_rel])
        my_date_info = np.array(my_date_info, dtype=np.int32)
        weekday_id = my_date_info[0]
        calendar_id = my_date_info[3]
        month_id = my_date_info[1]
        step_id = my_date_info[-1]

        mark = []
        mark.append(weekday_id)
        mark.append(calendar_id)
        mark.append(month_id)
        mark.append(step_id)
        mark = np.array(mark)
        all_stock_id = np.array(all_stock_id)

        # weekday_ids = []
        # calendar_ids = []
        # for i in range(self.hist_len+self.future_len):
        #     weekday_ids.append(weekday_id)
        #     calendar_ids.append(calendar_id)
        # weekday_ids = np.array([weekday_id] * self.total_len)
        # calendar_ids = np.array([calendar_id] * self.total_len)
        # week_ids=[]
        # calen_ids=[]
        # for i in range(all_stock_id.shape[0]):
        #     week_ids.append(weekday_ids)
        #     calen_ids.append(calendar_ids)

        mark_data = np.zeros((self.total_len, 4),
                             dtype=np.int32)

        for i in range(mark_data.shape[0]):
            mark_data[i, :] = mark
        mark_data_in = mark_data
        all_slice_data = np.concatenate(
            [slice_data_np, slice_diff], axis=1).astype(np.float32)
        slice_y_in = np.concatenate(
            slice_y_in, axis=0).astype(np.float32)
        slice_np_in = np.concatenate(
            slice_np_in, axis=0).astype(np.float32)
        all_slice_data_in = np.concatenate(
            [slice_np_in, slice_y_in], axis=2).astype(np.float32)
        # all_label_data = np.concatenate(
        #     all_label_data, axis=0).astype(np.float32)
        # print(all_slice_data.shape, all_label_data.shape)
        # all_qt_label = np.concatenate(all_qt_label, axis=0).astype(np.float32)
        # all_mid_data = np.concatenate(all_mid_data, axis=0).astype(np.float32)
        # all_three_data = np.concatenate(
        #     all_three_data, axis=0).astype(np.float32)
        # all_mid_data = all_mid_data[:, self.hist_len:]
        # all_slice_data = all_slice_data[:, :self.hist_len, :]
        # all_slice_data=all_slice_data[:,:,:]*self.price_quantile_interval_inverse
        # all_label_data = all_label_data[:, self.hist_len:, 0]
        # all_qt_label = all_qt_label[:, self.hist_len:]
        # all_three_data = all_three_data[:, self.hist_len:]
        # print(all_three_data.shape,'---',all_qt_label.shape)
        # print(all_mid_data.shape)
        # 4

        # return all_stock_id, calen_ids, week_ids, \
        #     all_slice_data, \
        #     all_label_data, \
        #     all_qt_label, \
        #     all_mid_data, \
        #     all_three_data
        print(all_slice_data.shape, mark_data.shape, all_slice_data_in.shape,)
        exit()
        return all_slice_data, mark_data, all_slice_data_in, mark_data_in


class public_stock_dataset_train(public_stock_dataset_base):
    """
    1) choose a segment in random for training, or
    """

    def __init__(self, name,
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
                 seq_len=30):
        super(public_stock_dataset_train, self).__init__(
            name,
            data_root,
            window,
            feat_len,
            proportion,
            save2npy,
            neg_one_to_one,
            seed,
            period,
            output_dir,
            predict_length,
            missing_ratio,
            style,
            distribution,
            adapt,
            adapt_h,
            adapt_num_step,
            mean_mask_length,
            seq_len)

        self.test_interval = 20
        assert self.future_lag == self.minute_tick * 5

        period_count = [0] * len(self.trading_hours)  # 0-3
        sample_mid = []
        sample_mid_rel = []
        for period in self.trading_hours:  # [0,1,2,3]
            # period = torch.randint(low=0, high=4, size=(1,))

            st_0 = self.feat_len + self.adapt_num_step*self.his_interval
            ed_0 = self.trading_hour_tick - \
                (self.real_pred_len + self.future_lag)
            if period == 0:
                st_0 = st_0 + self.pred_len  # + self.hist_len
            elif period == 3:
                ed_0 = ed_0 - self.future_lag

            # print(st, ed)
            # print()
            # mid_rel = torch.randint(low=st, high=ed, size=(1,))

            mid_rel = st_0
            cnt = 0
            while mid_rel + self.pred_len < ed_0:
                mid = self.trading_hour_tick * period + mid_rel

                sample_mid.append(mid)
                sample_mid_rel.append(mid_rel)

                mid_rel += self.test_interval
                cnt += 1

            period_count[period] = cnt

        self.period_count = period_count
        self.sample_mid = sample_mid[:-1]
        self.sample_mid_rel = sample_mid_rel[:-1]

        all_slice_data = []
        all_price_data = []
        all_mark_data = []

        x_incontext_gt = []
        x_in_y_gt = []

        x_incontext_data = []
        x_incontext_y = []
        x_incontext_mark = []

        all_stock_id = []
        print(len(self.total_trade_date))
        for my_date_id in range(0, len(self.total_trade_date), 4):

            my_date = self.total_trade_date[my_date_id]
            my_date_info = self.total_trade_info[my_date]
            temp = self.data_index[my_date]

            # print('haha')

            ix = 0
            for stock_id_ in range(0, len(temp), 100):
                stock_id = temp[stock_id_]
                all_data = self.data_storage.get_data(
                    stock_id, my_date, 'PubFeature')

                if all_data is None:
                    continue
                all_data_np = all_data.to_numpy()

                all_data_np[:, [2, 4]] = np.log10(
                    np.abs(all_data_np[:, [2, 4]]) + 1)

                all_data_y = self.data_storage.get_data(
                    stock_id, my_date, 'Label5min')

                if all_data_y is None:
                    continue

                # (140,4) logy_lag_10  logy_lag_20  logy_lag_60
                all_data_y_np = all_data_y.to_numpy()
                all_stock_id.append(self.name2id[stock_id])

                all_data_y_np_diff = all_data_y_np[:, :1]
                all_mid = all_data_y_np[:, -1]
                all_price = all_data_y_np[:, -2]
                # all_qt_label = get_quantile_class(all_data_y_np_diff*1000,d3=True)
                # all_lb_label = get_lb_class(all_data_y_np_diff*1000,d3=True)

                cnt = 0
                for mid, mid_rel in zip(self.sample_mid, self.sample_mid_rel):
                    # period = mid // self.trading_hour_tick
                    # print(period, mid, mid_rel)
                    st_mid = mid - self.feat_len
                    mid_ed = mid + self.pred_len
                    mid_ed_incontext = mid+self.real_pred_len

                    slice_data_np = all_data_np[st_mid:mid_ed, :]
                    # slice_mid = all_mid[st_mid:mid_ed]
                    # the price of after 5mins
                    # slice_price = all_price[st_mid +
                    #                         self.future_lag:mid_ed + self.future_lag]

                    assert mid_ed + self.future_lag < all_price.shape[0]
                    # print(slice_price.shape,"====")
                    slice_data_y_np_diff = all_data_y_np_diff[st_mid:mid_ed, :]

                    slice_data_np_true_in = all_data_np[st_mid:mid_ed_incontext, :]
                    slice_data_y_np_in = all_data_y_np_diff[st_mid:mid_ed_incontext, :]

                    # print(slice_data_y_np_diff.shape)

                    # print(slice_price.shape, st_mid, mid_ed)
                    if np.any(np.isinf(slice_data_np)) or np.any(np.isinf(slice_data_np_true_in)):
                        print('Nan data', stock_id)
                        continue

                    elif np.any(np.isinf(slice_data_y_np_diff)) or np.any(np.isinf(slice_data_y_np_in)):
                        # print('Nan label')
                        # all_valid_mask[cnt, ix] = 0
                        continue
                    # print(slice_data_np[:,0])
                    if np.any(np.isnan(slice_data_np)) or np.any(np.isnan(slice_data_np_true_in)):
                        print('Nan data', stock_id)
                        continue

                    elif np.any(np.isnan(slice_data_y_np_diff)) or np.any(np.isnan(slice_data_y_np_in)):
                        # print('Nan label')
                        # all_valid_mask[cnt, ix] = 0
                        continue

                    elif self.zero_control is not None and np.count_nonzero(slice_data_np[:, 0]) < slice_data_np.shape[
                            0] // self.zero_control:
                        # print('skip',slice_data_np.shape, np.count_nonzero(slice_data_np[:,0]))
                        # all_valid_mask[cnt, ix] = 0
                        continue

                    else:
                        # slice_data_y_np_diff *= 1000.0
                        # slice_data_y_np_in *= 1000.0

                        all_slice_data.append(slice_data_np[np.newaxis, ...])
                        all_price_data.append(
                            slice_data_y_np_diff[np.newaxis, ...])

                        x_incontext_gt.append(
                            slice_data_np_true_in[np.newaxis, ...])
                        x_in_y_gt.append(slice_data_y_np_in[np.newaxis, ...])

                        # all_qt_label = get_quantile_class(
                        #     slice_data_y_np_diff, d3=True)

                        # #all_lb_label = get_lb_class(slice_data_y_np_diff, d3=True)
                        # #all_valid_mask[cnt, ix] = 1
                        # all_slice_data[cnt, ix, :, :] = slice_data_np
                        # all_future_5min_data[cnt, ix, :, :] = slice_data_y_np_diff
                        # slice_mid = slice_mid.reshape(slice_mid.shape[0], 1)
                        # slice_price = slice_price.reshape(slice_price.shape[0], 1)
                        # # print(slice_price.shape, all_price_data.shape)
                        # all_mid_data[cnt, ix, :] = slice_mid
                        # all_price_data[cnt, ix, :] = slice_price
                        # all_qt_data[cnt, ix, :, :] = all_qt_label
                        # all_lb_data[cnt, ix, :, :] = all_lb_label
                    slice_np_in = []
                    slice_y_in = []
                    MARK = 0
                    for st_0_incontext in range(mid_rel-self.adapt_num_step*self.his_interval, mid_rel, self.his_interval):
                        # st_0_incontext = mid_rel-self.adapt_num_step*self.his_interval
                        st_mid_in = st_0_incontext - self.feat_len
                        mid_ed_in = st_0_incontext + self.pred_len
                        slice_data_np_incontext = all_data_np[st_mid_in:mid_ed_in, :]
                        slice_data_y_np_diff_incontext = all_data_y_np_diff[st_mid_in:mid_ed_in, :]
                        slice_diff_in_incontext = slice_data_y_np_diff_incontext  # * 1000
                        if np.any(np.isnan(slice_data_np_incontext)) or np.any(np.isnan(slice_diff_in_incontext)):
                            print('Nan data', stock_id)
                            MARK = 1
                            break

                        slice_np_in.append(
                            slice_data_np_incontext[np.newaxis, ...])
                        slice_y_in.append(
                            slice_diff_in_incontext[np.newaxis, ...])
                    if MARK == 1:
                        continue
                    slice_np_in = np.concatenate(
                        slice_np_in, axis=0).astype(np.float32)
                    slice_y_in = np.concatenate(
                        slice_y_in, axis=0).astype(np.float32)

                    x_incontext_data.append(slice_np_in[np.newaxis, ...])
                    x_incontext_y.append(slice_y_in[np.newaxis, ...])

                    my_date_info.extend([mid_rel])
                    my_date_info1 = np.array(my_date_info, dtype=np.int32)
                    weekday_id = my_date_info1[0]
                    calendar_id = my_date_info1[3]
                    month_id = my_date_info1[1]
                    step_id = my_date_info1[-1]

                    mark = []
                    mark.append(weekday_id)
                    mark.append(calendar_id)
                    mark.append(month_id)
                    mark.append(step_id)
                    mark = np.array(mark)

                    mark_data = np.zeros((self.total_len, 4),
                                         dtype=np.int32)
                    mark_data_1 = np.zeros((slice_y_in.shape[0], self.total_len, 4),
                                           dtype=np.int32)
                    for kk in range(mark_data.shape[0]):
                        mark_data[kk, :] = mark
                    for kkk in range(mark_data_1.shape[0]):
                        for kkkk in range(mark_data_1.shape[1]):
                            mark_data_1[kkk, kkkk, :] = mark
                    mark_data_in = mark_data

                    all_mark_data.append(mark_data_in[np.newaxis, ...])
                    x_incontext_mark.append(mark_data_1[np.newaxis, ...])

                    # all_stock_id = np.array(all_stock_id)
                    cnt += 1
                ix += 1
                # return [时间点,股票数,len,feature]
        all_slice_data = np.concatenate(
            all_slice_data, axis=0).astype(np.float32)
        all_price_data = np.concatenate(
            all_price_data, axis=0).astype(np.float32)
        all_mark_data = np.concatenate(
            all_mark_data, axis=0).astype(np.float32)

        x_incontext_gt = np.concatenate(
            x_incontext_gt, axis=0).astype(np.float32)
        x_in_y_gt = np.concatenate(x_in_y_gt, axis=0).astype(np.float32)

        x_incontext_data = np.concatenate(
            x_incontext_data, axis=0).astype(np.float32)
        x_incontext_y = np.concatenate(
            x_incontext_y, axis=0).astype(np.float32)
        x_incontext_mark = np.concatenate(
            x_incontext_mark, axis=0).astype(np.float32)

        self.all_data = np.concatenate(
            [all_slice_data, all_price_data], axis=-1).astype(np.float32)
        self.all_mark = all_mark_data

        x_gt = np.concatenate([x_incontext_gt, x_in_y_gt],
                              axis=-1).astype(np.float32)
        # np.save(os.path.join(
        #    self.dir, f"{self.name}_norm_truth_{self.window}_test_adapt.npy"), x_gt)

        self.x_incontext = np.concatenate(
            [x_incontext_data, x_incontext_y], axis=-1).astype(np.float32)
        self.x_incontext_mark = x_incontext_mark
        print(self.x_incontext.shape, self.all_data.shape, x_gt.shape)
        # print(self.all_data)
        # exit()

        self.var_num = self.x_incontext.shape[-1]

        scalerdata = self.all_data.reshape(-1, self.all_data.shape[-1])
        scaler = StandardScaler()
        scaler = scaler.fit(scalerdata)
        scalerdata = scaler.transform(scalerdata)
        scaler_ = MinMaxScaler()
        scaler_ = scaler_.fit(scalerdata)
        scalerdata = scaler_.transform(scalerdata)
        scalerdata = scalerdata*2-1
        scalerdata = scalerdata.reshape(self.all_data.shape)
        self.all_data = scalerdata

        file11 = "scaler11.sav"
        pkl.dump(scaler, open(file11, 'wb'))
        file = "ecl_scaler.sav"
        pkl.dump(scaler, open(file, 'wb'))
        file12 = "scaler12.sav"
        pkl.dump(scaler_, open(file12, 'wb'))
        file2 = "ecl_minmax_scaler.sav"
        pkl.dump(scaler_, open(file2, 'wb'))

        scalerdata2 = self.x_incontext.reshape(-1, self.x_incontext.shape[-1])
        scaler = StandardScaler()
        scaler = scaler.fit(scalerdata2)
        scalerdata2 = scaler.transform(scalerdata2)
        scaler_ = MinMaxScaler()
        scaler_ = scaler_.fit(scalerdata2)
        scalerdata2 = scaler_.transform(scalerdata2)
        scalerdata2 = scalerdata2*2-1
        scalerdata2 = scalerdata2.reshape(self.x_incontext.shape)
        self.x_incontext = scalerdata2
        print('1111122222')
        file21 = "scaler21.sav"
        pkl.dump(scaler, open(file21, 'wb'))
        file22 = "scaler22.sav"
        pkl.dump(scaler_, open(file22, 'wb'))

        print(self.x_incontext.shape, self.all_data.shape)

    def __len__(self):
        """__len__"""
        return self.all_data.shape[0]

    def __getitem__(self, index: int):

        # return self.all_data[index], self.all_mark[index], \
        #     self.x_incontext[index], self.x_incontext_mark[index]
        return self.all_data[index, :self.feat_len], self.all_data[index, self.feat_len//2:, :], \
            self.all_mark[index, :self.feat_len,
                          :], self.all_mark[index, self.feat_len//2:, :]


class public_stock_dataset_vali(public_stock_dataset_base):
    """
    1) choose a segment in random for training, or
    """

    def __init__(self, name,
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
                 seq_len=30):
        super(public_stock_dataset_test, self).__init__(
            name,
            data_root,
            window,
            feat_len,
            proportion,
            save2npy,
            neg_one_to_one,
            seed,
            period,
            output_dir,
            predict_length,
            missing_ratio,
            style,
            distribution,
            adapt,
            adapt_h,
            adapt_num_step,
            mean_mask_length,
            seq_len)

        self.test_interval = 100
        assert self.future_lag == self.minute_tick * 5

        period_count = [0] * len(self.trading_hours)  # 0-3
        sample_mid = []
        sample_mid_rel = []
        for period in self.trading_hours:  # [0,1,2,3]
            # period = torch.randint(low=0, high=4, size=(1,))

            st_0 = self.feat_len + self.adapt_num_step*self.his_interval
            ed_0 = self.trading_hour_tick - \
                (self.real_pred_len + self.future_lag)
            if period == 0:
                st_0 = st_0 + self.pred_len  # + self.hist_len
            elif period == 3:
                ed_0 = ed_0 - self.future_lag

            # print(st, ed)
            # print()
            # mid_rel = torch.randint(low=st, high=ed, size=(1,))

            mid_rel = st_0
            cnt = 0
            while mid_rel + self.pred_len < ed_0:
                mid = self.trading_hour_tick * period + mid_rel

                sample_mid.append(mid)
                sample_mid_rel.append(mid_rel)

                mid_rel += self.test_interval
                cnt += 1

            period_count[period] = cnt

        self.period_count = period_count
        self.sample_mid = sample_mid[:-1]
        self.sample_mid_rel = sample_mid_rel[:-1]

        all_slice_data = []
        all_price_data = []
        all_mark_data = []

        x_incontext_gt = []
        x_in_y_gt = []

        x_incontext_data = []
        x_incontext_y = []
        x_incontext_mark = []

        all_stock_id = []
        print(len(self.total_trade_date))
        for my_date_id in range(0, len(self.total_trade_date), 20):

            my_date = self.total_trade_date[my_date_id]
            my_date_info = self.total_trade_info[my_date]
            temp = self.data_index[my_date]

            # print('haha')

            ix = 0
            for stock_id_ in range(0, len(temp), 300):
                stock_id = temp[stock_id_]
                all_data = self.data_storage.get_data(
                    stock_id, my_date, 'PubFeature')

                if all_data is None:
                    continue
                all_data_np = all_data.to_numpy()

                all_data_np[:, [2, 4]] = np.log10(
                    np.abs(all_data_np[:, [2, 4]]) + 1)

                all_data_y = self.data_storage.get_data(
                    stock_id, my_date, 'Label5min')

                if all_data_y is None:
                    continue

                # (140,4) logy_lag_10  logy_lag_20  logy_lag_60
                all_data_y_np = all_data_y.to_numpy()
                all_stock_id.append(self.name2id[stock_id])

                all_data_y_np_diff = all_data_y_np[:, :1]
                all_mid = all_data_y_np[:, -1]
                all_price = all_data_y_np[:, -2]
                # all_qt_label = get_quantile_class(all_data_y_np_diff*1000,d3=True)
                # all_lb_label = get_lb_class(all_data_y_np_diff*1000,d3=True)

                cnt = 0
                for mid, mid_rel in zip(self.sample_mid, self.sample_mid_rel):
                    # period = mid // self.trading_hour_tick
                    # print(period, mid, mid_rel)
                    st_mid = mid - self.feat_len
                    mid_ed = mid + self.pred_len
                    mid_ed_incontext = mid+self.real_pred_len

                    slice_data_np = all_data_np[st_mid:mid_ed, :]
                    # slice_mid = all_mid[st_mid:mid_ed]
                    # the price of after 5mins
                    # slice_price = all_price[st_mid +
                    #                         self.future_lag:mid_ed + self.future_lag]

                    assert mid_ed + self.future_lag < all_price.shape[0]
                    # print(slice_price.shape,"====")
                    slice_data_y_np_diff = all_data_y_np_diff[st_mid:mid_ed, :]

                    slice_data_np_true_in = all_data_np[st_mid:mid_ed_incontext, :]
                    slice_data_y_np_in = all_data_y_np_diff[st_mid:mid_ed_incontext, :]

                    # print(slice_data_y_np_diff.shape)

                    # print(slice_price.shape, st_mid, mid_ed)

                    # print(slice_data_np[:,0])
                    if np.any(np.isnan(slice_data_np)):
                        print('Nan data', stock_id)
                        continue

                    elif np.any(np.isnan(slice_data_y_np_diff)):
                        # print('Nan label')
                        # all_valid_mask[cnt, ix] = 0
                        continue

                    elif self.zero_control is not None and np.count_nonzero(slice_data_np[:, 0]) < slice_data_np.shape[
                            0] // self.zero_control:
                        # print('skip',slice_data_np.shape, np.count_nonzero(slice_data_np[:,0]))
                        # all_valid_mask[cnt, ix] = 0
                        continue

                    else:
                        # slice_data_y_np_diff *= 1000.0
                        # slice_data_y_np_in *= 1000.0

                        all_slice_data.append(slice_data_np[np.newaxis, ...])
                        all_price_data.append(
                            slice_data_y_np_diff[np.newaxis, ...])

                        x_incontext_gt.append(
                            slice_data_np_true_in[np.newaxis, ...])
                        x_in_y_gt.append(slice_data_y_np_in[np.newaxis, ...])

                        # all_qt_label = get_quantile_class(
                        #     slice_data_y_np_diff, d3=True)

                        # #all_lb_label = get_lb_class(slice_data_y_np_diff, d3=True)
                        # #all_valid_mask[cnt, ix] = 1
                        # all_slice_data[cnt, ix, :, :] = slice_data_np
                        # all_future_5min_data[cnt, ix, :, :] = slice_data_y_np_diff
                        # slice_mid = slice_mid.reshape(slice_mid.shape[0], 1)
                        # slice_price = slice_price.reshape(slice_price.shape[0], 1)
                        # # print(slice_price.shape, all_price_data.shape)
                        # all_mid_data[cnt, ix, :] = slice_mid
                        # all_price_data[cnt, ix, :] = slice_price
                        # all_qt_data[cnt, ix, :, :] = all_qt_label
                        # all_lb_data[cnt, ix, :, :] = all_lb_label
                    slice_np_in = []
                    slice_y_in = []
                    for st_0_incontext in range(mid_rel-self.adapt_num_step*self.his_interval, mid_rel, self.his_interval):
                        # st_0_incontext = mid_rel-self.adapt_num_step*self.his_interval
                        st_mid_in = st_0_incontext - self.feat_len
                        mid_ed_in = st_0_incontext + self.pred_len
                        slice_data_np_incontext = all_data_np[st_mid_in:mid_ed_in, :]
                        slice_data_y_np_diff_incontext = all_data_y_np_diff[st_mid_in:mid_ed_in, :]
                        slice_diff_in_incontext = slice_data_y_np_diff_incontext  # * 1000

                        slice_np_in.append(
                            slice_data_np_incontext[np.newaxis, ...])
                        slice_y_in.append(
                            slice_diff_in_incontext[np.newaxis, ...])

                    slice_np_in = np.concatenate(
                        slice_np_in, axis=0).astype(np.float32)
                    slice_y_in = np.concatenate(
                        slice_y_in, axis=0).astype(np.float32)

                    x_incontext_data.append(slice_np_in[np.newaxis, ...])
                    x_incontext_y.append(slice_y_in[np.newaxis, ...])

                    my_date_info.extend([mid_rel])
                    my_date_info1 = np.array(my_date_info, dtype=np.int32)
                    weekday_id = my_date_info1[0]
                    calendar_id = my_date_info1[3]
                    month_id = my_date_info1[1]
                    step_id = my_date_info1[-1]

                    mark = []
                    mark.append(weekday_id)
                    mark.append(calendar_id)
                    mark.append(month_id)
                    mark.append(step_id)
                    mark = np.array(mark)

                    mark_data = np.zeros((self.total_len, 4),
                                         dtype=np.int32)
                    mark_data_1 = np.zeros((slice_y_in.shape[0], self.total_len, 4),
                                           dtype=np.int32)
                    for kk in range(mark_data.shape[0]):
                        mark_data[kk, :] = mark
                    for kkk in range(mark_data_1.shape[0]):
                        for kkkk in range(mark_data_1.shape[1]):
                            mark_data_1[kkk, kkkk, :] = mark
                    mark_data_in = mark_data

                    all_mark_data.append(mark_data_in[np.newaxis, ...])
                    x_incontext_mark.append(mark_data_1[np.newaxis, ...])

                    # all_stock_id = np.array(all_stock_id)
                    cnt += 1
                ix += 1
                # return [时间点,股票数,len,feature]
        all_slice_data = np.concatenate(
            all_slice_data, axis=0).astype(np.float32)
        all_price_data = np.concatenate(
            all_price_data, axis=0).astype(np.float32)
        all_mark_data = np.concatenate(
            all_mark_data, axis=0).astype(np.float32)

        x_incontext_gt = np.concatenate(
            x_incontext_gt, axis=0).astype(np.float32)
        x_in_y_gt = np.concatenate(x_in_y_gt, axis=0).astype(np.float32)

        x_incontext_data = np.concatenate(
            x_incontext_data, axis=0).astype(np.float32)
        x_incontext_y = np.concatenate(
            x_incontext_y, axis=0).astype(np.float32)
        x_incontext_mark = np.concatenate(
            x_incontext_mark, axis=0).astype(np.float32)

        self.all_data = np.concatenate(
            [all_slice_data, all_price_data], axis=-1).astype(np.float32)
        self.all_mark = all_mark_data

        x_gt = np.concatenate([x_incontext_gt, x_in_y_gt],
                              axis=-1).astype(np.float32)

        self.x_incontext = np.concatenate(
            [x_incontext_data, x_incontext_y], axis=-1).astype(np.float32)
        self.x_incontext_mark = x_incontext_mark
        self.var_num = self.x_incontext.shape[-1]
        print('test', self.x_incontext.shape, self.all_data.shape, x_gt.shape)
        self.mark = np.ones(
            self.all_data.shape, dtype=bool)
        self.mark[:, self.feat_len:, :] = False

        scaler_file = 'scaler11.sav'
        scaler11 = pkl.load(open(scaler_file, 'rb'))
        scaler_file = 'scaler12.sav'
        scaler12 = pkl.load(open(scaler_file, 'rb'))
        scaler_file = 'scaler21.sav'
        scaler21 = pkl.load(open(scaler_file, 'rb'))
        scaler_file = 'scaler22.sav'
        scaler22 = pkl.load(open(scaler_file, 'rb'))

        scalerdata = self.all_data.reshape(-1, self.all_data.shape[-1])
        # scaler = StandardScaler()
        # scaler = scaler.fit(scalerdata)
        scalerdata = scaler11.transform(scalerdata)
        # scaler_ = MinMaxScaler()
        # scaler_ = scaler_.fit(scalerdata)
        scalerdata = scaler12.transform(scalerdata)
        scalerdata = scalerdata*2-1
        scalerdata = scalerdata.reshape(self.all_data.shape)
        self.all_data = scalerdata

        scalerdata2 = self.x_incontext.reshape(-1, self.x_incontext.shape[-1])
        # scaler = StandardScaler()
        # scaler = scaler.fit(scalerdata2)
        scalerdata2 = scaler21.transform(scalerdata2)
        # scaler_ = MinMaxScaler()
        # scaler_ = scaler_.fit(scalerdata2)
        scalerdata2 = scaler22.transform(scalerdata2)
        scalerdata2 = scalerdata2*2-1
        scalerdata2 = scalerdata2.reshape(self.x_incontext.shape)
        self.x_incontext = scalerdata2

    def __len__(self):
        """__len__"""
        return self.all_data.shape[0]

    def __getitem__(self, index: int):

        # return self.all_data[index], self.mark[index], self.all_mark[index], \
        #     self.x_incontext[index], self.x_incontext_mark[index]
        return self.all_data[index, :self.feat_len], self.all_data[index, self.feat_len//2:, :], \
            self.all_mark[index, :self.feat_len,
                          :], self.all_mark[index, self.feat_len//2:, :]


class public_stock_dataset_test(public_stock_dataset_base):
    """
    1) choose a segment in random for training, or
    """

    def __init__(self, name,
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
                 seq_len=30):
        super(public_stock_dataset_test, self).__init__(
            name,
            data_root,
            window,
            feat_len,
            proportion,
            save2npy,
            neg_one_to_one,
            seed,
            period,
            output_dir,
            predict_length,
            missing_ratio,
            style,
            distribution,
            adapt,
            adapt_h,
            adapt_num_step,
            mean_mask_length,
            seq_len)

        self.test_interval = 100
        assert self.future_lag == self.minute_tick * 5

        period_count = [0] * len(self.trading_hours)  # 0-3
        sample_mid = []
        sample_mid_rel = []
        for period in self.trading_hours:  # [0,1,2,3]
            # period = torch.randint(low=0, high=4, size=(1,))

            st_0 = self.feat_len + self.adapt_num_step*self.his_interval
            ed_0 = self.trading_hour_tick - \
                (self.real_pred_len + self.future_lag)
            if period == 0:
                st_0 = st_0 + self.pred_len  # + self.hist_len
            elif period == 3:
                ed_0 = ed_0 - self.future_lag

            # print(st, ed)
            # print()
            # mid_rel = torch.randint(low=st, high=ed, size=(1,))

            mid_rel = st_0
            cnt = 0
            while mid_rel + self.pred_len < ed_0:
                mid = self.trading_hour_tick * period + mid_rel

                sample_mid.append(mid)
                sample_mid_rel.append(mid_rel)

                mid_rel += self.test_interval
                cnt += 1

            period_count[period] = cnt

        self.period_count = period_count
        self.sample_mid = sample_mid[:-1]
        self.sample_mid_rel = sample_mid_rel[:-1]

        all_slice_data = []
        all_price_data = []
        all_mark_data = []

        x_incontext_gt = []
        x_in_y_gt = []

        x_incontext_data = []
        x_incontext_y = []
        x_incontext_mark = []

        all_stock_id = []
        print(len(self.total_trade_date))
        for my_date_id in range(0, len(self.total_trade_date), 20):

            my_date = self.total_trade_date[my_date_id]
            my_date_info = self.total_trade_info[my_date]
            temp = self.data_index[my_date]

            # print('haha')

            ix = 0
            for stock_id_ in range(0, len(temp), 300):
                stock_id = temp[stock_id_]
                all_data = self.data_storage.get_data(
                    stock_id, my_date, 'PubFeature')

                if all_data is None:
                    continue
                all_data_np = all_data.to_numpy()

                all_data_np[:, [2, 4]] = np.log10(
                    np.abs(all_data_np[:, [2, 4]]) + 1)

                all_data_y = self.data_storage.get_data(
                    stock_id, my_date, 'Label5min')

                if all_data_y is None:
                    continue

                # (140,4) logy_lag_10  logy_lag_20  logy_lag_60
                all_data_y_np = all_data_y.to_numpy()
                all_stock_id.append(self.name2id[stock_id])

                all_data_y_np_diff = all_data_y_np[:, :1]
                all_mid = all_data_y_np[:, -1]
                all_price = all_data_y_np[:, -2]
                # all_qt_label = get_quantile_class(all_data_y_np_diff*1000,d3=True)
                # all_lb_label = get_lb_class(all_data_y_np_diff*1000,d3=True)

                cnt = 0
                for mid, mid_rel in zip(self.sample_mid, self.sample_mid_rel):
                    # period = mid // self.trading_hour_tick
                    # print(period, mid, mid_rel)
                    st_mid = mid - self.feat_len
                    mid_ed = mid + self.pred_len
                    mid_ed_incontext = mid+self.real_pred_len

                    slice_data_np = all_data_np[st_mid:mid_ed, :]
                    # slice_mid = all_mid[st_mid:mid_ed]
                    # the price of after 5mins
                    # slice_price = all_price[st_mid +
                    #                         self.future_lag:mid_ed + self.future_lag]

                    assert mid_ed + self.future_lag < all_price.shape[0]
                    # print(slice_price.shape,"====")
                    slice_data_y_np_diff = all_data_y_np_diff[st_mid:mid_ed, :]

                    slice_data_np_true_in = all_data_np[st_mid:mid_ed_incontext, :]
                    slice_data_y_np_in = all_data_y_np_diff[st_mid:mid_ed_incontext, :]

                    # print(slice_data_y_np_diff.shape)

                    # print(slice_price.shape, st_mid, mid_ed)

                    # print(slice_data_np[:,0])
                    if np.any(np.isnan(slice_data_np)):
                        print('Nan data', stock_id)
                        continue

                    elif np.any(np.isnan(slice_data_y_np_diff)):
                        # print('Nan label')
                        # all_valid_mask[cnt, ix] = 0
                        continue

                    elif self.zero_control is not None and np.count_nonzero(slice_data_np[:, 0]) < slice_data_np.shape[
                            0] // self.zero_control:
                        # print('skip',slice_data_np.shape, np.count_nonzero(slice_data_np[:,0]))
                        # all_valid_mask[cnt, ix] = 0
                        continue

                    else:
                        # slice_data_y_np_diff *= 1000.0
                        # slice_data_y_np_in *= 1000.0

                        all_slice_data.append(slice_data_np[np.newaxis, ...])
                        all_price_data.append(
                            slice_data_y_np_diff[np.newaxis, ...])

                        x_incontext_gt.append(
                            slice_data_np_true_in[np.newaxis, ...])
                        x_in_y_gt.append(slice_data_y_np_in[np.newaxis, ...])

                        # all_qt_label = get_quantile_class(
                        #     slice_data_y_np_diff, d3=True)

                        # #all_lb_label = get_lb_class(slice_data_y_np_diff, d3=True)
                        # #all_valid_mask[cnt, ix] = 1
                        # all_slice_data[cnt, ix, :, :] = slice_data_np
                        # all_future_5min_data[cnt, ix, :, :] = slice_data_y_np_diff
                        # slice_mid = slice_mid.reshape(slice_mid.shape[0], 1)
                        # slice_price = slice_price.reshape(slice_price.shape[0], 1)
                        # # print(slice_price.shape, all_price_data.shape)
                        # all_mid_data[cnt, ix, :] = slice_mid
                        # all_price_data[cnt, ix, :] = slice_price
                        # all_qt_data[cnt, ix, :, :] = all_qt_label
                        # all_lb_data[cnt, ix, :, :] = all_lb_label
                    slice_np_in = []
                    slice_y_in = []
                    for st_0_incontext in range(mid_rel-self.adapt_num_step*self.his_interval, mid_rel, self.his_interval):
                        # st_0_incontext = mid_rel-self.adapt_num_step*self.his_interval
                        st_mid_in = st_0_incontext - self.feat_len
                        mid_ed_in = st_0_incontext + self.pred_len
                        slice_data_np_incontext = all_data_np[st_mid_in:mid_ed_in, :]
                        slice_data_y_np_diff_incontext = all_data_y_np_diff[st_mid_in:mid_ed_in, :]
                        slice_diff_in_incontext = slice_data_y_np_diff_incontext  # * 1000

                        slice_np_in.append(
                            slice_data_np_incontext[np.newaxis, ...])
                        slice_y_in.append(
                            slice_diff_in_incontext[np.newaxis, ...])

                    slice_np_in = np.concatenate(
                        slice_np_in, axis=0).astype(np.float32)
                    slice_y_in = np.concatenate(
                        slice_y_in, axis=0).astype(np.float32)

                    x_incontext_data.append(slice_np_in[np.newaxis, ...])
                    x_incontext_y.append(slice_y_in[np.newaxis, ...])

                    my_date_info.extend([mid_rel])
                    my_date_info1 = np.array(my_date_info, dtype=np.int32)
                    weekday_id = my_date_info1[0]
                    calendar_id = my_date_info1[3]
                    month_id = my_date_info1[1]
                    step_id = my_date_info1[-1]

                    mark = []
                    mark.append(weekday_id)
                    mark.append(calendar_id)
                    mark.append(month_id)
                    mark.append(step_id)
                    mark = np.array(mark)

                    mark_data = np.zeros((self.total_len, 4),
                                         dtype=np.int32)
                    mark_data_1 = np.zeros((slice_y_in.shape[0], self.total_len, 4),
                                           dtype=np.int32)
                    for kk in range(mark_data.shape[0]):
                        mark_data[kk, :] = mark
                    for kkk in range(mark_data_1.shape[0]):
                        for kkkk in range(mark_data_1.shape[1]):
                            mark_data_1[kkk, kkkk, :] = mark
                    mark_data_in = mark_data

                    all_mark_data.append(mark_data_in[np.newaxis, ...])
                    x_incontext_mark.append(mark_data_1[np.newaxis, ...])

                    # all_stock_id = np.array(all_stock_id)
                    cnt += 1
                ix += 1
                # return [时间点,股票数,len,feature]
        all_slice_data = np.concatenate(
            all_slice_data, axis=0).astype(np.float32)
        all_price_data = np.concatenate(
            all_price_data, axis=0).astype(np.float32)

        all_mark_data = np.concatenate(
            all_mark_data, axis=0).astype(np.float32)

        x_incontext_gt = np.concatenate(
            x_incontext_gt, axis=0).astype(np.float32)
        x_in_y_gt = np.concatenate(x_in_y_gt, axis=0).astype(np.float32)

        x_incontext_data = np.concatenate(
            x_incontext_data, axis=0).astype(np.float32)
        x_incontext_y = np.concatenate(
            x_incontext_y, axis=0).astype(np.float32)
        x_incontext_mark = np.concatenate(
            x_incontext_mark, axis=0).astype(np.float32)

        self.all_data = np.concatenate(
            [all_slice_data, all_price_data], axis=-1).astype(np.float32)
        self.all_mark = all_mark_data

        x_gt = np.concatenate([x_incontext_gt, x_in_y_gt],
                              axis=-1).astype(np.float32)
        # np.save(os.path.join(
        #    self.dir, f"{self.name}_norm_truth_{self.window}_test_adapt.npy"), x_gt)

        self.x_incontext = np.concatenate(
            [x_incontext_data, x_incontext_y], axis=-1).astype(np.float32)
        self.x_incontext_mark = x_incontext_mark
        self.var_num = self.x_incontext.shape[-1]
        print('test', self.x_incontext.shape, self.all_data.shape, x_gt.shape)
        self.mark = np.ones(
            self.all_data.shape, dtype=bool)
        self.mark[:, self.feat_len:, :] = False

        scaler_file = 'scaler11.sav'
        scaler11 = pkl.load(open(scaler_file, 'rb'))
        scaler_file = 'scaler12.sav'
        scaler12 = pkl.load(open(scaler_file, 'rb'))
        scaler_file = 'scaler21.sav'
        scaler21 = pkl.load(open(scaler_file, 'rb'))
        scaler_file = 'scaler22.sav'
        scaler22 = pkl.load(open(scaler_file, 'rb'))

        scalerdata = self.all_data.reshape(-1, self.all_data.shape[-1])
        # scaler = StandardScaler()
        # scaler = scaler.fit(scalerdata)
        scalerdata = scaler11.transform(scalerdata)
        # scaler_ = MinMaxScaler()
        # scaler_ = scaler_.fit(scalerdata)
        scalerdata = scaler12.transform(scalerdata)
        scalerdata = scalerdata*2-1
        scalerdata = scalerdata.reshape(self.all_data.shape)
        self.all_data = scalerdata

        scalerdata2 = self.x_incontext.reshape(-1, self.x_incontext.shape[-1])
        # scaler = StandardScaler()
        # scaler = scaler.fit(scalerdata2)
        scalerdata2 = scaler21.transform(scalerdata2)
        # scaler_ = MinMaxScaler()
        # scaler_ = scaler_.fit(scalerdata2)
        scalerdata2 = scaler22.transform(scalerdata2)
        scalerdata2 = scalerdata2*2-1
        scalerdata2 = scalerdata2.reshape(self.x_incontext.shape)
        self.x_incontext = scalerdata2

        scalerdata3 = x_gt.reshape(-1, x_gt.shape[-1])
        # scaler = StandardScaler()
        # scaler = scaler.fit(scalerdata3)
        scalerdata3 = scaler11.transform(scalerdata3)
        scalerdata3 = scalerdata3.reshape(x_gt.shape)
        x_gt = scalerdata3
        print(x_gt.shape, '11111')

        np.save(os.path.join(
            self.dir, f"{self.name}_norm_truth_{self.window}_test_adapt.npy"), x_gt)

    def __len__(self):
        """__len__"""
        return self.all_data.shape[0]

    def __getitem__(self, index: int):

        # return self.all_data[index], self.mark[index], self.all_mark[index], \
        #     self.x_incontext[index], self.x_incontext_mark[index]
        return self.all_data[index, :self.feat_len], self.all_data[index, self.feat_len//2:, :], \
            self.all_mark[index, :self.feat_len,
                          :], self.all_mark[index, self.feat_len//2:, :]
    # def __getitem__(self, index: int):

    #     my_date = self.total_trade_date[index]
    #     my_date_info = self.total_trade_info[my_date]
    #     temp = self.data_index[my_date]

    #     all_slice_data = np.zeros((len(self.sample_mid), len(
    #         temp), self.total_len, 25), dtype=np.float32)
    #     # all_future_5min_data = np.zeros(
    #     #     (len(self.sample_mid), len(temp), self.future_len, 1), dtype=np.float32)
    #     # all_mid_data = np.zeros((len(self.sample_mid), len(
    #     #     temp), self.total_len, 1), dtype=np.float32)
    #     all_price_data = np.zeros((len(self.sample_mid), len(
    #         temp), self.total_len, 1), dtype=np.float32)
    #     # all_qt_data = np.zeros((len(self.sample_mid), len(
    #     #     temp), self.future_len, 1), dtype=np.float32)
    #     # all_lb_data = np.zeros((len(self.sample_mid), len(
    #     #     temp), self.future_len, 1), dtype=np.float32)
    #     # week_all_ids = np.zeros(
    #     #     (len(self.sample_mid), len(temp), self.total_len), dtype=np.int32)
    #     # calen_all_ids = np.zeros(
    #     #     (len(self.sample_mid), len(temp), self.total_len), dtype=np.int32)
    #     # all_valid_mask = np.ones(
    #     #     (len(self.sample_mid), len(temp)), dtype=np.int32)
    #     all_stock_id = []

    #     # print('haha')

    #     ix = 0
    #     for stock_id in temp:

    #         all_data = self.data_storage.get_data(
    #             stock_id, my_date, 'PubFeature')

    #         if all_data is None:
    #             continue
    #         all_data_np = all_data.to_numpy()

    #         all_data_np[:, [2, 4]] = np.log10(
    #             np.abs(all_data_np[:, [2, 4]]) + 1)

    #         all_data_y = self.data_storage.get_data(
    #             stock_id, my_date, 'Label5min')

    #         if all_data_y is None:
    #             continue

    #         # (140,4) logy_lag_10  logy_lag_20  logy_lag_60
    #         all_data_y_np = all_data_y.to_numpy()
    #         all_stock_id.append(self.name2id[stock_id])

    #         all_data_y_np_diff = all_data_y_np[:, :1]
    #         all_mid = all_data_y_np[:, -1]
    #         all_price = all_data_y_np[:, -2]
    #         # all_qt_label = get_quantile_class(all_data_y_np_diff*1000,d3=True)
    #         # all_lb_label = get_lb_class(all_data_y_np_diff*1000,d3=True)

    #         cnt = 0
    #         for mid, mid_rel in zip(self.sample_mid, self.sample_mid_rel):
    #             # period = mid // self.trading_hour_tick
    #             # print(period, mid, mid_rel)
    #             st_mid = mid - self.hist_len
    #             mid_ed = mid + self.future_len

    #             slice_data_np = all_data_np[st_mid:mid_ed, :]
    #             slice_mid = all_mid[st_mid:mid_ed]
    #             # the price of after 5mins
    #             slice_price = all_price[st_mid +
    #                                     self.future_lag:mid_ed + self.future_lag]

    #             assert mid_ed + self.future_lag < all_price.shape[0]
    #             # print(slice_price.shape,"====")
    #             slice_data_y_np_diff = all_data_y_np_diff[mid:mid_ed, :]

    #             # print(slice_data_y_np_diff.shape)

    #             # print(slice_price.shape, st_mid, mid_ed)

    #             # print(slice_data_np[:,0])
    #             if np.any(np.isnan(slice_data_np)):
    #                 print('Nan data', stock_id)
    #                 all_valid_mask[cnt, ix] = 0

    #             elif np.any(np.isnan(slice_data_y_np_diff)):
    #                 # print('Nan label')
    #                 all_valid_mask[cnt, ix] = 0

    #             elif self.zero_control is not None and np.count_nonzero(slice_data_np[:, 0]) < slice_data_np.shape[
    #                     0] // self.zero_control:
    #                 # print('skip',slice_data_np.shape, np.count_nonzero(slice_data_np[:,0]))
    #                 all_valid_mask[cnt, ix] = 0

    #             else:
    #                 slice_data_y_np_diff *= 1000.0
    #                 all_qt_label = get_quantile_class(
    #                     slice_data_y_np_diff, d3=True)

    #                 all_lb_label = get_lb_class(slice_data_y_np_diff, d3=True)
    #                 all_valid_mask[cnt, ix] = 1
    #                 all_slice_data[cnt, ix, :, :] = slice_data_np
    #                 all_future_5min_data[cnt, ix, :, :] = slice_data_y_np_diff
    #                 slice_mid = slice_mid.reshape(slice_mid.shape[0], 1)
    #                 slice_price = slice_price.reshape(slice_price.shape[0], 1)
    #                 # print(slice_price.shape, all_price_data.shape)
    #                 all_mid_data[cnt, ix, :] = slice_mid
    #                 all_price_data[cnt, ix, :] = slice_price
    #                 all_qt_data[cnt, ix, :, :] = all_qt_label
    #                 all_lb_data[cnt, ix, :, :] = all_lb_label

    #             cnt += 1
    #         ix += 1
    #         # return [时间点,股票数,len,feature]

    #     all_future_5min_data = all_future_5min_data[:, :, :, 0]
    #     all_qt_data = all_qt_data[:, :, :, 0]
    #     all_lb_data = all_lb_data[:, :, :, 0]
    #     # all_label_data *= 1000 # maybe not?

    #     # [wk, mt, qt, day1, period=1, mid_ind=1644]
    #     weekday_id = my_date_info[0]
    #     calendar_id = my_date_info[2]
    #     week_all_ids[:] = weekday_id
    #     calen_all_ids[:] = calendar_id

    #     # for mid, mid_rel in zip(self.sample_mid, self.sample_mid_rel):
    #     #     # period = mid // self.trading_hour_tick
    #     #     # cur_date_info = my_date_info + [period, mid_rel]
    #     #     # cur_date_info = np.array(cur_date_info, dtype=np.int32)
    #     #     week_all_ids[cnt, :] = week_ids
    #     #     calen_all_ids[cnt,:] = calen_ids
    #     #     # assert my_date_info[0] < 7
    #     #     # print(cur_date_info)
    #     #     cnt += 1
    #     # print(calen_all_ids.shape)
    #     # print(all_qt_data.shape)
    #     return all_stock_id, calen_all_ids, week_all_ids, \
    #         all_slice_data, \
    #         all_future_5min_data, \
    #         all_qt_data, \
    #         all_mid_data, \
    #         all_lb_data, \
    #         all_valid_mask, \
    #         all_price_data


class public_stock_rl_dataset(public_stock_dataset_base):
    """
    1) choose a segment in random for training a consecutive sequence
    """

    def __init__(self, dset, hist_len, future_len, num_rl_steps):
        super(public_stock_rl_dataset, self).__init__(
            dset, hist_len, future_len)
        self.num_rl_steps = num_rl_steps
        self.gain_mark_1 = [4635, 4347, 4786, 4456, 4496, 4756, 4722, 4595, 4553, 4729, 4385, 4465, 4546, 4421, 4662, 4538, 4729, 4504, 4398, 4661, 4710, 4361, 4345, 4346, 4652, 4560,
                            4617, 4605, 4646, 4601, 4501, 4462, 4556, 4393, 4786, 4679, 4742, 4576, 4365, 4370, 4701, 4538, 4420, 4616, 4750, 4679, 4766, 4721, 4358, 4598, 4336, 4379, 4349, 4372, 4395]
        self.gain_mark_0 = [4554, 4518, 4739, 4743, 4738, 4777, 4457, 4635, 4378, 4538, 4375, 4722, 4722, 4661, 4692, 4385, 4496, 4592, 4398, 4515, 4670, 4352, 4741, 4394, 4736, 4694, 4420, 4487, 4777,
                            4496, 4408, 4556, 4586, 4456, 4439, 4760, 4503, 4785, 4645, 4763, 4395, 4547, 4687, 4551, 4670, 4451, 4365, 4754, 4558, 4578, 4652, 4416, 4380, 4596, 4593, 4416, 4455, 4605, 4388, 4621, 4332]

    def __len__(self):
        """__len__"""
        return len(self.total_trade_date)

    def __getitem__(self, index: int):
        my_date = self.total_trade_date[index]
        my_date_info = self.total_trade_info[my_date]
        stock_ids = self.data_index[my_date]

        loop_count = 0

        # 20 steps to make a sequence for quoter evaluation
        num_rl_steps = self.num_rl_steps
        # hist_len_wind = self.hist_len + num_rl_steps
        future_len_wind = num_rl_steps
        # print(num_rl_steps)

        while True:
            stock_id_index = torch.randint(len(stock_ids), size=(1,)).item()
            stock_id = stock_ids[stock_id_index]
            if self.name2id[stock_id] in self.gain_mark_0:
                return 0
            loop_count += 1

            # if not found a valid stock in this date over 20 times
            if loop_count > 20:
                return 0

            period = torch.randint(low=0, high=4, size=(1,)).item()

            st_0 = self.hist_len
            ed_0 = self.trading_hour_tick - (future_len_wind + self.future_lag)

            if period == 0:
                st_0 = st_0 + self.future_lag
            elif period == 3:
                ed_0 = ed_0 - self.future_lag

            mid_rel = torch.randint(low=st_0, high=ed_0, size=(1,)).item()
            # mid_rel = mid_rel // 10 * 10  # round to every 30 seconds
            # print(st, mid_rel, mid_rel2)
            mid = self.trading_hour_tick * period + mid_rel
            # print(mid.item())
            st_mid = mid - self.hist_len + 1
            mid_ed = mid + future_len_wind
            # print(st_mid, mid_ed)
            slice_data = self.data_storage.get_slice_data(
                stock_id, my_date, 'PubFeature', beg=st_mid, end=mid_ed)
            # print(slice_data.shape)
            if slice_data is None:
                # print('Invalid', stock_id, my_date, st_mid, mid_ed)
                continue

            slice_data_np = slice_data.to_numpy()
            slice_data_np[:, [2, 4]] = np.log10(
                np.abs(slice_data_np[:, [2, 4]]) + 1)
            # print(slice_data_np.shape)

            if np.any(np.isnan(slice_data_np)):
                continue

            # use to be 8, now set to 6 when classification
            # print(slice_data_np.shape[0], self.zero_control, '===')
            if np.count_nonzero(slice_data_np[:, 0]) < slice_data_np.shape[0] // self.zero_control:
                # print(slice_data_np[:,0],'skip')
                continue

            slice_data_y = self.data_storage.get_slice_data(
                stock_id, my_date, 'Label5min', beg=mid, end=mid_ed)

            if slice_data_y is None:
                # print('Bad data - 2')
                continue

            # print(slice_data_y.shape)  # [20, 4]
            slice_data_y_np = slice_data_y.to_numpy()  # (_,4) 5mins _   _ log_mid_1min
            if np.any(np.isnan(slice_data_y_np)):
                continue

            # (20,)
            slice_data_y_diff = slice_data_y_np[:, 0]
            slice_diff = slice_data_y_diff * 1000
            slice_data_mid = slice_data_y_np[:, -1]
            # print(slice_data_mid.shape)

            future_qt_label = get_quantile_class(slice_diff, d3=False)
            future_lb_label = get_lb_class(slice_diff, d3=False)

            # (20, 18, 40), 20 consecutive sequences, each of 40 hist len
            slice_data_np_roll = sliding_window_view(
                slice_data_np, self.hist_len, axis=0)

            # print(slice_data_np_roll[:,0,0])
            # print(slice_data_np_roll[:,0,1])
            # print(slice_data_np[:,0])
            # (20, 40, 18)
            slice_data_np_roll = np.swapaxes(slice_data_np_roll, 1, 2)
            # print(slice_data_np_roll.shape)

            assert slice_data_np_roll.shape[0] == num_rl_steps

            all_stock_id = [self.name2id[stock_id]] * num_rl_steps

            # (20, 40, 18)
            all_slice_data = slice_data_np_roll.astype(np.float32)

            # (20,)
            all_three_data = future_lb_label.astype(np.int32)
            all_label_data = slice_diff.astype(np.float32)
            all_qt_label = future_qt_label.astype(np.int32)
            all_mid_data = slice_data_mid.astype(np.float32)

            break

        # if len(all_slice_data) == 0:
        #     continue

        # [wk, mt, qt, day1, period=1, mid_ind=1644]
        # my_date_info.extend([period, mid_rel])
        # my_date_info = np.array(my_date_info, dtype=np.int32)
        weekday_id = my_date_info[0]
        calendar_id = my_date_info[2]
        all_stock_id = np.array(all_stock_id)

        # print(all_mid_data.shape,'==')

        week_ids = np.zeros((len(all_stock_id), self.total_len),
                            dtype=np.int32) + int(weekday_id)
        calen_ids = np.zeros((len(all_stock_id), self.total_len),
                             dtype=np.int32) + int(calendar_id)

        # exit()
        return all_stock_id, calen_ids, week_ids, \
            all_slice_data, \
            all_label_data, \
            all_qt_label, \
            all_mid_data, \
            all_three_data


class public_stock_rl_dataset_test(public_stock_dataset_base):
    """
    1) choose a segment in random for training, or
    """

    def __init__(self, dset, hist_len, future_len, test_interval=1):
        super(public_stock_rl_dataset_test, self).__init__(
            dset, hist_len, future_len)

        assert test_interval == 1
        self.test_interval = test_interval
        assert self.future_lag == self.minute_tick * 5

        # every 1 tick we should extract
        del self.future_len
        future_len_wind = 1

        period_count = [0] * len(self.trading_hours)  # 0-3
        sample_mid = []
        sample_mid_vis = [[] for _ in range(len(self.trading_hours))]
        # sample_mid_rel = []
        for period in self.trading_hours:  # [0,1,2,3]
            # period = torch.randint(low=0, high=4, size=(1,))

            st_0 = self.hist_len
            ed_0 = self.trading_hour_tick - (future_len_wind + self.future_lag)

            if period == 0:
                st_0 = st_0 + self.future_lag
            elif period == 3:
                ed_0 = ed_0 - self.future_lag

            # print(st, ed)
            # print()
            # mid_rel = torch.randint(low=st, high=ed, size=(1,))

            mid_rel = st_0
            cnt = 0
            while mid_rel + future_len < ed_0:
                mid = self.trading_hour_tick * period + mid_rel

                sample_mid.append((mid, period, mid_rel))
                sample_mid_vis[period].append(mid)
                # sample_mid_rel.append(mid_rel)

                mid_rel += future_len_wind
                cnt += 1

            period_count[period] = cnt

        self.period_count = period_count
        self.sample_mid = sample_mid[:-1]
        # self.sample_mid_rel = sample_mid_rel[:-1]
        print('Evaluation time-step')
        print(self.period_count)
        # for mids in sample_mid_vis:
        #     print(mids)
        # exit()
        self.cur_stock = 0
        gain_mark_1 = [4635, 4347, 4786, 4456, 4496, 4756, 4722, 4595, 4553, 4729, 4385, 4465, 4546, 4421, 4662, 4538, 4729, 4504, 4398, 4661, 4710, 4361, 4345, 4346, 4652, 4560,
                       4617, 4605, 4646, 4601, 4501, 4462, 4556, 4393, 4786, 4679, 4742, 4576, 4365, 4370, 4701, 4538, 4420, 4616, 4750, 4679, 4766, 4721, 4358, 4598, 4336, 4379, 4349, 4372, 4395]
        gain_mark_0 = [4554, 4518, 4739, 4743, 4738, 4777, 4457, 4635, 4378, 4538, 4375, 4722, 4722, 4661, 4692, 4385, 4496, 4592, 4398, 4515, 4670, 4352, 4741, 4394, 4736, 4694, 4420, 4487, 4777,
                       4496, 4408, 4556, 4586, 4456, 4439, 4760, 4503, 4785, 4645, 4763, 4395, 4547, 4687, 4551, 4670, 4451, 4365, 4754, 4558, 4578, 4652, 4416, 4380, 4596, 4593, 4416, 4455, 4605, 4388, 4621, 4332]
        gain_mark_0 = []
        gain_mark_1 = []
        sample_stock_interval = 30
        date_stock_all = []
        for dt in self.total_trade_date:
            dt_stocks = self.data_index[dt]
            for st in dt_stocks:
                if self.name2id[st] in gain_mark_0:
                    continue
                else:
                    date_stock_all.append((dt, st))

        self.date_stock_all = date_stock_all[::sample_stock_interval]

        print('Steps %d each day' % (len(self.sample_mid),))
        # print(self.sample_mid, len(self.sample_mid))

    def __len__(self):
        """__len__"""
        return len(self.date_stock_all)

    def __getitem__(self, index: int):

        # my_date = self.total_trade_date[index]
        #
        # temp = self.data_index[my_date]
        my_date, stock_id = self.date_stock_all[index]
        my_date_info = self.total_trade_info[my_date]
        # num_rl_steps = cfg.num_rl_steps  # 20 steps to make a sequence for quoter evaluation
        # future_len_wind = num_rl_steps
        future_len_wind = 1

        all_slice_data = []
        all_future_5min_data = []
        all_mid_data = []
        all_price_data = []
        all_qt_label = []
        all_three_data = []
        all_stock_id = []

        all_valid_mid_index = []

        all_data = self.data_storage.get_data(stock_id, my_date, 'PubFeature')
        all_data_np = all_data.to_numpy()
        all_data_np[:, [2, 4]] = np.log10(np.abs(all_data_np[:, [2, 4]]) + 1)

        all_data_y = self.data_storage.get_data(stock_id, my_date, 'Label5min')
        # print(all_data.shape)

        assert all_data_y is not None
        assert (np.any(np.isnan(all_data_np)) == False)

        # (140,4) logy_lag_10  logy_lag_20  logy_lag_60
        all_data_y_np = all_data_y.to_numpy()
        all_stock_id.append(self.name2id[stock_id])

        all_data_y_np_diff = all_data_y_np[:, 0]

        all_diff = all_data_y_np_diff * 1000
        all_data_mid = all_data_y_np[:, -1]
        all_data_price = all_data_y_np[:, -2]

       #  print(all_diff.shape)

        all_future_qt_label = get_quantile_class(all_diff, d3=False)
        all_future_lb_label = get_lb_class(all_diff, d3=False)

        for ix, (mid, period, mid_rel) in enumerate(self.sample_mid):
            # period = mid // self.trading_hour_tick
            # print(period, mid, mid_rel)
            st_mid = mid - self.hist_len + 1
            mid_ed = mid + future_len_wind

            slice_data_np = all_data_np[st_mid:mid_ed, :]

            if np.count_nonzero(slice_data_np[:, 0]) < slice_data_np.shape[0] // self.zero_control:
                # print(slice_data_np[:,0],'skip')
                continue

            # slice_data_y_np = all_data_y_np[mid:mid_ed, :]

            all_valid_mid_index.append(mid)

            slice_data_y_diff = all_data_y_np_diff[mid:mid_ed]
            slice_diff = slice_data_y_diff * 1000
            slice_data_mid = all_data_mid[mid:mid_ed]
            slice_data_price = all_data_price[mid:mid_ed]
            future_qt_label = all_future_qt_label[mid:mid_ed]
            future_lb_label = all_future_lb_label[mid:mid_ed]

            slice_data_np_roll = sliding_window_view(
                slice_data_np, self.hist_len, axis=0)
            slice_data_np_roll = np.swapaxes(slice_data_np_roll, 1, 2)

            # assert slice_data_np_roll.shape[0] == num_rl_steps
            # print(slice_data_np_roll.shape, '----')

            # (1, 40, 18)
            # all_slice_data = slice_data_np_roll.astype(np.float32)

            # (1,)
            # print(slice_data_price.shape)

            # print(slice_data_mid.shape, slice_data_price.shape)
            all_slice_data.append(slice_data_np_roll)  # (1, 40, 18)
            all_future_5min_data.append(slice_diff)  # (1,)
            all_mid_data.append(slice_data_mid)  # (1,)
            all_price_data.append(slice_data_price)  # (1,)
            all_qt_label.append(future_qt_label)
            all_three_data.append(future_lb_label)
            # all_stock_id = []

        # for a bad trading date
        num_valid_sample_mid = len(all_three_data)
        if num_valid_sample_mid == 0:
            # print('Bad date ', my_date)
            return 0

        # (20,)

        all_stock_id = np.array([self.name2id[stock_id]] * future_len_wind)
        # all_stock_id.append(self.name2id[stock_id])

        weekday_id = my_date_info[0]
        calendar_id = my_date_info[2]
        assert len(all_valid_mid_index) == num_valid_sample_mid

        week_ids = np.zeros((num_valid_sample_mid, future_len_wind,
                            self.total_len), dtype=np.int32) + int(weekday_id)
        calen_ids = np.zeros((num_valid_sample_mid, future_len_wind,
                             self.total_len), dtype=np.int32) + int(calendar_id)

        # N_mid_sample is [34, 35, 35, 31] mids in four-hour periods
        # (N_mid_sample, 1)

        all_future_5min_data = np.stack(
            all_future_5min_data).astype(np.float32)
        # (N_mid_sample, 1)
        all_three_data = np.stack(all_three_data).astype(np.int32)

        # (N_mid_sample, 1)
        all_qt_label = np.stack(all_qt_label).astype(np.int32)
        # (N_mid_sample, 1)
        all_mid_data = np.stack(all_mid_data).astype(np.float32)
        # ((N_mid_sample, )
        all_valid_mid_index = np.array(all_valid_mid_index).astype(np.int32)
        # (N_mid_sample, 1)
        all_price_data = np.stack(all_price_data).astype(np.float32)
        # (N_mid_sample, 1, 40, 18)
        all_slice_data = np.stack(all_slice_data).astype(np.float32)
        # print(all_slice_data.shape)
        # , all_mid_data.shape, all_valid_mid_index.shape, all_pri
        # ce_data.shape)

        return all_stock_id, calen_ids, week_ids, \
            all_slice_data, \
            all_future_5min_data, \
            all_qt_label, \
            all_mid_data, \
            all_three_data, \
            all_valid_mid_index, \
            all_price_data


class public_stock_rl_dataset_test_slow(public_stock_dataset_base):
    """
    1) choose a segment in random for training, or
    """

    def __init__(self, dset, hist_len, future_len, test_interval=1):
        super(public_stock_rl_dataset_test, self).__init__(
            dset, hist_len, future_len)

        assert test_interval == 1
        self.test_interval = test_interval
        assert self.future_lag == self.minute_tick * 5
        num_rl_steps = cfg.num_rl_steps  # 20 steps to make a sequence for quoter evaluation
        # hist_len_wind = self.hist_len + num_rl_steps
        future_len_wind = num_rl_steps

        period_count = [0] * len(self.trading_hours)  # 0-3
        sample_mid = []
        sample_mid_vis = [[] for _ in range(len(self.trading_hours))]
        # sample_mid_rel = []
        for period in self.trading_hours:  # [0,1,2,3]
            # period = torch.randint(low=0, high=4, size=(1,))

            st_0 = self.hist_len
            ed_0 = self.trading_hour_tick - (future_len_wind + self.future_lag)

            if period == 0:
                st_0 = st_0 + future_len_wind  # + self.hist_len
            elif period == 3:
                ed_0 = ed_0 - self.future_lag

            # print(st, ed)
            # print()
            # mid_rel = torch.randint(low=st, high=ed, size=(1,))

            mid_rel = st_0
            cnt = 0
            while mid_rel + self.future_len < ed_0:
                mid = self.trading_hour_tick * period + mid_rel

                sample_mid.append((mid, period, mid_rel))
                sample_mid_vis[period].append(mid)
                # sample_mid_rel.append(mid_rel)

                mid_rel += self.test_interval
                cnt += 1

            period_count[period] = cnt

        self.period_count = period_count
        self.sample_mid = sample_mid[:-1]
        # self.sample_mid_rel = sample_mid_rel[:-1]
        print('Evaluation time-step')
        print(self.period_count)
        for mids in sample_mid_vis:
            print(mids)
        # exit()
        self.cur_stock = 0

        sample_stock_interval = 30
        date_stock_all = []
        for dt in self.total_trade_date:
            dt_stocks = self.data_index[dt]
            for st in dt_stocks:
                date_stock_all.append((dt, st))
        self.date_stock_all = date_stock_all[::sample_stock_interval]
        # print(self.date_stock_all, len(self.date_stock_all))
        print('Steps %d each day' % (len(self.sample_mid),))
        # print(self.sample_mid, len(self.sample_mid))

    def __len__(self):
        """__len__"""
        return len(self.date_stock_all)

    def __getitem__(self, index: int):

        # my_date = self.total_trade_date[index]
        #
        # temp = self.data_index[my_date]
        my_date, stock_id = self.date_stock_all[index]
        my_date_info = self.total_trade_info[my_date]
        # num_rl_steps = cfg.num_rl_steps  # 20 steps to make a sequence for quoter evaluation
        # future_len_wind = num_rl_steps
        future_len_wind = 1

        # (135, 20, 41, 18)
        # 135 is num of segments per trading day (4 hours)
        # 20 is number of RL steps
        # 41 is history + 1
        # 18 is feature dimension
        # all_slice_data = np.zeros((len_sample_mid, num_rl_steps, self.total_len, 18), dtype=np.float32)
        # all_future_5min_data = np.zeros((len_sample_mid, num_rl_steps, self.future_len, 1), dtype=np.float32)
        # all_mid_data = np.zeros((len_sample_mid, num_rl_steps, self.total_len, 1), dtype=np.float32)
        # all_price_data = np.zeros((len_sample_mid, num_rl_steps, self.total_len, 1), dtype=np.float32)
        # all_qt_data = np.zeros((len_sample_mid, num_rl_steps, self.future_len, 1), dtype=np.float32)
        # all_lb_data = np.zeros((len_sample_mid, num_rl_steps, self.future_len, 1), dtype=np.float32)
        # week_all_ids = np.zeros((len_sample_mid, num_rl_steps, self.total_len), dtype=np.int32)
        # calen_all_ids = np.zeros((len_sample_mid, num_rl_steps, self.total_len), dtype=np.int32)
        all_slice_data = []
        all_future_5min_data = []
        all_mid_data = []
        all_price_data = []
        all_qt_label = []
        all_three_data = []
        all_stock_id = []
        # week_all_ids = []
        # calen_all_ids = []
        # print(all_slice_data.shape)
        # exit()

        # all_data_y_np_diff = all_data_y_np[:, :1]
        # all_mid = all_data_y_np[:, -1]
        # all_price = all_data_y_np[:, -2]
        all_valid_mid_index = []
        # all_qt_label = get_quantile_class(all_data_y_np_diff*1000,d3=True)
        # all_lb_label = get_lb_class(all_data_y_np_diff*1000,d3=True)

        # all_data = self.data_storage.get_data(stock_id, my_date, 'PubFeature')
        # all_data_np = all_data.to_numpy()

        # all_data_np[:, [2, 4]] = np.log10(np.abs(all_data_np[:, [2, 4]]) + 1)

        # all_data_y = self.data_storage.get_data(stock_id, my_date, 'Label5min')

        # all_data_y_np = all_data_y.to_numpy()  # (140,4) logy_lag_10  logy_lag_20  logy_lag_60
        # all_stock_id.append(self.name2id[stock_id])

        # all_data_y_np_diff = all_data_y_np[:, :1]
        # all_mid = all_data_y_np[:, -1]
        # all_price = all_data_y_np[:, -2]

        for ix, (mid, period, mid_rel) in enumerate(self.sample_mid):
            # period = mid // self.trading_hour_tick
            # print(period, mid, mid_rel)
            st_mid = mid - self.hist_len + 1
            mid_ed = mid + future_len_wind

            slice_data = self.data_storage.get_slice_data(
                stock_id, my_date, 'PubFeature', beg=st_mid, end=mid_ed)
            # print(slice_data.shape)
            if slice_data is None:
                # print('Invalid', stock_id, my_date, st_mid, mid_ed)
                continue

            slice_data_np = slice_data.to_numpy()
            slice_data_np[:, [2, 4]] = np.log10(
                np.abs(slice_data_np[:, [2, 4]]) + 1)
            # print(slice_data_np.shape)

            if np.any(np.isnan(slice_data_np)):
                continue

            # use to be 8, now set to 6 when classification
            # print(slice_data_np.shape[0], self.zero_control, '===')
            if np.count_nonzero(slice_data_np[:, 0]) < slice_data_np.shape[0] // self.zero_control:
                # print(slice_data_np[:,0],'skip')
                continue

            slice_data_y = self.data_storage.get_slice_data(
                stock_id, my_date, 'Label5min', beg=mid, end=mid_ed)

            if slice_data_y is None:
                # print('Bad data - 2')
                continue

            # print(slice_data_y.shape)  # [20, 4]
            slice_data_y_np = slice_data_y.to_numpy()  # (_,4) 5mins _   _ log_mid_1min
            if np.any(np.isnan(slice_data_y_np)):
                continue

            all_valid_mid_index.append(mid)

            slice_data_y_diff = slice_data_y_np[:, 0]
            slice_diff = slice_data_y_diff * 1000
            slice_data_mid = slice_data_y_np[:, -1]
            slice_data_price = slice_data_y_np[:, -2]

            future_qt_label = get_quantile_class(slice_diff, d3=False)
            future_lb_label = get_lb_class(slice_diff, d3=False)

            slice_data_np_roll = sliding_window_view(
                slice_data_np, self.hist_len, axis=0)

            slice_data_np_roll = np.swapaxes(slice_data_np_roll, 1, 2)

            # assert slice_data_np_roll.shape[0] == num_rl_steps
            # print(slice_data_np_roll.shape, '----')

            # (1, 40, 18)
            # all_slice_data = slice_data_np_roll.astype(np.float32)

            # (1,)
            # print(slice_data_price.shape)
            # print(slice_data_mid.shape, slice_data_price.shape)
            all_slice_data.append(slice_data_np_roll)  # (1, 40, 18)
            all_future_5min_data.append(slice_diff)  # (1,)
            all_mid_data.append(slice_data_mid)  # (1,)
            all_price_data.append(slice_data_price)  # (1,)
            all_qt_label.append(future_qt_label)
            all_three_data.append(future_lb_label)
            # all_stock_id = []

        # for a bad trading date
        num_valid_sample_mid = len(all_three_data)
        if num_valid_sample_mid == 0:
            # print('Bad date ', my_date)
            return 0

        # (20,)
        all_stock_id = np.array([self.name2id[stock_id]] * future_len_wind)
        # all_stock_id.append(self.name2id[stock_id])

        weekday_id = my_date_info[0]
        calendar_id = my_date_info[2]
        assert len(all_valid_mid_index) == num_valid_sample_mid

        week_ids = np.zeros((num_valid_sample_mid, future_len_wind,
                            self.total_len), dtype=np.int32) + int(weekday_id)
        calen_ids = np.zeros((num_valid_sample_mid, future_len_wind,
                             self.total_len), dtype=np.int32) + int(calendar_id)

        # N_mid_sample is [34, 35, 35, 31] mids in four-hour periods
        # (N_mid_sample, 1)

        all_future_5min_data = np.stack(
            all_future_5min_data).astype(np.float32)
        # (N_mid_sample, 1)
        all_three_data = np.stack(all_three_data).astype(np.int32)

        # (N_mid_sample, 1)
        all_qt_label = np.stack(all_qt_label).astype(np.int32)
        # (N_mid_sample, 1)
        all_mid_data = np.stack(all_mid_data).astype(np.float32)
        # ((N_mid_sample, )
        all_valid_mid_index = np.array(all_valid_mid_index).astype(np.int32)
        # (N_mid_sample, 1)
        all_price_data = np.stack(all_price_data).astype(np.float32)
        # (N_mid_sample, 1, 40, 18)
        all_slice_data = np.stack(all_slice_data).astype(np.float32)
        # print(all_slice_data.shape)
        # , all_mid_data.shape, all_valid_mid_index.shape, all_price_data.shape)

        return all_stock_id, calen_ids, week_ids, \
            all_slice_data, \
            all_future_5min_data, \
            all_qt_label, \
            all_mid_data, \
            all_three_data, \
            all_valid_mid_index, \
            all_price_data


class public_stock_rl_dataset_test_old(public_stock_dataset_base):
    """
    1) choose a segment in random for training, or
    """

    def __init__(self, dset, hist_len, future_len, test_interval=1):
        super(public_stock_rl_dataset_test, self).__init__(
            dset, hist_len, future_len)

        assert test_interval == 1
        self.test_interval = test_interval
        assert self.future_lag == self.minute_tick * 5
        num_rl_steps = cfg.num_rl_steps  # 20 steps to make a sequence for quoter evaluation
        # hist_len_wind = self.hist_len + num_rl_steps
        future_len_wind = num_rl_steps

        period_count = [0] * len(self.trading_hours)  # 0-3
        sample_mid = []
        sample_mid_vis = [[] for _ in range(len(self.trading_hours))]
        # sample_mid_rel = []
        for period in self.trading_hours:  # [0,1,2,3]
            # period = torch.randint(low=0, high=4, size=(1,))

            st_0 = self.hist_len
            ed_0 = self.trading_hour_tick - (future_len_wind + self.future_lag)

            if period == 0:
                st_0 = st_0 + future_len_wind  # + self.hist_len
            elif period == 3:
                ed_0 = ed_0 - self.future_lag

            # print(st, ed)
            # print()
            # mid_rel = torch.randint(low=st, high=ed, size=(1,))

            mid_rel = st_0
            cnt = 0
            while mid_rel + self.future_len < ed_0:
                mid = self.trading_hour_tick * period + mid_rel

                sample_mid.append((mid, period, mid_rel))
                sample_mid_vis[period].append(mid)
                # sample_mid_rel.append(mid_rel)

                mid_rel += self.test_interval
                cnt += 1

            period_count[period] = cnt

        self.period_count = period_count
        self.sample_mid = sample_mid[:-1]
        # self.sample_mid_rel = sample_mid_rel[:-1]
        print('Evaluation time-step')
        print(self.period_count)
        for mids in sample_mid_vis:
            print(mids)
        # exit()
        self.cur_stock = 0

        sample_stock_interval = 50
        date_stock_all = []
        for dt in self.total_trade_date:
            dt_stocks = self.data_index[dt]
            for st in dt_stocks:
                date_stock_all.append((dt, st))
        self.date_stock_all = date_stock_all[::sample_stock_interval]
        print(self.date_stock_all, len(self.date_stock_all))

        print(self.sample_mid, len(self.sample_mid))

    def __len__(self):
        """__len__"""
        return len(self.date_stock_all)

    def __getitem__(self, index: int):

        # my_date = self.total_trade_date[index]
        #
        # temp = self.data_index[my_date]
        my_date, stock_id = self.date_stock_all[index]
        my_date_info = self.total_trade_info[my_date]
        num_rl_steps = cfg.num_rl_steps  # 20 steps to make a sequence for quoter evaluation
        future_len_wind = num_rl_steps

        # (135, 20, 41, 18)
        # 135 is num of segments per trading day (4 hours)
        # 20 is number of RL steps
        # 41 is history + 1
        # 18 is feature dimension
        # all_slice_data = np.zeros((len_sample_mid, num_rl_steps, self.total_len, 18), dtype=np.float32)
        # all_future_5min_data = np.zeros((len_sample_mid, num_rl_steps, self.future_len, 1), dtype=np.float32)
        # all_mid_data = np.zeros((len_sample_mid, num_rl_steps, self.total_len, 1), dtype=np.float32)
        # all_price_data = np.zeros((len_sample_mid, num_rl_steps, self.total_len, 1), dtype=np.float32)
        # all_qt_data = np.zeros((len_sample_mid, num_rl_steps, self.future_len, 1), dtype=np.float32)
        # all_lb_data = np.zeros((len_sample_mid, num_rl_steps, self.future_len, 1), dtype=np.float32)
        # week_all_ids = np.zeros((len_sample_mid, num_rl_steps, self.total_len), dtype=np.int32)
        # calen_all_ids = np.zeros((len_sample_mid, num_rl_steps, self.total_len), dtype=np.int32)
        all_slice_data = []
        all_future_5min_data = []
        all_mid_data = []
        all_price_data = []
        all_qt_label = []
        all_three_data = []
        all_stock_id = []
        # week_all_ids = []
        # calen_all_ids = []
        # print(all_slice_data.shape)
        # exit()

        # all_data_y_np_diff = all_data_y_np[:, :1]
        # all_mid = all_data_y_np[:, -1]
        # all_price = all_data_y_np[:, -2]
        all_valid_mid_index = []
        # all_qt_label = get_quantile_class(all_data_y_np_diff*1000,d3=True)
        # all_lb_label = get_lb_class(all_data_y_np_diff*1000,d3=True)

        for ix, (mid, period, mid_rel) in enumerate(self.sample_mid):
            # period = mid // self.trading_hour_tick
            # print(period, mid, mid_rel)
            st_mid = mid - self.hist_len + 1
            mid_ed = mid + future_len_wind

            slice_data = self.data_storage.get_slice_data(
                stock_id, my_date, 'PubFeature', beg=st_mid, end=mid_ed)
            # print(slice_data.shape)
            if slice_data is None:
                # print('Invalid', stock_id, my_date, st_mid, mid_ed)
                continue

            slice_data_np = slice_data.to_numpy()
            slice_data_np[:, [2, 4]] = np.log10(
                np.abs(slice_data_np[:, [2, 4]]) + 1)
            # print(slice_data_np.shape)

            if np.any(np.isnan(slice_data_np)):
                continue

            # use to be 8, now set to 6 when classification
            # print(slice_data_np.shape[0], self.zero_control, '===')
            if np.count_nonzero(slice_data_np[:, 0]) < slice_data_np.shape[0] // self.zero_control:
                # print(slice_data_np[:,0],'skip')
                continue

            slice_data_y = self.data_storage.get_slice_data(
                stock_id, my_date, 'Label5min', beg=mid, end=mid_ed)

            if slice_data_y is None:
                # print('Bad data - 2')
                continue

            # print(slice_data_y.shape)  # [20, 4]
            slice_data_y_np = slice_data_y.to_numpy()  # (_,4) 5mins _   _ log_mid_1min
            if np.any(np.isnan(slice_data_y_np)):
                continue

            all_valid_mid_index.append(mid)

            slice_data_y_diff = slice_data_y_np[:, 0]
            slice_diff = slice_data_y_diff * 1000
            slice_data_mid = slice_data_y_np[:, -1]
            slice_data_price = slice_data_y_np[:, -2]

            future_qt_label = get_quantile_class(slice_diff, d3=False)
            future_lb_label = get_lb_class(slice_diff, d3=False)

            slice_data_np_roll = sliding_window_view(
                slice_data_np, self.hist_len, axis=0)

            slice_data_np_roll = np.swapaxes(slice_data_np_roll, 1, 2)

            assert slice_data_np_roll.shape[0] == num_rl_steps

            # (20, 40, 18)
            # all_slice_data = slice_data_np_roll.astype(np.float32)

            # (20,)
            # print(slice_data_mid.shape, slice_data_price.shape)
            all_slice_data.append(slice_data_np_roll)  # (20, 40, 18)
            all_future_5min_data.append(slice_diff)  # (20,)
            all_mid_data.append(slice_data_mid)  # (20,)
            all_price_data.append(slice_data_price)  # (20,)
            all_qt_label.append(future_qt_label)
            all_three_data.append(future_lb_label)
            # all_stock_id = []

        # for a bad trading date
        num_valid_sample_mid = len(all_three_data)
        if num_valid_sample_mid == 0:
            # print('Bad date ', my_date)
            return 0

        # (20,)
        all_stock_id = np.array([self.name2id[stock_id]] * num_rl_steps)
        # all_stock_id.append(self.name2id[stock_id])

        weekday_id = my_date_info[0]
        calendar_id = my_date_info[2]
        assert len(all_valid_mid_index) == num_valid_sample_mid

        week_ids = np.zeros((num_valid_sample_mid, num_rl_steps,
                            self.total_len), dtype=np.int32) + int(weekday_id)
        calen_ids = np.zeros((num_valid_sample_mid, num_rl_steps,
                             self.total_len), dtype=np.int32) + int(calendar_id)

        # N_mid_sample is [34, 35, 35, 31] mids in four-hour periods
        # (N_mid_sample, 20)

        all_future_5min_data = np.stack(
            all_future_5min_data).astype(np.float32)
        # (N_mid_sample, 20)
        all_three_data = np.stack(all_three_data).astype(np.int32)

        # (N_mid_sample, 20)
        all_qt_label = np.stack(all_qt_label).astype(np.int32)
        # (N_mid_sample, 20)
        all_mid_data = np.stack(all_mid_data).astype(np.float32)
        # ((N_mid_sample, )
        all_valid_mid_index = np.array(all_valid_mid_index).astype(np.int32)
        # (N_mid_sample, 20)
        all_price_data = np.stack(all_price_data).astype(np.float32)
        # (N_mid_sample, 20, 40, 18)
        all_slice_data = np.stack(all_slice_data).astype(np.float32)
        # print(all_slice_data.shape)
        # , all_mid_data.shape, all_valid_mid_index.shape, all_price_data.shape)

        return all_stock_id, calen_ids, week_ids, \
            all_slice_data, \
            all_future_5min_data, \
            all_qt_label, \
            all_mid_data, \
            all_three_data, \
            all_valid_mid_index, \
            all_price_data
