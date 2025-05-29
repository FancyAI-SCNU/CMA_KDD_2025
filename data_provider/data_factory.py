from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar, Dataset_PEMS, \
    Dataset_Pred
from torch.utils.data import DataLoader
from Utils.Data_utils.stock_sequence_dataset_label5min import public_stock_dataset_train, public_stock_dataset_vali, public_stock_dataset_test
from Utils.Data_utils.real_datasets_all_adapt_us import CustomDataset
data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Solar': Dataset_Solar,
    'PEMS': Dataset_PEMS,
    'custom': Dataset_Custom,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
    )
    # if flag == 'train':
    #     data_set = CustomDataset(
    #         name='stock',
    #         data_root=None,
    #         window=args.seq_len + args.pred_len,
    #         feat_len=args.seq_len,
    #         proportion=0.8,
    #         save2npy=True,
    #         neg_one_to_one=True,
    #         seed=123,
    #         period=flag,
    #         output_dir='./OUTPUT',
    #         predict_length=args.pred_len,
    #         missing_ratio=None,
    #         style='separate',
    #         distribution='geometric',
    #         adapt=0,
    #         adapt_h=False,
    #         adapt_num_step=5,
    #         mean_mask_length=3,
    #         seq_len=args.pred_len
    #     )
    # else:
    #     _ = CustomDataset(
    #         name='stock',
    #         data_root=None,
    #         window=args.seq_len + args.pred_len,
    #         feat_len=args.seq_len,
    #         proportion=0.8,
    #         save2npy=True,
    #         neg_one_to_one=True,
    #         seed=123,
    #         period=flag,
    #         output_dir='./OUTPUT',
    #         predict_length=args.pred_len,
    #         missing_ratio=None,
    #         style='separate',
    #         distribution='geometric',
    #         adapt=0,
    #         adapt_h=False,
    #         adapt_num_step=5,
    #         mean_mask_length=3,
    #         seq_len=args.pred_len
    #     )
    #     data_set = CustomDataset(
    #         name='stock',
    #         data_root=None,
    #         window=args.seq_len + args.pred_len,
    #         feat_len=args.seq_len,
    #         proportion=0.8,
    #         save2npy=True,
    #         neg_one_to_one=True,
    #         seed=123,
    #         period=flag,
    #         output_dir='./OUTPUT',
    #         predict_length=args.pred_len,
    #         missing_ratio=None,
    #         style='separate',
    #         distribution='geometric',
    #         adapt=0,
    #         adapt_h=False,
    #         adapt_num_step=5,
    #         mean_mask_length=3,
    #         seq_len=args.pred_len
    #     )

    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
