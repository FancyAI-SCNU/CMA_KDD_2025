from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from util.tools import EarlyStopping, adjust_learning_rate, visual
from util.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pickle
from copy import deepcopy
warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(
                    batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:,
                                  f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def copy_model_and_optimizer(self, model, optimizer):
        """Copy the model and optimizer states for resetting after adaptation."""
        model_state = deepcopy(model.state_dict())
        model_anchor = deepcopy(model)
        optimizer_state = deepcopy(optimizer.state_dict())
        ema_model = deepcopy(model)
        for param in ema_model.parameters():
            param.detach_()
        return model_state, optimizer_state, ema_model, model_anchor

    def softmax_entropy(self, x, x_ema):  # -> torch.Tensor:
        """Entropy of softmax distribution from logits."""
        return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

    def update_ema_variables(self, ema_model, model, alpha_teacher):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + \
                (1 - alpha_teacher) * param[:].data[:]
        return ema_model

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True)
        self.model.load_state_dict(torch.load(os.path.join(
            'checkpoints/' + setting, 'checkpoint.pth')))
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            self.copy_model_and_optimizer(self.model, model_optim)
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            print(sum(p.numel() for p in self.model.parameters()))

            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(
                    batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:,
                                          f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:

                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    anchor_prob = torch.nn.functional.softmax(
                        self.model_anchor(batch_x, batch_x_mark, dec_inp, batch_y_mark), dim=1).max(1)[0]

                    standard_ema = self.model_ema(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    # Augmentation-averaged Prediction
                    N = 32
                    self.steps = 1
                    self.mt = 0.99
                    self.rst = 0.1
                    self.ap = 0.9
                    outputs_emas = []
                    for i in range(N):
                        outputs_ = self.model_ema(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark).detach()
                        outputs_emas.append(outputs_)
                    # Threshold choice discussed in supplementary
                    if anchor_prob.mean(0).mean() < self.ap:
                        outputs_ema = torch.stack(outputs_emas).mean(0)
                    else:
                        outputs_ema = standard_ema

                    loss = (self.softmax_entropy(outputs, outputs_ema)).sum()

                    loss.backward()
                    model_optim.step()
                    model_optim.zero_grad()
                    # Teacher update
                    self.model_ema = self.update_ema_variables(
                        ema_model=self.model_ema, model=self.model, alpha_teacher=self.mt)
                    # Stochastic restore
                    if True:
                        for nm, m in self.model.named_modules():
                            for npp, p in m.named_parameters():
                                if npp in ['weight', 'bias'] and p.requires_grad:
                                    mask = (torch.rand(p.shape) <
                                            self.rst).float().cuda()
                                    with torch.no_grad():
                                        p.data = self.model_state[f"{nm}.{npp}"] * \
                                            mask + p * (1.-mask)

                    # f_dim = -1 if self.args.features == 'MS' else 0
                    # outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    # batch_y = batch_y[:, -self.args.pred_len:,
                    #                   f_dim:].to(self.device)
                    # loss = criterion(outputs, batch_y)
                    # train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                        i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * \
                        ((self.args.train_epochs - epoch) * train_steps - i)
                    print(
                        '\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # if self.args.use_amp:
                #     scaler.scale(loss).backward()
                #     scaler.step(model_optim)
                #     scaler.update()
                # else:
                #     loss.backward()
                #     model_optim.step()

            print("Epoch: {} cost time: {}".format(
                epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(
                'checkpoints/' + setting, 'checkpoint.pth')))
            # self.model.load_state_dict(torch.load(os.path.join(
            #     'pretrain_checkpoints/Check_itrans_60/', 'checkpoint_shanghai.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            t = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(
                    batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:,
                                  f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                # if test_data.scale and self.args.inverse:
                #   shape = outputs.shape
                #   outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                #   batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = outputs
                true = batch_y

                scaler_file = 'ecl_scaler.sav'
                scaler = pickle.load(open(scaler_file, 'rb'))

                if len(pred.shape) > 2:
                    for i in range(pred.shape[0]):
                        pred[i, :, :] = scaler.inverse_transform(pred[i, :, :])
                        true[i, :, :] = scaler.inverse_transform(true[i, :, :])
                else:
                    true = scaler.inverse_transform(true)
                    pred = scaler.inverse_transform(pred)
                preds.append(pred)
                trues.append(true)
                # if i % 20 == 0:
                #   input = batch_x.detach().cpu().numpy()
                #    if test_data.scale and self.args.inverse:
                #        shape = input.shape
                #        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                #    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
        print(time.time()-t)
        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # print(preds[:, :, -1:], trues[:, :, -1:])
        # exit()
        mae, mse, rmse, mape, mspe = metric(
            preds[:, :, -1:], trues[:, :, -1:])
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy',
                np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(
                    batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(
                        outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
