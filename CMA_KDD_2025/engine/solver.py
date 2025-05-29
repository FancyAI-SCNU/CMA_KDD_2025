import os
import sys
import time
import torch
import numpy as np
# import lightning as L

from pathlib import Path
from tqdm.auto import tqdm
from ema_pytorch import EMA
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from Utils.io_utils import instantiate_from_config, get_model_parameters_info
import learn2learn as l2l
from learn2learn.utils import clone_module, update_module
# from Models.interpretable_diffusion.AF_pretrain import Model
from Models.interpretable_diffusion.Itransformer_pretrain import ITransformer
from Models.interpretable_diffusion.TimeXer import Model

from peft import LoraConfig, get_peft_model

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))


def cycle(dl):
    while True:
        for data in dl:
            yield data

# class LitAutoEncoder(L.LightingModule):
#     def __init__(self):
#         super().__init__()
#         self.


class Trainer(object):
    def __init__(self, config, args, model, dataloader, logger=None):
        super().__init__()
        self.model = model

        self.device = self.model.betas.device
        self.train_num_steps = config['solver']['max_epochs']
        self.gradient_accumulate_every = config['solver']['gradient_accumulate_every']
        self.save_cycle = config['solver']['save_cycle']
        self.dl = cycle(dataloader['dataloader'])
        self.step = 0
        self.milestone = 0
        self.args = args
        self.logger = logger

        self.adapt_num = config['dataloader']['test_dataset']['params']['adapt']

        self.results_folder = Path(
            config['solver']['results_folder'] + f'_{model.seq_length}')

        os.makedirs(self.results_folder, exist_ok=True)

        start_lr = config['solver'].get('base_lr', 1.0e-4)
        ema_decay = config['solver']['ema']['decay']
        ema_update_every = config['solver']['ema']['update_interval']
        print(start_lr, "start")
        self.start_lr = start_lr
        self.adapt_lr = start_lr * 0.4  # 0.6 #1.2 carbon,traffic:0.9 carbon0.7
        # self.opt = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=start_lr, betas=[0.9, 0.96],eps=1e-4)
        self.maml = l2l.algorithms.MAML(self.model, lr=self.adapt_lr, first_order=True,
                                        allow_unused=True)
        self.opt = Adam(self.maml.parameters(), lr=start_lr,
                        betas=(0.9, 0.96), eps=1e-4)
        # self.scheduler_model = torch.optim.lr_scheduler.MultiStepLR(self.opt, milestones=[60, 120], gamma=0.6)
        self.ema = EMA(self.model, beta=ema_decay,
                       update_every=ema_update_every).to(self.device)
        moving_avg = 25
        # self.af = Model(n_feat=config['model']['params']['feature_size'], seq_len=config['model']['params']['seq_length'],feat_len=config['model']['params']['feat_len'], moving_avg=moving_avg,dropout=config['model']['params']['attn_pd'],factor=config['solver']['scheduler']['params']['factor'],n_layer_enc=2, n_layer_dec=1,
        #                          n_embd=config['model']['params']['d_model'],n_heads=config['model']['params']['n_heads'],activation='GELU',output_attention = False,freq = 'h' ,embed_type = "timeF")
        self.af = Model(n_feat=17, seq_len=config['model']['params']['seq_length'], feat_len=config['model']['params']['feat_len'], moving_avg=moving_avg, dropout=config['model']['params']['attn_pd'], factor=config['solver']['scheduler']['params']['factor'], n_layer_enc=1, n_layer_dec=1,
                        n_embd=config['model']['params']['d_model'], n_heads=8, activation='GELU', output_attention=False, freq='h', embed_type="timeF")
        # self.itrans = ITransformer(n_feat=config['model']['params']['feature_size'], seq_len=config['model']['params']['seq_length'],feat_len=config['model']['params']['feat_len'], moving_avg=moving_avg,dropout=config['model']['params']['attn_pd'],factor=config['solver']['scheduler']['params']['factor'],n_layer_enc=4, n_layer_dec=1,
        #                          n_embd=config['model']['params']['d_model'],n_heads=config['model']['params']['n_heads'],activation='GELU',output_attention = False,freq = 'h' ,embed_type = "timeF")
        self.itrans = ITransformer(n_feat=config['model']['params']['feature_size'], seq_len=config['model']['params']['seq_length'], feat_len=config['model']['params']['feat_len'], moving_avg=moving_avg, dropout=config['model']['params']['attn_pd'], factor=config['solver']['scheduler']['params']['factor'], n_layer_enc=3, n_layer_dec=0,
                                   n_embd=512, n_heads=8, activation='GELU', output_attention=False, freq='h', embed_type="timeF")

        # self.itrans = ITransformer(n_feat=config['model']['params']['feature_size'], seq_len=config['model']['params']['seq_length'],feat_len=config['model']['params']['feat_len'], moving_avg=moving_avg,dropout=config['model']['params']['attn_pd'],factor=config['solver']['scheduler']['params']['factor'],n_layer_enc=2, n_layer_dec=1,
        #                          n_embd=128,n_heads=config['model']['params']['n_heads'],activation='GELU',output_attention = False,freq = 'h' ,embed_type = "timeF")

        self.af_choice = True
        self.incontext = True
        self.time = True
        self.name = config['dataloader']['train_dataset']['params']['name']
        self.feat_len = config['model']['params']['feat_len']
        self.pred_len = config['model']['params']['seq_length']
        self.adapt_h = config['dataloader']['train_dataset']['params']['adapt_h']
        self.adapt_f_num = config['dataloader']['train_dataset']['params']['adapt']
        if self.adapt_f_num == 0:
            self.adapt_f = False
        else:
            self.adapt_f = True
        sc_cfg = config['solver']['scheduler']
        sc_cfg['params']['optimizer'] = self.opt

        self.sch = instantiate_from_config(sc_cfg)

        if self.logger is not None:
            self.logger.log_info(str(get_model_parameters_info(self.model)))
        self.log_frequency = 100

    def save(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Save current model to {}'.format(
                str(self.results_folder / f'checkpoint-{milestone}.pt')))
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict(),
        }

        torch.save(data, str(self.results_folder /
                   f'checkpoint-{milestone}.pt'))

    def load(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Resume from {}'.format(
                str(self.results_folder / f'checkpoint-{milestone}.pt')))

        device = self.device
        data = torch.load(
            str(self.results_folder / f'checkpoint-{milestone}.pt'), map_location=device)
        self.model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])
        self.milestone = milestone

    def load_meta(self, milestone, verbose=False):

        if self.logger is not None and verbose:
            self.logger.log_info('Resume from {}'.format(
                str(self.results_folder / f'checkpoint-org-{milestone}.pt')))
        device = self.device
        data = torch.load(
            str(self.results_folder / f'checkpoint-org-{milestone}.pt'), map_location=device)
        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])
        self.milestone = milestone

    def train(self):
        # self.model.train()
        self.opt = Adam(self.model.parameters(),
                        lr=self.start_lr, betas=(0.9, 0.96), eps=1e-4)
        device = self.device
        step = 0
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('{}: start training...'.format(
                self.args.name), check_primary=False)

        with tqdm(initial=step, total=self.train_num_steps) as pbar:
            while step < self.train_num_steps:

                total_loss = 0.

                total_loss_meta = 0.
                for _ in range(self.gradient_accumulate_every):

                    data_raw = next(self.dl)
                    data = data_raw[0].to(device)
                    data_mark = data_raw[1].to(device)

                    # train denoise transformer  len:24->24
                    # loss = self.model(data,data_mark ,target=data)
                    loss = self.model(data, data_mark, target=data)

                    # loss = self.model(data ,target=data)
                    loss = loss / self.gradient_accumulate_every

                    loss.backward()

                    total_loss += loss.item()
                    # total_loss_meta +=loss_meta.item()
                pbar.set_description(f'loss: {total_loss:.6f}')
                clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.sch.step(total_loss)
                self.opt.zero_grad()
                self.step += 1
                step += 1
                self.ema.update()

                with torch.no_grad():
                    if self.step != 0 and self.step % self.save_cycle == 0:
                        self.milestone += 1
                        self.save(self.milestone)
                        # self.logger.log_info('saved in {}'.format(str(self.results_folder / f'checkpoint-{self.milestone}.pt')))

                    if self.logger is not None and self.step % self.log_frequency == 0:
                        # info = '{}: train'.format(self.args.name)
                        # info = info + ': Epoch {}/{}'.format(self.step, self.train_num_steps)
                        # info += ' ||'
                        # info += '' if loss_f == 'none' else ' Fourier Loss: {:.4f}'.format(loss_f.item())
                        # info += '' if loss_r == 'none' else ' Reglarization: {:.4f}'.format(loss_r.item())
                        # info += ' | Total Loss: {:.6f}'.format(total_loss)
                        # self.logger.log_info(info)
                        self.logger.add_scalar(
                            tag='train/loss', scalar_value=total_loss, global_step=self.step)

                pbar.update(1)

        print('training complete')
        if self.logger is not None:
            self.logger.log_info(
                'Training done, time: {:.2f}'.format(time.time() - tic))
        num_params = sum(p.numel()
                         for p in self.model.parameters() if p.requires_grad)
        print(num_params)

    def train_meta(self):
        # self.model.train()
        self.opt = Adam(self.maml.parameters(), lr=self.start_lr,
                        betas=(0.9, 0.96), eps=1e-4)
        device = self.device
        step = 0
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('{}: start training...'.format(
                self.args.name), check_primary=False)
        if self.af_choice:
            if self.time:
                self.af.load_state_dict(torch.load(os.path.join(
                    './pretrain_checkpoints/Check_time_90/', 'checkpoint_'+self.name+'.pth')))
            else:
                self.itrans.load_state_dict(torch.load(os.path.join(
                    './pretrain_checkpoints/Check_itrans_90/', 'checkpoint_'+self.name+'.pth')))

            # self.af.load_state_dict(torch.load(os.path.join('/home/shannon_research/share/jhq/pretrain_checkpoints/Check_af/' , 'checkpoint_'+self.name+'.pth')))

        with tqdm(initial=step, total=self.train_num_steps) as pbar:
            while step < self.train_num_steps:
                total_loss = 0.

                total_loss_meta = 0.
                for _ in range(self.gradient_accumulate_every):
                    if self.af_choice:
                        data_raw = next(self.dl)
                        # af_pred = self.af(data_raw[0][:,:self.feat_len,:],data_raw[1][:,:self.feat_len,:],\
                        #                 data_raw[0][:,self.feat_len//2:,:],data_raw[1][:,self.feat_len//2:,:])

                        itrans_pred = self.itrans(data_raw[0][:, :self.feat_len, :], data_raw[1][:, :self.feat_len, :],
                                                  data_raw[0][:, self.feat_len//2:, :], data_raw[1][:, self.feat_len//2:, :])
                        # if self.incontext == False:
                        data = data_raw[0].to(device)
                        data_mark = data_raw[1].to(device)
                        # train denoise transformer  len:24->24

                        loss = self.model.forward_meta(
                            data, data_mark, self.maml, af_pred=itrans_pred, target=data)
                        # else:
                        #     data = data_raw[0].to(device)
                        #     data_mark = data_raw[1].to(device)
                        #     # train denoise transformer  len:24->24
                        #     data_incontext = data_raw[2].to(device)
                        #     data_incontext_mark = data_raw[3].to(device)

                        #     loss = self.model.forward_meta_incontext(data,data_mark,data_incontext,data_incontext_mark,\
                        #                                              self.maml,af_pred=itrans_pred,target=data)
                    else:
                        data_raw = next(self.dl)
                        data = data_raw[0].to(device)
                        data_mark = data_raw[1].to(device)

                        loss = self.model.forward_meta(
                            data, data_mark, self.maml, target=data)
                    # # loss = self.model(data,data_mark ,target=data)
                    # train_loss = self.loss_fn(y_pred_final, self.target, reduction='none')

                    # loss = self.model(data ,target=data)
                    loss = loss / self.gradient_accumulate_every

                    loss.backward()

                    total_loss += loss.item()
                    # total_loss_meta +=loss_meta.item()
                pbar.set_description(f'loss: {total_loss:.6f}')
                clip_grad_norm_(self.maml.parameters(), 1.0)
                self.opt.step()
                self.sch.step(total_loss)
                self.opt.zero_grad()
                self.step += 1
                step += 1
                self.ema.update()

                with torch.no_grad():
                    if self.step != 0 and self.step % self.save_cycle == 0:
                        self.milestone += 1
                        self.save(self.milestone)
                        # self.logger.log_info('saved in {}'.format(str(self.results_folder / f'checkpoint-{self.milestone}.pt')))

                    if self.logger is not None and self.step % self.log_frequency == 0:
                        # info = '{}: train'.format(self.args.name)
                        # info = info + ': Epoch {}/{}'.format(self.step, self.train_num_steps)
                        # info += ' ||'
                        # info += '' if loss_f == 'none' else ' Fourier Loss: {:.4f}'.format(loss_f.item())
                        # info += '' if loss_r == 'none' else ' Reglarization: {:.4f}'.format(loss_r.item())
                        # info += ' | Total Loss: {:.6f}'.format(total_loss)
                        # self.logger.log_info(info)
                        self.logger.add_scalar(
                            tag='train/loss', scalar_value=total_loss, global_step=self.step)

                pbar.update(1)

        print('training complete')
        if self.logger is not None:
            self.logger.log_info(
                'Training done, time: {:.2f}'.format(time.time() - tic))
        num_params = sum(p.numel()
                         for p in self.model.parameters() if p.requires_grad)
        print(num_params)

    def sample(self, num, size_every, shape=None):
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to sample...')
        samples = np.empty([0, shape[0], shape[1]])
        num_cycle = int(num // size_every) + 1

        for _ in range(num_cycle):

            sample = self.ema.ema_model.generate_mts(batch_size=size_every)
            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            torch.cuda.empty_cache()

        if self.logger is not None:
            self.logger.log_info(
                'Sampling done, time: {:.2f}'.format(time.time() - tic))
        return samples

    def restore(self, raw_dataloader, shape=None, coef=1e-1, stepsize=1e-1, sampling_steps=50):
        self.model.eval()
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to restore...')
        model_kwargs = {}
        model_kwargs['coef'] = coef
        model_kwargs['learning_rate'] = stepsize
        samples = np.empty([0, shape[0], shape[1]])
        reals = np.empty([0, shape[0], shape[1]])
        masks = np.empty([0, shape[0], shape[1]])

        # mse = []
        for idx, (x, t_m, x_mark, x_incontext, x_in_mark) in enumerate(raw_dataloader):

            x, t_m, x_mark = x.to(self.device), t_m.to(
                self.device), x_mark.to(self.device)

            # if sampling_steps == self.model.num_timesteps:
            #    sample = self.ema.ema_model.sample_infill(x=x,x_mark=x_mark,shape=x.shape, target=x*t_m, partial_mask=t_m,
            #                                              )
            # else:
            #    sample = self.ema.ema_model.fast_sample_infill(x=x,x_mark=x_mark,shape=x.shape, target=x*t_m, partial_mask=t_m,
            #
            #                                                   sampling_timesteps=sampling_steps)

            sample = self.ema.ema_model.fast_sample_pre(x=x, x_mark=x_mark, shape=x.shape, target=x*t_m, partial_mask=t_m,
                                                        model_kwargs=model_kwargs,
                                                        sampling_timesteps=sampling_steps)

            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            # reals = np.row_stack([reals, x.detach().cpu().numpy()])
            # masks = np.row_stack([masks, t_m.detach().cpu().numpy()])

        # print(np.mean(mse))
        if self.logger is not None:
            self.logger.log_info(
                'Imputation done, time: {:.2f}'.format(time.time() - tic))
        return samples, reals, masks
        # return samples

    def restore_adapt(self, raw_dataloader, shape=None, coef=1e-1, stepsize=1e-1, sampling_steps=50, future_incontext=False):
        # self.model.train()
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to restore...')
        model_kwargs = {}
        model_kwargs['coef'] = coef
        model_kwargs['learning_rate'] = stepsize

        samples = np.empty([0, shape[0], shape[1]])
        reals = np.empty([0, shape[0], shape[1]])
        masks = np.empty([0, shape[0], shape[1]])

        if future_incontext == True:

            samples = np.empty([0, shape[0]*(self.adapt_num), shape[1]])
            reals = np.empty([0, shape[0]*(self.adapt_num), shape[1]])
            masks = np.empty([0, shape[0]*(self.adapt_num), shape[1]])

        if self.af_choice:
            if self.time:
                self.af.load_state_dict(torch.load(os.path.join(
                    './pretrain_checkpoints/Check_time_90/', 'checkpoint_'+self.name+'.pth')))
            else:
                self.itrans.load_state_dict(torch.load(os.path.join(
                    './pretrain_checkpoints/Check_itrans_90/', 'checkpoint_'+self.name+'.pth')))

        # mse = []
        for idx, (x, t_m, x_mark, x_incontext, x_in_mark) in enumerate(raw_dataloader):
            x, t_m, x_mark, x_incontext, x_in_mark = x.to(self.device), t_m.to(self.device), \
                x_mark.to(self.device), x_incontext.to(
                    self.device), x_in_mark.to(self.device)

            # sample = self.ema.ema_model.fast_sample_pre(x=x,x_mark=x_mark,shape=x.shape, target=x*t_m, partial_mask=t_m,
            #                                                 model_kwargs=model_kwargs,
            #                                                 sampling_timesteps=sampling_steps)

            # if sampling_steps == self.model.num_timesteps:
            # self.model.zero_grad()

            # for name, param in self.model.named_parameters():
            #     print(name, param.numel())
            # total = sum(p.numel() for p in self.model.parameters())
            # print(total)

            lora = True
            if lora:
                lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, target_modules=[
                                         'query_projection', 'value_projection'])
                model_ = get_peft_model(self.model, lora_config)
                # for name, param in model_.named_parameters():
                #     print(name, param.numel())
                # total = sum(p.numel()
                #             for p in model_.parameters())
                # print(total)
                # exit()
                maml = l2l.algorithms.MAML(model_, lr=self.adapt_lr, first_order=True,
                                           allow_unused=True, allow_nograd=True)
            else:
                maml = l2l.algorithms.MAML(self.model, lr=self.adapt_lr, first_order=True,
                                           allow_unused=True)

            if self.af_choice:
                # af_pred = self.af(x[:,:self.feat_len,:].cpu().to(torch.float32),x_mark[:,:self.feat_len,:].cpu().to(torch.float32),\
                #                     x[:,self.feat_len//2:,:].cpu().to(torch.float32),x_mark[:,self.feat_len//2:,:].cpu().to(torch.float32))

                af_pred = self.af(x[:, :self.feat_len, :].cpu().to(torch.float32), x_mark[:, :self.feat_len, :].cpu().to(torch.float32),
                                  x[:, self.feat_len//2:, :].cpu().to(torch.float32), x_mark[:, self.feat_len//2:, :].cpu().to(torch.float32))

                output = []
                if len(x_incontext.shape) > 3:
                    for kk in range(x_incontext.shape[1]):
                        x_input = x_incontext[:, kk, :, :]
                        x_input_mark = x_in_mark[:, kk, :, :]
                        af_ = self.itrans(x_input[:, :self.feat_len, :].cpu().to(torch.float32), x_input_mark[:, :self.feat_len, :].cpu().to(torch.float32),
                                          x_input[:, self.feat_len//2:, :].cpu().to(torch.float32), x_input_mark[:, self.feat_len//2:, :].cpu().to(torch.float32))

                        output.append(af_)

                # af_pred -- (64,48,17)

                # if future_incontext == False:
                #     sample= self.model.sample_infill_adapt(x=x,x_mark=x_mark,shape=x.shape, target=x*t_m, partial_mask=t_m,maml=maml,af_pred=af_pred,
                #                                             model_kwargs=model_kwargs)
                # else:
                #     print("2222222")

                sample = self.model.sample_infill_adapt_incontext(x=x, x_mark=x_mark, x_incontext=x_incontext, x_in_mark=x_in_mark, shape=x.shape, target=x*t_m,
                                                                  partial_mask=t_m, maml=maml, af_pred=af_pred, af_incontext=output, adapt_h=self.adapt_h, adapt_f=self.adapt_f,
                                                                  adapt_f_num=self.adapt_f_num, model_kwargs=model_kwargs)
            else:
                sample = self.model.sample_infill_adapt(x=x, x_mark=x_mark, shape=x.shape, target=x*t_m, partial_mask=t_m, maml=maml,
                                                        model_kwargs=model_kwargs)

            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            # reals = np.row_stack([reals, x.detach().cpu().numpy()])
            # masks = np.row_stack([masks, t_m.detach().cpu().numpy()])

        # print(np.mean(mse))
        if self.logger is not None:
            self.logger.log_info(
                'Imputation done, time: {:.2f}'.format(time.time() - tic))
        return samples, reals, masks
        # return samples

    def restore_jump(self, raw_dataloader, shape=None, coef=1e-1, stepsize=1e-1, sampling_steps=50):
        # self.model.train()
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to restore...')
        model_kwargs = {}
        model_kwargs['coef'] = coef
        model_kwargs['learning_rate'] = stepsize
        samples = np.empty([0, shape[0], shape[1]])
        reals = np.empty([0, shape[0], shape[1]])
        masks = np.empty([0, shape[0], shape[1]])

        # mse = []
        for idx, (x, t_m, x_mark) in enumerate(raw_dataloader):
            x, t_m, x_mark = x.to(self.device), t_m.to(
                self.device), x_mark.to(self.device)

            # sample = self.ema.ema_model.fast_sample_pre(x=x,x_mark=x_mark,shape=x.shape, target=x*t_m, partial_mask=t_m,
            #                                                 model_kwargs=model_kwargs,
            #                                                 sampling_timesteps=sampling_steps)

            # if sampling_steps == self.model.num_timesteps:
            # self.model.zero_grad()

            sample = self.model.sample_infill_no_adapt(x=x, x_mark=x_mark, shape=x.shape, target=x*t_m, partial_mask=t_m,
                                                       model_kwargs=model_kwargs)
            # else:
            #     sample = self.model.fast_sample_infill_adapt(x=x,x_mark=x_mark,shape=x.shape, target=x*t_m, partial_mask=t_m,
            #                                                 model_kwargs=model_kwargs,
            #                                                 sampling_timesteps=sampling_steps)

            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            # reals = np.row_stack([reals, x.detach().cpu().numpy()])
            # masks = np.row_stack([masks, t_m.detach().cpu().numpy()])

        # print(np.mean(mse))
        if self.logger is not None:
            self.logger.log_info(
                'Imputation done, time: {:.2f}'.format(time.time() - tic))
        return samples, reals, masks
        # return samples
