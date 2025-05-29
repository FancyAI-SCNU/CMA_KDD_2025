import math
import torch
import torch.nn.functional as F

from torch import nn
from einops import reduce
from tqdm.auto import tqdm
from functools import partial
# from Models.interpretable_diffusion.transformer import Transformer
# from Models.interpretable_diffusion.Autoformer import Transformer
# from Models.interpretable_diffusion.Autoformer_attn import Transformer
from Models.interpretable_diffusion.Autoformer_all import Model
# from Models.interpretable_diffusion.Fedformer import Model
from Models.interpretable_diffusion.Itransformer import ITransformer
from Models.interpretable_diffusion.model_utils import default, identity, extract
import learn2learn as l2l
from learn2learn.utils import clone_module, update_module

# gaussian diffusion trainer class


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class Diffusion_TS(nn.Module):
    def __init__(
            self,
            seq_length,
            feat_len,
            feature_size,
            # moving_avg=[25],
            moving_avg=25,
            factor=3,
            n_layer_enc=3,
            n_layer_dec=6,
            d_model=None,
            timesteps=1000,
            sampling_timesteps=None,
            loss_type='l1',
            beta_schedule='cosine',
            n_heads=8,
            mlp_hidden_times=4,
            eta=0.,
            attn_pd=0.,
            resid_pd=0.,
            kernel_size=None,
            padding_size=None,
            use_ff=True,
            reg_weight=None,
            **kwargs
    ):
        super(Diffusion_TS, self).__init__()

        self.eta, self.use_ff = eta, use_ff

        self.seq_length = seq_length
        self.feat_len = feat_len
        self.feature_size = feature_size
        self.ff_weight = default(reg_weight, math.sqrt(self.seq_length) / 5)
        learning_rate = 5.0-2
        self.adapt_lr = learning_rate * 0.6

        self.model = ITransformer(n_feat=feature_size, seq_len=seq_length, feat_len=feat_len, n_layer_enc=n_layer_enc, n_layer_dec=n_layer_dec,
                                  n_embd=d_model, n_heads=n_heads, attn_pdrop=attn_pd, resid_pdrop=resid_pd, mlp_hidden_times=mlp_hidden_times,
                                  max_len=seq_length, **kwargs)

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.pretrain_step = 480
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(
            sampling_timesteps, timesteps)  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.fast_sampling = self.sampling_timesteps < timesteps

        # helper function to register buffer from float64 to float32

        def register_buffer(name, val): return self.register_buffer(
            name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod',
                        torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod',
                        torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas *
                        torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev)
                        * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate reweighting

        register_buffer('loss_weight', torch.sqrt(alphas) *
                        torch.sqrt(1. - alphas_cumprod) / betas / 100)

    def predict_noise_from_start(self, x_t, t, x0):

        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_start_from_noise(self, x_t, t, noise):

        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def output(self, x, x_feat, x_pred_mark, x_feat_mark, t, padding_masks=None):

        # model_output = self.model(x_enc=x_feat,x_mark_enc=x_feat_mark,x_dec=x,x_mark_dec=x_pred_mark,t=t,
        #                             enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None)
        model_output = self.model(x_enc=x_feat, x_mark_enc=x_feat_mark, x_dec=x, x_mark_dec=x_pred_mark, t=t,
                                  )
        # model_output = trend + season

        return model_output

    def model_predictions(self, all, all_mark, x, t, clip_x_start=False, padding_masks=None):

        if padding_masks is None:
            padding_masks = torch.ones(
                x.shape[0], self.seq_length, dtype=bool, device=x.device)

        x_feat = all[:, :self.feat_len, :].to(torch.float32)
        x_feat_mark = all_mark[:, :self.feat_len, :].to(torch.float32)
        x_pred_mark = all_mark[:, self.feat_len//2:, :].to(torch.float32)
        maybe_clip = partial(torch.clamp, min=-1.,
                             max=1.) if clip_x_start else identity

        x_start = self.output(x, x_feat, x_pred_mark,
                              x_feat_mark, t, padding_masks)

        x_start = maybe_clip(x_start)

        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start

    def model_predictions_inf(self, all, all_mark, x, t, clip_x_start=False, padding_masks=None):

        if padding_masks is None:
            padding_masks = torch.ones(
                x.shape[0], self.seq_length, dtype=bool, device=x.device)

        x_feat = all[:, :self.feat_len, :].to(torch.float32)
        x_feat_mark = all_mark[:, :self.feat_len, :].to(torch.float32)
        x_pred_mark = all_mark[:, self.feat_len:, :].to(torch.float32)
        maybe_clip = partial(torch.clamp, min=-1.,
                             max=1.) if clip_x_start else identity

        x_start = self.output(x, x_feat, x_pred_mark,
                              x_feat_mark, t, padding_masks)

        x_start = maybe_clip(x_start)

        pred_noise = self.predict_noise_from_start(x, t, x_start)

        x_start = x_start.detach()
        pred_noise = pred_noise.detach()

        return pred_noise, x_start

    def p_mean_variance(self, all, all_mark, x, t, clip_denoised=True):
        _, x_start = self.model_predictions(all, all_mark, x, t)
        if clip_denoised:
            x_start.clamp_(-1., 1.)
        model_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def p_sample(self, all, all_mark, x, t: int, clip_denoised=True):
        batched_times = torch.full(
            (x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = \
            self.p_mean_variance(all=all, all_mark=all_mark,
                                 x=x, t=batched_times, clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def sample(self, shape):
        device = self.betas.device
        img = torch.randn(shape, device=device)
        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='sampling loop time step', total=self.num_timesteps):
            img, _ = self.p_sample(img, t)
        return img

    @torch.no_grad()
    def fast_sample(self, shape, clip_denoised=True):
        batch, device, total_timesteps, sampling_timesteps, eta = \
            shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        time_pairs = list(zip(times[:-1], times[1:]))
        img = torch.randn(shape, device=device)

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full(
                (batch,), time, device=device, dtype=torch.long)
            # pred_noise, x_start, *_ = self.model_predictions(x,x_mark,img, time_cond, clip_x_start=clip_denoised)

            pred_noise, x_start, * \
                _ = self.model_predictions(
                    img, time_cond, clip_x_start=clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) *
                           (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + \
                c * pred_noise + \
                sigma * noise

        return img

    def generate_mts(self, batch_size=16):
        feature_size, seq_length = self.feature_size, self.seq_length
        sample_fn = self.fast_sample if self.fast_sampling else self.sample
        return sample_fn((batch_size, seq_length, feature_size))

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )

    # adapt+final loss
    def p_sample_infill_train_loop_accelerate(self, maml, num_step, x, x_mark, time_cond, adapt_grad=True, af_pred=None, **kwargs):
        model_adapt = maml.clone()
        x_feat = x[:, :self.feat_len, :]
        x_pred_mark = x_mark[:, self.feat_len:, :]
        x_feat_mark = x_mark[:, :self.feat_len, :]

        target = x[:, self.feat_len//2:, :]
        self.target = target
        times = torch.linspace(self.num_timesteps-self.pretrain_step-1,
                               self.num_timesteps - 1, steps=self.sampling_timesteps + 1)  # [201]

        times = list(reversed(times.int().tolist()))
        # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        time_pairs_ = list(zip(times[:-1], times[1:]))

        img = torch.randn(x[:, self.feat_len//2:, :].shape,
                          device=x.device)  # torch.Size([48, 144, 21])
        img_noise = img

        for time, time_next in time_pairs_:

            pred_noise, x_start, * \
                _ = self.model_predictions_inf(x, x_mark, img, time_cond)

            if time_next < num_step:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = self.eta * ((1 - alpha / alpha_next) *
                                (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
            noise = torch.randn_like(img)

            img = pred_mean + sigma * noise

        y_pred_raw = img.detach()

        partial_mask_2 = torch.zeros(
            y_pred_raw[:, self.feat_len//2:, :].size(), dtype=torch.bool)
        partial_mask_1 = torch.ones(
            y_pred_raw[:, :self.feat_len//2, :].size(), dtype=torch.bool)
        partial_mask = torch.cat([partial_mask_1, partial_mask_2], dim=1)

        def get_sample_fast(n_step, cur_y, model_adapt, adapt_grad=True):

            n_count = 0
            times = torch.linspace(-1, self.num_timesteps -
                                   self.pretrain_step - 1, steps=num_step + 1)  # [201]
            times = list(reversed(times.int().tolist()))
            # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
            time_pairs = list(zip(times[:-1], times[1:]))
            for i in reversed(range(num_step)):

                t = time_pairs[n_count][0]
                t = torch.tensor([t]).to(x.device)

                x_start = model_adapt.output(
                    cur_y, x_feat, x_pred_mark, x_feat_mark, t)
                if time_pairs[n_count][1] < 0:
                    cur_y = x_start
                    continue
                maybe_clip = partial(torch.clamp, min=-1., max=1.)
                x_start = maybe_clip(x_start)
                pred_noise = self.predict_noise_from_start(cur_y, t, x_start)

                alpha = self.alphas_cumprod[time_pairs[n_count][0]]
                alpha_next = self.alphas_cumprod[time_pairs[n_count][1]]
                sigma = self.eta * ((1 - alpha / alpha_next)
                                    * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()
                pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
                noise = torch.randn_like(cur_y)
                pred_img = pred_mean + sigma * noise

                if adapt_grad:
                    cur_y = pred_img.detach()
                # else:

                #     cur_y = pred_img

                n_count += 1
                if n_count >= n_step:
                    if adapt_grad:
                        model_adapt = self.langevin_fn_train_fast(x=x, x_mark=x_mark, sample=pred_img, mean=pred_mean, sigma=sigma, t=t,
                                                                  tgt_embs=target, partial_mask=partial_mask, model_initializer=model_adapt)
                # else:
                #     print('haha', n_count, i)

                    break
            return cur_y, model_adapt
        # 这里的n_step是20

        def get_sample(n_step, cur_y, model_adapt, adapt_grad=True):
            n_count = 0

            for i in reversed(range(num_step)):

                t = torch.tensor([i]).to(x.device)

                x_start = model_adapt.output(
                    cur_y, x_feat, x_pred_mark, x_feat_mark, t)

                model_mean, _, posterior_log_variance = \
                    self.q_posterior(x_start=x_start, x_t=cur_y, t=t)
                noise = torch.randn_like(cur_y) if t > 0 else 0.
                sigma = (0.5 * posterior_log_variance).exp()
                pred_img = model_mean + sigma * noise

                if adapt_grad:
                    cur_y = pred_img.detach()

                n_count += 1
                if n_count == num_step:
                    cur_y = pred_img
                    break
                if n_count >= n_step:
                    if adapt_grad:
                        model_adapt = self.langevin_fn_train_fast(x=x, x_mark=x_mark, sample=pred_img, mean=model_mean, sigma=sigma, t=t,
                                                                  tgt_embs=target, partial_mask=partial_mask, model_initializer=model_adapt)
                # else:
                #     print('haha', n_count, i)

                    break
            return cur_y, model_adapt

        def get_sample_fast_first(n_step, cur_y, model_adapt, adapt_grad=True):

            n_count = 0
            times = torch.linspace(-1, self.num_timesteps -
                                   self.pretrain_step - 1, steps=num_step + 1)  # [201]
            times = list(reversed(times.int().tolist()))
            # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
            time_pairs = list(zip(times[:-1], times[1:]))
            for i in reversed(range(num_step)):

                t = time_pairs[n_count][0]
                t = torch.tensor([t]).to(x.device)

                x_start = self.output(
                    cur_y, x_feat, x_pred_mark, x_feat_mark, t)
                if time_pairs[n_count][1] < 0:
                    cur_y = x_start
                    continue
                maybe_clip = partial(torch.clamp, min=-1., max=1.)
                x_start = maybe_clip(x_start)
                pred_noise = self.predict_noise_from_start(cur_y, t, x_start)

                alpha = self.alphas_cumprod[time_pairs[n_count][0]]
                alpha_next = self.alphas_cumprod[time_pairs[n_count][1]]
                sigma = self.eta * ((1 - alpha / alpha_next)
                                    * (1 - alpha_next) / (1 - alpha)).sqrt()

                c = (1 - alpha_next - sigma ** 2).sqrt()
                pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
                noise = torch.randn_like(cur_y)
                pred_img = pred_mean + sigma * noise

                if adapt_grad:
                    cur_y = pred_img.detach()

                n_count += 1
                if n_count >= n_step:
                    if adapt_grad:
                        model_adapt = self.langevin_fn_train_fast_first(x=x, x_mark=x_mark, sample=pred_img, mean=pred_mean, sigma=sigma, t=t,
                                                                        tgt_embs=target, partial_mask=partial_mask, model_initializer=model_adapt)
                # else:
                #     print('haha', n_count, i)

                    break
            return cur_y, model_adapt
        # 这里的n_step是20

        def get_sample_first(n_step, cur_y, model_adapt, adapt_grad=True):
            n_count = 0

            for i in reversed(range(num_step)):

                t = torch.tensor([i]).to(x.device)

                x_start = self.output(
                    cur_y, x_feat, x_pred_mark, x_feat_mark, t)
                model_mean, _, posterior_log_variance = \
                    self.q_posterior(x_start=x_start, x_t=cur_y, t=t)
                noise = torch.randn_like(cur_y) if t > 0 else 0.
                sigma = (0.5 * posterior_log_variance).exp()
                pred_img = model_mean + sigma * noise

                if adapt_grad:
                    cur_y = pred_img.detach()

                n_count += 1
                if n_count == num_step:
                    cur_y = pred_img
                    break
                if n_count >= n_step:
                    if adapt_grad:
                        model_adapt = self.langevin_fn_train_fast_first(x=x, x_mark=x_mark, sample=pred_img, mean=model_mean, sigma=sigma, t=t,
                                                                        tgt_embs=target, partial_mask=partial_mask, model_initializer=model_adapt)
                    break
            return cur_y, model_adapt

        if self.pretrain_step == self.num_timesteps-num_step:

            cur_y, model_adapt = get_sample_first(
                1, y_pred_raw, model_adapt, adapt_grad=True)

            for k in range(2, num_step):
                img = img_noise
                for time, time_next in time_pairs_:

                    pred_noise, x_start, * \
                        _ = model_adapt.model_predictions_inf(
                            x, x_mark, img, time_cond)

                    if time_next < num_step:
                        img = x_start
                        continue

                    alpha = self.alphas_cumprod[time]
                    alpha_next = self.alphas_cumprod[time_next]
                    sigma = self.eta * \
                        ((1 - alpha / alpha_next) *
                         (1 - alpha_next) / (1 - alpha)).sqrt()
                    c = (1 - alpha_next - sigma ** 2).sqrt()
                    pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
                    noise = torch.randn_like(img)

                    img = pred_mean + sigma * noise
                    img = img.detach()
                    cur_y = img

                cur_y, model_adapt = get_sample(
                    k, cur_y, model_adapt, adapt_grad=True)

            img = img_noise
            for time, time_next in time_pairs_:
                pred_noise, x_start, * \
                    _ = model_adapt.model_predictions_inf(
                        x, x_mark, img, time_cond)

                if time_next < num_step:
                    img = x_start
                    continue

                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]
                sigma = self.eta * ((1 - alpha / alpha_next)
                                    * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()
                pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
                noise = torch.randn_like(img)

                img = pred_mean + sigma * noise

                y_pred_raw = img.detach()

            cur_y_new, model_adapt = get_sample(
                num_step, y_pred_raw, model_adapt, adapt_grad=False)

        else:
            cur_y, model_adapt = get_sample_fast_first(
                1, y_pred_raw, model_adapt, adapt_grad=True)

            for k in range(2, num_step+1):

                img = img_noise
                for time, time_next in time_pairs_:

                    pred_noise, x_start, * \
                        _ = model_adapt.model_predictions_inf(
                            x, x_mark, img, time_cond)

                    if time_next < num_step:
                        img = x_start
                        continue

                    alpha = self.alphas_cumprod[time]
                    alpha_next = self.alphas_cumprod[time_next]
                    sigma = self.eta * \
                        ((1 - alpha / alpha_next) *
                         (1 - alpha_next) / (1 - alpha)).sqrt()
                    c = (1 - alpha_next - sigma ** 2).sqrt()
                    pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
                    noise = torch.randn_like(img)

                    img = pred_mean + sigma * noise

                    cur_y = img.detach()

                cur_y, model_adapt = get_sample_fast(
                    k, cur_y, model_adapt, adapt_grad=True)
                # 最后一次adapt

            img = img_noise
            for time, time_next in time_pairs_:
                pred_noise, x_start, * \
                    _ = model_adapt.model_predictions_inf(
                        x, x_mark, img, time_cond)

                if time_next < num_step:
                    img = x_start
                    continue

                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]
                sigma = self.eta * ((1 - alpha / alpha_next)
                                    * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()
                pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
                noise = torch.randn_like(img)

                img = pred_mean + sigma * noise

                y_pred_raw = img.detach()

            cur_y_new, model_adapt = get_sample_fast(
                num_step+1, y_pred_raw, model_adapt, adapt_grad=False)

        return cur_y_new, model_adapt

    def p_sample_infill_train_loop_accelerate_short_af(self, maml, num_step, x, x_mark, time_cond, adapt_grad=True, af_pred=None, **kwargs):

        model_adapt = maml.clone()
        x_feat = x[:, :self.feat_len, :]
        x_pred_mark = x_mark[:, self.feat_len:, :]
        x_feat_mark = x_mark[:, :self.feat_len, :]

        target = x[:, self.feat_len//2:, :]
        self.target = target

        y_pred_raw = af_pred.to(x.device).detach()

        partial_mask_2 = torch.zeros(
            y_pred_raw[:, self.feat_len//2:, :].size(), dtype=torch.bool)
        partial_mask_1 = torch.ones(
            y_pred_raw[:, :self.feat_len//2, :].size(), dtype=torch.bool)
        partial_mask = torch.cat([partial_mask_1, partial_mask_2], dim=1)

        def get_sample_fast(n_step, cur_y, model_adapt, adapt_grad=True):

            n_count = 0
            times = torch.linspace(-1, self.num_timesteps -
                                   self.pretrain_step - 1, steps=num_step + 1)  # [201]
            times = list(reversed(times.int().tolist()))
            # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
            time_pairs = list(zip(times[:-1], times[1:]))
            for i in reversed(range(num_step)):

                t = time_pairs[n_count][0]
                t = torch.tensor([t]).to(x.device)

                x_start = model_adapt.output(
                    cur_y, x_feat, x_pred_mark, x_feat_mark, t)
                if time_pairs[n_count][1] < 0:
                    cur_y = x_start
                    continue
                maybe_clip = partial(torch.clamp, min=-1., max=1.)
                x_start = maybe_clip(x_start)
                pred_noise = self.predict_noise_from_start(cur_y, t, x_start)

                alpha = self.alphas_cumprod[time_pairs[n_count][0]]
                alpha_next = self.alphas_cumprod[time_pairs[n_count][1]]
                sigma = self.eta * ((1 - alpha / alpha_next)
                                    * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()
                pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
                noise = torch.randn_like(cur_y)
                pred_img = pred_mean + sigma * noise

                if adapt_grad:
                    cur_y = pred_img.detach()

                n_count += 1
                if n_count >= n_step:
                    if adapt_grad:
                        model_adapt = self.langevin_fn_train_fast(x=x, x_mark=x_mark, sample=pred_img, mean=pred_mean, sigma=sigma, t=t,
                                                                  tgt_embs=target, partial_mask=partial_mask, model_initializer=model_adapt)
                # else:
                #     print('haha', n_count, i)

                    break
            return cur_y, model_adapt

        def get_sample(n_step, cur_y, model_adapt, adapt_grad=True):
            n_count = 0
            for i in reversed(range(num_step)):

                t = torch.tensor([i]).to(x.device)

                x_start = model_adapt.output(
                    cur_y, x_feat, x_pred_mark, x_feat_mark, t)
                model_mean, _, posterior_log_variance = \
                    self.q_posterior(x_start=x_start, x_t=cur_y, t=t)
                noise = torch.randn_like(cur_y) if t > 0 else 0.
                sigma = (0.5 * posterior_log_variance).exp()
                pred_img = model_mean + sigma * noise

                if adapt_grad:
                    cur_y = pred_img.detach()

                n_count += 1
                if n_count == num_step:
                    cur_y = pred_img
                    break
                if n_count >= n_step:
                    if adapt_grad:
                        model_adapt = self.langevin_fn_train_fast(x=x, x_mark=x_mark, sample=pred_img, mean=model_mean, sigma=sigma, t=t,
                                                                  tgt_embs=target, partial_mask=partial_mask, model_initializer=model_adapt)
                    break
            return cur_y, model_adapt

        def get_sample_fast_first(n_step, cur_y, model_adapt, adapt_grad=True):

            n_count = 0
            times = torch.linspace(-1, self.num_timesteps -
                                   self.pretrain_step - 1, steps=num_step + 1)  # [201]
            times = list(reversed(times.int().tolist()))
            # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
            time_pairs = list(zip(times[:-1], times[1:]))
            for i in reversed(range(num_step)):

                t = time_pairs[n_count][0]
                t = torch.tensor([t]).to(x.device)

                x_start = self.output(
                    cur_y, x_feat, x_pred_mark, x_feat_mark, t)
                if time_pairs[n_count][1] < 0:
                    cur_y = x_start
                    continue
                maybe_clip = partial(torch.clamp, min=-1., max=1.)
                x_start = maybe_clip(x_start)
                pred_noise = self.predict_noise_from_start(cur_y, t, x_start)

                alpha = self.alphas_cumprod[time_pairs[n_count][0]]
                alpha_next = self.alphas_cumprod[time_pairs[n_count][1]]
                sigma = self.eta * ((1 - alpha / alpha_next)
                                    * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()
                pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
                noise = torch.randn_like(cur_y)
                pred_img = pred_mean + sigma * noise

                if adapt_grad:
                    cur_y = pred_img.detach()

                n_count += 1
                if n_count >= n_step:
                    if adapt_grad:
                        model_adapt = self.langevin_fn_train_fast_first(x=x, x_mark=x_mark, sample=pred_img, mean=pred_mean, sigma=sigma, t=t,
                                                                        tgt_embs=target, partial_mask=partial_mask, model_initializer=model_adapt)
                # else:
                #     print('haha', n_count, i)

                    break
            return cur_y, model_adapt

        def get_sample_first(n_step, cur_y, model_adapt, adapt_grad=True):
            n_count = 0

            for i in reversed(range(num_step)):

                t = torch.tensor([i]).to(x.device)

                x_start = self.output(
                    cur_y, x_feat, x_pred_mark, x_feat_mark, t)
                model_mean, _, posterior_log_variance = \
                    self.q_posterior(x_start=x_start, x_t=cur_y, t=t)
                noise = torch.randn_like(cur_y) if t > 0 else 0.
                sigma = (0.5 * posterior_log_variance).exp()
                pred_img = model_mean + sigma * noise

                if adapt_grad:
                    cur_y = pred_img.detach()

                n_count += 1
                if n_count >= n_step:
                    if adapt_grad:
                        model_adapt = self.langevin_fn_train_fast_first(x=x, x_mark=x_mark, sample=pred_img, mean=model_mean, sigma=sigma, t=t,
                                                                        tgt_embs=target, partial_mask=partial_mask, model_initializer=model_adapt)
                    break
            return cur_y, model_adapt

        if self.pretrain_step == self.num_timesteps-num_step:

            cur_y, model_adapt = get_sample_first(
                1, y_pred_raw, model_adapt, adapt_grad=True)

            for k in range(2, num_step):

                cur_y, model_adapt = get_sample(
                    k, y_pred_raw, model_adapt, adapt_grad=True)

            cur_y_new, model_adapt = get_sample(
                num_step, y_pred_raw, model_adapt, adapt_grad=False)

        else:
            cur_y, model_adapt = get_sample_fast_first(
                1, y_pred_raw, model_adapt, adapt_grad=True)

            for k in range(2, num_step+1):

                cur_y, model_adapt = get_sample_fast(
                    k, y_pred_raw, model_adapt, adapt_grad=True)

            cur_y_new, model_adapt = get_sample_fast(
                num_step+1, y_pred_raw, model_adapt, adapt_grad=False)

        return cur_y_new, model_adapt

    def p_sample_infill_train_loop_accelerate_short_af_incontext(self, maml, num_step, x, x_mark, x_incontext, x_incontext_mark, time_cond, adapt_grad=True, af_pred=None, **kwargs):

        model_adapt = maml.clone()
        x_feat = x[:, :self.feat_len, :]
        x_pred_mark = x_mark[:, self.feat_len:, :]
        x_feat_mark = x_mark[:, :self.feat_len, :]

        target = x[:, self.feat_len//2:, :]
        self.target = target

        y_pred_raw = af_pred.to(x.device).detach()

        partial_mask_2 = torch.zeros(
            y_pred_raw[:, self.feat_len//2:, :].size(), dtype=torch.bool)
        partial_mask_1 = torch.ones(
            y_pred_raw[:, :self.feat_len//2, :].size(), dtype=torch.bool)
        partial_mask = torch.cat([partial_mask_1, partial_mask_2], dim=1)

        def get_sample_fast(n_step, cur_y, x_incontext_, x_incontext_mark_, model_adapt, adapt_grad=True):

            n_count = 0
            times = torch.linspace(-1, self.num_timesteps -
                                   self.pretrain_step - 1, steps=num_step + 1)  # [201]
            times = list(reversed(times.int().tolist()))
            # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
            time_pairs = list(zip(times[:-1], times[1:]))
            for i in reversed(range(num_step)):

                t = time_pairs[n_count][0]
                t = torch.tensor([t]).to(x.device)

                x_start = model_adapt.output(
                    cur_y, x_feat, x_pred_mark, x_feat_mark, t)
                if time_pairs[n_count][1] < 0:
                    cur_y = x_start
                    continue
                maybe_clip = partial(torch.clamp, min=-1., max=1.)
                x_start = maybe_clip(x_start)
                pred_noise = self.predict_noise_from_start(cur_y, t, x_start)

                alpha = self.alphas_cumprod[time_pairs[n_count][0]]
                alpha_next = self.alphas_cumprod[time_pairs[n_count][1]]
                sigma = self.eta * ((1 - alpha / alpha_next)
                                    * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()
                pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
                noise = torch.randn_like(cur_y)
                pred_img = pred_mean + sigma * noise

                if adapt_grad:
                    cur_y = pred_img.detach()

                n_count += 1
                if n_count >= n_step:
                    if adapt_grad:
                        model_adapt = self.langevin_fn_train_fast(x=x, x_mark=x_mark, sample=pred_img, mean=pred_mean, sigma=sigma, t=t,
                                                                  tgt_embs=target, partial_mask=partial_mask, model_initializer=model_adapt)
                # else:
                #     print('haha', n_count, i)

                    break
            return cur_y, model_adapt

        def get_sample(n_step, cur_y, x_incontext_, x_incontext_mark_, model_adapt, adapt_grad=True):
            n_count = 0
            for i in reversed(range(num_step)):

                t = torch.tensor([i]).to(x.device)

                x_start = model_adapt.output(
                    cur_y, x_feat, x_pred_mark, x_feat_mark, t)
                model_mean, _, posterior_log_variance = \
                    self.q_posterior(x_start=x_start, x_t=cur_y, t=t)
                noise = torch.randn_like(cur_y) if t > 0 else 0.
                sigma = (0.5 * posterior_log_variance).exp()
                pred_img = model_mean + sigma * noise

                if adapt_grad:
                    cur_y = pred_img.detach()

                n_count += 1
                if n_count == num_step:
                    cur_y = pred_img
                    break
                if n_count >= n_step:
                    if adapt_grad:
                        model_adapt = self.langevin_fn_train_fast(x=x, x_mark=x_mark, sample=pred_img, mean=model_mean, sigma=sigma, t=t,
                                                                  tgt_embs=target, partial_mask=partial_mask, model_initializer=model_adapt)
                    break
            return cur_y, model_adapt

        def get_sample_fast_first(n_step, cur_y, x_incontext_, x_incontext_mark_, model_adapt, adapt_grad=True):

            n_count = 0
            times = torch.linspace(-1, self.num_timesteps -
                                   self.pretrain_step - 1, steps=num_step + 1)  # [201]
            times = list(reversed(times.int().tolist()))
            # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
            time_pairs = list(zip(times[:-1], times[1:]))
            for i in reversed(range(num_step)):

                t = time_pairs[n_count][0]
                t = torch.tensor([t]).to(x.device)

                x_start = self.output(
                    cur_y, x_feat, x_pred_mark, x_feat_mark, t)
                if time_pairs[n_count][1] < 0:
                    cur_y = x_start
                    continue
                maybe_clip = partial(torch.clamp, min=-1., max=1.)
                x_start = maybe_clip(x_start)
                pred_noise = self.predict_noise_from_start(cur_y, t, x_start)

                alpha = self.alphas_cumprod[time_pairs[n_count][0]]
                alpha_next = self.alphas_cumprod[time_pairs[n_count][1]]
                sigma = self.eta * ((1 - alpha / alpha_next)
                                    * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()
                pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
                noise = torch.randn_like(cur_y)
                pred_img = pred_mean + sigma * noise

                if adapt_grad:
                    cur_y = pred_img.detach()

                n_count += 1
                if n_count >= n_step:
                    if adapt_grad:
                        model_adapt = self.langevin_fn_train_fast_first(x=x, x_mark=x_mark, sample=pred_img, mean=pred_mean, sigma=sigma, t=t,
                                                                        tgt_embs=target, partial_mask=partial_mask, model_initializer=model_adapt)
                # else:
                #     print('haha', n_count, i)

                    break
            return cur_y, model_adapt

        def get_sample_first(n_step, cur_y, x_incontext_, x_incontext_mark_, model_adapt, adapt_grad=True):
            n_count = 0

            for i in reversed(range(num_step)):

                t = torch.tensor([i]).to(x.device)

                x_start = self.output(
                    cur_y, x_feat, x_pred_mark, x_feat_mark, t)
                model_mean, _, posterior_log_variance = \
                    self.q_posterior(x_start=x_start, x_t=cur_y, t=t)
                noise = torch.randn_like(cur_y) if t > 0 else 0.
                sigma = (0.5 * posterior_log_variance).exp()
                pred_img = model_mean + sigma * noise

                if adapt_grad:
                    cur_y = pred_img.detach()

                n_count += 1
                if n_count >= n_step:
                    if adapt_grad:
                        model_adapt = self.langevin_fn_train_fast_first(x=x, x_mark=x_mark, sample=pred_img, mean=model_mean, sigma=sigma, t=t,
                                                                        tgt_embs=target, partial_mask=partial_mask, model_initializer=model_adapt)
                    break
            return cur_y, model_adapt

        if self.pretrain_step == self.num_timesteps-num_step:
            ix = 0

            cur_y, model_adapt = get_sample_first(1, y_pred_raw, x_incontext[:, ix, :, :],
                                                  x_incontext_mark[:, ix, :, :], model_adapt, adapt_grad=True)

            for k in range(2, num_step):
                ix += 1

                cur_y, model_adapt = get_sample(k, y_pred_raw, x_incontext[:, ix, :, :],
                                                x_incontext_mark[:, ix, :, :], model_adapt, adapt_grad=True)

            cur_y_new, model_adapt = get_sample(num_step, y_pred_raw, x_incontext[:, ix+1, :, :],
                                                x_incontext_mark[:, ix+1, :, :], model_adapt, adapt_grad=False)

        else:
            cur_y, model_adapt = get_sample_fast_first(
                1, y_pred_raw, model_adapt, adapt_grad=True)

            for k in range(2, num_step+1):

                cur_y, model_adapt = get_sample_fast(
                    k, y_pred_raw, model_adapt, adapt_grad=True)

            cur_y_new, model_adapt = get_sample_fast(
                num_step+1, y_pred_raw, model_adapt, adapt_grad=False)

        return cur_y_new, model_adapt

    def langevin_fn_train_fast(self,  x, x_mark, sample, mean, sigma, t, tgt_embs, partial_mask, model_initializer):
        coef = 2.0e-2
        partial_mask = partial_mask[:, :, -1]
        # tgt_embs = tgt_embs[:,self.feat_len:,:]

        def get_loss(input_embs_param, x_start):
            if sigma.mean() == 0:
                logp_term = coef * ((mean - input_embs_param) ** 2 / 1.).mean()
                infill_loss = (
                    x_start[:, :, :][partial_mask] - tgt_embs[:, :, :][partial_mask]) ** 2
                infill_loss = infill_loss.mean()
            else:
                logp_term = coef * \
                    ((mean - input_embs_param) ** 2 / sigma).mean()
                infill_loss0 = (
                    x_start[:, :, :][partial_mask] - tgt_embs[:, :, :][partial_mask]) ** 2
                # print(x_start.size(),tgt_embs.size(),partial_mask.size(),'==')
                infill_loss = (infill_loss0 / sigma.mean()).mean()

            # infill_loss *= 1e-3
            # print(infill_loss)
            loss = logp_term + infill_loss
            # loss = infill_loss
            return loss

        x_feat = x[:, :self.feat_len, :]
        x_pred_mark = x_mark[:, self.feat_len:, :]
        x_feat_mark = x_mark[:, :self.feat_len, :]
        input_embs_param = sample.detach()
        x_start = model_initializer.output(
            input_embs_param, x_feat, x_pred_mark, x_feat_mark, t)

        loss = get_loss(input_embs_param, x_start)

        model_initializer.adapt(loss)

        return model_initializer

    def langevin_fn_train_fast_incontext(self,  x, x_mark, sample, mean, sigma, t, tgt_embs, partial_mask, model_initializer):
        coef = 2.0e-2
        partial_mask = partial_mask[:, :, -1]
        # tgt_embs = tgt_embs[:,self.feat_len:,:]

        def get_loss(input_embs_param, x_start):
            if sigma.mean() == 0:
                logp_term = coef * ((mean - input_embs_param) ** 2 / 1.).mean()
                infill_loss = (x_start - tgt_embs) ** 2
                infill_loss = infill_loss.mean()
            else:
                logp_term = coef * \
                    ((mean - input_embs_param) ** 2 / sigma).mean()
                infill_loss0 = (x_start - tgt_embs) ** 2
                # print(x_start.size(),tgt_embs.size(),partial_mask.size(),'==')
                infill_loss = (infill_loss0 / sigma.mean()).mean()

            # infill_loss *= 1e-3
            # print(infill_loss)
            loss = logp_term + infill_loss
            # loss = infill_loss
            return loss

        x_feat = x[:, :self.feat_len, :]
        x_pred_mark = x_mark[:, self.feat_len:, :]
        x_feat_mark = x_mark[:, :self.feat_len, :]
        input_embs_param = sample.detach()
        x_start = model_initializer.output(
            input_embs_param, x_feat, x_pred_mark, x_feat_mark, t)

        loss = get_loss(input_embs_param, x_start)

        model_initializer.adapt(loss)

        return model_initializer

    def langevin_fn_train_fast_first(self,  x, x_mark, sample, mean, sigma, t, tgt_embs, partial_mask, model_initializer):
        coef = 2.0e-2
        partial_mask = partial_mask[:, :, -1]
        # tgt_embs = tgt_embs[:,self.feat_len:,:]

        def get_loss(input_embs_param, x_start):
            if sigma.mean() == 0:
                logp_term = coef * ((mean - input_embs_param) ** 2 / 1.).mean()
                infill_loss = (
                    x_start[:, :, :][partial_mask] - tgt_embs[:, :, :][partial_mask]) ** 2
                infill_loss = infill_loss.mean()
            else:
                logp_term = coef * \
                    ((mean - input_embs_param) ** 2 / sigma).mean()
                infill_loss0 = (
                    x_start[:, :, :][partial_mask] - tgt_embs[:, :, :][partial_mask]) ** 2
                # print(x_start.size(),tgt_embs.size(),partial_mask.size(),'==')
                infill_loss = (infill_loss0 / sigma.mean()).mean()

            # infill_loss *= 1e-3
            # print(infill_loss)
            loss = logp_term + infill_loss
            # loss =  infill_loss
            return loss

        x_feat = x[:, :self.feat_len, :]
        x_pred_mark = x_mark[:, self.feat_len:, :]
        x_feat_mark = x_mark[:, :self.feat_len, :]
        input_embs_param = sample.detach()
        x_start = self.output(input_embs_param, x_feat,
                              x_pred_mark, x_feat_mark, t)
        # y_pred,target = self.meta_pred(x,x_mark, t=t) # 预测x_start

        loss = get_loss(input_embs_param, x_start)

        model_initializer.adapt(loss)

        return model_initializer

    def langevin_fn_train_fast_his(self,  x, x_mark, sample, mean, sigma, t, tgt_embs, partial_mask, model_initializer):
        coef = 2.0e-2
        partial_mask = partial_mask[:, :, -1]
        # tgt_embs = tgt_embs[:,self.feat_len:,:]

        def get_loss(input_embs_param, x_start):
            if sigma.mean() == 0:
                logp_term = coef * ((mean - input_embs_param) ** 2 / 1.).mean()
                infill_loss = (x_start[:, :self.feat_len//2, :] -
                               tgt_embs[:, self.feat_len//2:self.feat_len, :]) ** 2
                infill_loss = infill_loss.mean()
            else:
                logp_term = coef * \
                    ((mean - input_embs_param) ** 2 / sigma).mean()
                infill_loss0 = (x_start[:, :self.feat_len//2, :] -
                                tgt_embs[:, self.feat_len//2:self.feat_len, :]) ** 2
                # print(x_start.size(),tgt_embs.size(),partial_mask.size(),'==')
                infill_loss = (infill_loss0 / sigma.mean()).mean()

            # infill_loss *= 1e-3
            # print(infill_loss)
            loss = logp_term + infill_loss
            # loss =  infill_loss
            return loss

        x_feat = x[:, :self.feat_len, :]
        x_pred_mark = x_mark[:, self.feat_len:, :]
        x_feat_mark = x_mark[:, :self.feat_len, :]
        input_embs_param = sample.detach()
        x_start = self.output(input_embs_param, x_feat,
                              x_pred_mark, x_feat_mark, t)
        # y_pred,target = self.meta_pred(x,x_mark, t=t) # 预测x_start

        loss = get_loss(input_embs_param, x_start)

        model_initializer.adapt(loss)

        return model_initializer

    def _train_loss(self, x_start, x_mark, t, target=None, noise=None, padding_masks=None):
        # print(x_start.size(),"1111")
        x_pred = x_start[:, self.feat_len//2:, :]  # -1
        x_feat = x_start[:, :self.feat_len, :]

        x_pred_mark = x_mark[:, self.feat_len//2:, :]
        x_feat_mark = x_mark[:, :self.feat_len, :]

        noise = default(noise, lambda: torch.randn_like(x_pred))
        # if target is None:
        #     # target = x_pred
        target = x_pred
        target = x_start[:, self.feat_len//2:, :]

        x = self.q_sample(x_start=x_pred, t=t, noise=noise)  # noise sample 加噪

        model_out = self.output(x, x_feat, x_pred_mark,
                                x_feat_mark, t, padding_masks)  # 用transformer去噪

        model_out_ = model_out
        model_out = model_out[:, self.feat_len//2:, :]
        target = target[:, self.feat_len//2:, :]
        # print(model_out.shape, target.shape)
        # exit()
        train_loss = self.loss_fn(model_out, target, reduction='none')

        fourier_loss = torch.tensor([0.])
        if self.use_ff:
            fft1 = torch.fft.fft(model_out.transpose(1, 2), norm='forward')
            fft2 = torch.fft.fft(target.transpose(1, 2), norm='forward')
            fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
            fourier_loss = self.loss_fn(torch.real(fft1), torch.real(fft2), reduction='none')\
                + self.loss_fn(torch.imag(fft1),
                               torch.imag(fft2), reduction='none')
            train_loss += self.ff_weight * fourier_loss

        train_loss = reduce(train_loss, 'b ... -> b (...)', 'mean')
        train_loss = train_loss * \
            extract(self.loss_weight, t, train_loss.shape)

        return train_loss.mean()

    def meta_pred(self, x_start, x_mark, t, target=None, noise=None, padding_masks=None):
        x_pred = x_start[:, self.feat_len//2:, :]  # -1
        x_feat = x_start[:, :self.feat_len, :]

        x_pred_mark = x_mark[:, self.feat_len//2:, :]
        x_feat_mark = x_mark[:, :self.feat_len, :]

        noise = default(noise, lambda: torch.randn_like(x_pred))
        # if target is None:
        #     # target = x_pred
        self.target = x_pred
        target = self.target
        t_org = t + self.num_timesteps-self.pretrain_step - 1

        x = self.q_sample(x_start=x_pred, t=t_org, noise=noise)

        model_out = self.output(x, x_feat, x_pred_mark,
                                x_feat_mark, t, padding_masks)

        return model_out, target

    def forward(self, x, x_mark, **kwargs):

        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        assert n == feature_size, f'number of variable must be {feature_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self._train_loss(x_start=x, x_mark=x_mark, t=t, **kwargs)

    def forward_meta(self, x, x_mark, maml, af_pred=None, **kwargs):
        num_step = 20
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        assert n == feature_size, f'number of variable must be {feature_size}'
        # t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        t_cond = torch.tensor([self.pretrain_step]*b).long().to(device)
        if af_pred is None:
            print("long")
            y_pred_final, model_adapt = self.p_sample_infill_train_loop_accelerate(
                maml, num_step, x, x_mark, t_cond, adapt_grad=True, af_pred=af_pred)
        else:
            print("short")
            y_pred_final, model_adapt = self.p_sample_infill_train_loop_accelerate_short_af(
                maml, num_step, x, x_mark, t_cond, adapt_grad=True, af_pred=af_pred)

        # train_loss = self.loss_fn(y_pred_final, self.target, reduction='none')

        y_pred_final = y_pred_final[:, self.feat_len//2:, :]
        target = self.target[:, self.feat_len//2:, :]
        train_loss = (y_pred_final-target)**2
        fourier_loss = torch.tensor([0.])
        if self.use_ff:
            fft1 = torch.fft.fft(y_pred_final.transpose(1, 2), norm='forward')
            fft2 = torch.fft.fft(target.transpose(1, 2), norm='forward')
            fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
            fourier_loss = self.loss_fn(torch.real(fft1), torch.real(fft2), reduction='none')\
                + self.loss_fn(torch.imag(fft1),
                               torch.imag(fft2), reduction='none')

            train_loss += self.ff_weight * fourier_loss

        train_loss = reduce(train_loss, 'b ... -> b (...)', 'mean')
        # train_loss = train_loss * extract(self.loss_weight, t_cond, train_loss.shape)

        return train_loss.mean()

    def forward_meta_incontext(self, x, x_mark, x_incontext, x_incontext_mark, maml, af_pred=None, **kwargs):
        num_step = 20
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        assert n == feature_size, f'number of variable must be {feature_size}'
        # t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        t_cond = torch.tensor([self.pretrain_step]*b).long().to(device)
        if af_pred is None:
            print("long")
            y_pred_final, model_adapt = self.p_sample_infill_train_loop_accelerate(
                maml, num_step, x, x_mark, x_incontext, x_incontext_mark, t_cond, adapt_grad=True, af_pred=af_pred)
        else:
            print("short")
            y_pred_final, model_adapt = self.p_sample_infill_train_loop_accelerate_short_af_incontext(
                maml, num_step, x, x_mark, x_incontext, x_incontext_mark, t_cond, adapt_grad=True, af_pred=af_pred)

        # train_loss = self.loss_fn(y_pred_final, self.target, reduction='none')

        y_pred_final = y_pred_final[:, self.feat_len//2:, :]
        target = self.target[:, self.feat_len//2:, :]
        train_loss = (y_pred_final-target)**2
        fourier_loss = torch.tensor([0.])
        if self.use_ff:
            fft1 = torch.fft.fft(y_pred_final.transpose(1, 2), norm='forward')
            fft2 = torch.fft.fft(target.transpose(1, 2), norm='forward')
            fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
            fourier_loss = self.loss_fn(torch.real(fft1), torch.real(fft2), reduction='none')\
                + self.loss_fn(torch.imag(fft1),
                               torch.imag(fft2), reduction='none')

            train_loss += self.ff_weight * fourier_loss

        train_loss = reduce(train_loss, 'b ... -> b (...)', 'mean')
        # train_loss = train_loss * extract(self.loss_weight, t_cond, train_loss.shape)

        return train_loss.mean()

    def return_components(self, x, t: int):
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        assert n == feature_size, f'number of variable must be {feature_size}'
        t = torch.tensor([t])
        t = t.repeat(b).to(device)
        x = self.q_sample(x, t)
        trend, season, residual = self.model(x, t, return_res=True)
        return trend, season, residual, x

    def fast_sample_infill(self, x, x_mark, shape, target, sampling_timesteps, partial_mask=None, clip_denoised=True, model_kwargs=None):
        batch, device, total_timesteps, eta = shape[0], self.betas.device, self.num_timesteps, self.eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)  # [201]

        times = list(reversed(times.int().tolist()))
        # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        time_pairs = list(zip(times[:-1], times[1:]))

        shape = torch.Size(
            [shape[0], self.seq_length+self.feat_len//2, shape[2]])
        img = torch.randn(shape, device=device)
        partial_mask = partial_mask[:, self.feat_len//2:, :]
        target = target[:, self.feat_len//2:, :]

        for time, time_next in tqdm(time_pairs, desc='conditional sampling loop time step'):

            time_cond = torch.full(
                (batch,), time, device=device, dtype=torch.long)

            pred_noise, x_start, * \
                _ = self.model_predictions(
                    x, x_mark, img, time_cond, clip_x_start=clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) *
                           (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
            noise = torch.randn_like(img)

            img = pred_mean + sigma * noise

            # print(img,pred_mean,sigma,time_cond,target,partial_mask,x,x_mark)

            # exit()
            img = self.langevin_fn(sample=img, mean=pred_mean, sigma=sigma, t=time_cond,
                                   tgt_embs=target, partial_mask=partial_mask, all=x, all_mark=x_mark,
                                   **model_kwargs)

            target_t = self.q_sample(target, t=time_cond)

            img[partial_mask] = target_t[partial_mask]

        img[partial_mask] = target[partial_mask]
        img = img[:, -self.seq_length:, :]

        return img

    def fast_sample_infill_adapt(self, x, x_mark, shape, target, sampling_timesteps, maml, partial_mask=None, clip_denoised=True, model_kwargs=None):
        batch, device, total_timesteps, eta = shape[0], self.betas.device, self.num_timesteps, self.eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps-self.pretrain_step -
                               1, steps=sampling_timesteps + 1)  # [201]

        times = list(reversed(times.int().tolist()))
        # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        time_pairs = list(zip(times[:-1], times[1:]))

        shape = torch.Size(
            [shape[0], self.seq_length+self.feat_len//2, shape[2]])
        img = torch.randn(shape, device=device)
        partial_mask = partial_mask[:, self.feat_len//2:, :]
        target = target[:, self.feat_len//2:, :]

        for time, time_next in tqdm(time_pairs, desc='conditional sampling loop time step'):

            time_cond = torch.full(
                (batch,), time, device=device, dtype=torch.long)

            pred_noise, x_start, * \
                _ = self.model_predictions(
                    x, x_mark, img, time_cond, clip_x_start=clip_denoised)

            if time_next < total_timesteps-self.pretrain_step:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) *
                           (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
            noise = torch.randn_like(img)

            img = pred_mean + sigma * noise

            img = self.langevin_fn(sample=img, mean=pred_mean, sigma=sigma, t=time_cond,
                                   tgt_embs=target, partial_mask=partial_mask, all=x, all_mark=x_mark,
                                   **model_kwargs)

            target_t = self.q_sample(target, t=time_cond)

            img[partial_mask] = target_t[partial_mask]

        # ----------
        num_step = 20
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        assert n == feature_size, f'number of variable must be {feature_size}'
        # t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        self.pretrain_step = 480
        t = self.num_timesteps
        t_next = self.num_timesteps - self.pretrain_step
        t_cond = torch.tensor([self.num_timesteps]*b).long().to(device)

        y_pred_final, model_adapt = self.p_sample_infill_train_loop_accelerate(
            maml, num_step, x, x_mark, t_cond, t, t_next, adapt_grad=True)

        y_pred_final[partial_mask] = target[partial_mask]
        img = y_pred_final[:, -self.seq_length:, :]

        return img

    def fast_sample_pre(self, x, x_mark, shape, target, sampling_timesteps, partial_mask=None,
                        clip_denoised=True, model_kwargs=None):
        batch, device, total_timesteps, eta = shape[0], self.betas.device, self.num_timesteps, self.eta
        print(sampling_timesteps)
        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)  # [201]

        times = list(reversed(times.int().tolist()))
        # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        time_pairs = list(zip(times[:-1], times[1:]))

        shape = torch.Size(
            [shape[0], self.seq_length+self.feat_len//2, shape[2]])

        img = torch.randn(shape, device=device)

        for time, time_next in tqdm(time_pairs, desc='conditional sampling loop time step'):

            time_cond = torch.full(
                (batch,), time, device=device, dtype=torch.long)
            # print(time_cond.size()) --256

            pred_noise, x_start, * \
                _ = self.model_predictions(
                    x, x_mark, img, time_cond, clip_x_start=clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) *
                           (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(x_start)
            img = x_start * alpha_next.sqrt() + \
                c * pred_noise + \
                sigma * noise
        img = img[:, -self.seq_length:, :]

        return img

    def sample_infill(
        self, x, x_mark,
        shape,
        target,
        partial_mask=None,
        clip_denoised=True,
        model_kwargs=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        """
        batch, device = shape[0], self.betas.device
        shape = torch.Size(
            [shape[0], self.seq_length+self.feat_len//2, shape[2]])
        img = torch.randn(shape, device=device)

        partial_mask = partial_mask[:, self.feat_len//2:, :]
        target = target[:, self.feat_len//2:, :]

        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='conditional sampling loop time step', total=self.num_timesteps):
            img = self.p_sample_infill(all=x, all_mark=x_mark, x=img, t=t, clip_denoised=clip_denoised, target=target,
                                       partial_mask=partial_mask, model_kwargs=model_kwargs)

        img[partial_mask] = target[partial_mask]
        img = img[:, -self.seq_length:, :]
        return img

    def sample_infill_adapt(
        self, x, x_mark,
        shape,
        target,
        maml,
        af_pred=None,
        partial_mask=None,
        clip_denoised=True,
        model_kwargs=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        """
        num_step = 20
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        assert n == feature_size, f'number of variable must be {feature_size}'
        # t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        x = x.to(torch.float32)
        x_mark = x_mark.to(torch.float32)
        t_cond = torch.tensor([self.pretrain_step]*b).long().to(device)
        if af_pred is None:
            y_pred_final = self.p_sample_infill_test_loop_accelerate(
                maml, num_step, x, x_mark, t_cond, adapt_grad=True, af_pred=af_pred)
        else:
            y_pred_final = self.p_sample_infill_test_loop_accelerate_short_af(
                maml, num_step, x, x_mark, t_cond, adapt_grad=True, af_pred=af_pred)

        y_pred_final = y_pred_final[:, -self.seq_length:, :]
        return y_pred_final

    def sample_infill_adapt_incontext(
        self, x, x_mark, x_incontext, x_in_mark,
        shape,
        target,
        maml,
        af_pred=None,
        af_incontext=None,
        adapt_h=False,
        adapt_f=False,
        adapt_f_num=0,
        partial_mask=None,
        clip_denoised=True,
        model_kwargs=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        """
        num_step = 20
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        assert n == feature_size, f'number of variable must be {feature_size}'
        # t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        x = x.to(torch.float32)
        x_mark = x_mark.to(torch.float32)
        x_incontext = x_incontext.to(torch.float32)
        x_in_mark = x_in_mark.to(torch.float32)
        t_cond = torch.tensor([self.pretrain_step]*b).long().to(device)
        if af_pred is None:

            y_pred_final = self.p_sample_infill_test_loop_accelerate(
                maml, num_step, x, x_mark, t_cond, adapt_grad=True, af_pred=af_pred)
        else:
            # if adapt_f is False and adapt_h is False:
            y_pred_final = self.p_sample_infill_test_loop_accelerate_short_af_incontext(maml, num_step, x, x_mark, x_incontext, x_in_mark, t_cond, adapt_grad=True,
                                                                                        af_pred=af_pred, af_incontext=af_incontext, adapt_h=adapt_h, adapt_f=adapt_f, adapt_f_num=adapt_f_num)

        # y_pred_final = y_pred_final[:,-self.seq_length:,:]
        return y_pred_final

    def sample_infill_no_adapt(
        self, x, x_mark,
        shape,
        target,
        partial_mask=None,
        clip_denoised=True,
        model_kwargs=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        """
        num_step = 20
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        assert n == feature_size, f'number of variable must be {feature_size}'
        # t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        x = x.to(torch.float32)
        x_mark = x_mark.to(torch.float32)
        t_cond = torch.tensor([self.pretrain_step]*b).long().to(device)

        partial_mask = partial_mask[:, self.feat_len//2:, :]
        target = target[:, self.feat_len//2:, :]
        y_pred_raw, target = self.meta_pred(x, x_mark, t=t_cond)
        img = y_pred_raw
        for t in tqdm(reversed(range(0, self.num_timesteps-self.pretrain_step)),
                      desc='conditional sampling loop time step', total=self.num_timesteps-self.pretrain_step):
            img = self.p_sample_infill(all=x, all_mark=x_mark, x=img, t=t, clip_denoised=clip_denoised, target=target,
                                       partial_mask=partial_mask, model_kwargs=model_kwargs)

        img[partial_mask] = target[partial_mask]
        img = img[:, -self.seq_length:, :]
        img = img.detach()
        return img

    def p_sample_infill(
        self, all, all_mark,
        x,
        target,
        t: int,
        partial_mask=None,
        clip_denoised=True,
        model_kwargs=None
    ):
        b, *_, device = *x.shape, self.betas.device
        batched_times = torch.full(
            (x.shape[0],), t, device=x.device, dtype=torch.long)

        model_mean, _, model_log_variance, _ = \
            self.p_mean_variance(all=all, all_mark=all_mark,
                                 x=x, t=batched_times, clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        sigma = (0.5 * model_log_variance).exp()
        pred_img = model_mean + sigma * noise

        pred_img = self.langevin_fn(sample=pred_img, mean=model_mean, sigma=sigma, t=batched_times,
                                    tgt_embs=target, partial_mask=partial_mask, all=all, all_mark=all_mark, **model_kwargs)

        target_t = self.q_sample(target, t=batched_times)
        pred_img[partial_mask] = target_t[partial_mask]
        pred_img = pred_img.detach()
        return pred_img

    def p_sample_infill_adapt(
        self, all, all_mark,
        x,
        target,
        maml,
        t: int,
        partial_mask=None,
        clip_denoised=True,
        model_kwargs=None
    ):

        b, *_, device = *x.shape, self.betas.device
        batched_times = torch.full(
            (x.shape[0],), t, device=x.device, dtype=torch.long)

        model_mean, _, model_log_variance, _ = \
            self.p_mean_variance(all=all, all_mark=all_mark,
                                 x=x, t=batched_times, clip_denoised=clip_denoised)

        noise = torch.randn_like(
            x) if t > self.num_timesteps-self.pretrain_step else 0.  # no noise if t == 0
        sigma = (0.5 * model_log_variance).exp()
        pred_img = model_mean + sigma * noise
        pred_img = pred_img.detach()
        pred_img = self.langevin_fn(sample=pred_img, mean=model_mean, sigma=sigma, t=batched_times,
                                    tgt_embs=target, partial_mask=partial_mask, all=all, all_mark=all_mark, **model_kwargs)

        target_t = self.q_sample(target, t=batched_times)
        pred_img[partial_mask] = target_t[partial_mask]

        return pred_img

    def langevin_fn(
        self,
        coef,
        partial_mask,
        tgt_embs,

        sample,
        mean,
        sigma,
        t,
        all,
        all_mark,

        learning_rate,
        coef_=0.,

    ):

        if t[0].item() < self.num_timesteps * 0.05:
            K = 0
        elif t[0].item() > self.num_timesteps * 0.9:
            K = 3
        elif t[0].item() > self.num_timesteps * 0.75:
            K = 2
            learning_rate = learning_rate * 0.5
        else:
            K = 1
            learning_rate = learning_rate * 0.25
        # print(tgt_embs)
        input_embs_param = torch.nn.Parameter(sample)
        partial_mask = partial_mask[:, :, -1]
        with torch.enable_grad():
            for i in range(K):

                optimizer = torch.optim.Adagrad(
                    [input_embs_param], lr=learning_rate)
                optimizer.zero_grad()

                #
                x_feat = all[:, :self.seq_length, :].to(torch.float32)
                x_feat_mark = all_mark[:, :self.seq_length, :].to(
                    torch.float32)
                x_pred_mark = all_mark[:, self.seq_length:, :].to(
                    torch.float32)
                # x_start = self.output(x=input_embs_param, t=t)
                x_start = self.output(
                    input_embs_param, x_feat, x_pred_mark, x_feat_mark, t)  # 预测 x_start

                if sigma.mean() == 0:
                    logp_term = coef * \
                        ((mean - input_embs_param) ** 2 / 1.).mean(dim=0).sum()
                    infill_loss = (
                        x_start[:, :, -1][partial_mask] - tgt_embs[:, :, -1][partial_mask]) ** 2
                    infill_loss = infill_loss.mean(dim=0).sum()
                else:
                    logp_term = coef * \
                        ((mean - input_embs_param)**2 / sigma).mean(dim=0).sum()
                    infill_loss = (
                        x_start[:, :, -1][partial_mask] - tgt_embs[:, :, -1][partial_mask]) ** 2
                    infill_loss = (infill_loss/sigma.mean()).mean(dim=0).sum()

                loss = logp_term + infill_loss

                loss.backward()
                optimizer.step()
                # grad = torch.autograd.grad(loss, input_embs_param, create_graph=True, retain_graph=True)

                # input_embs_param.data -= learning_rate*grad[0]
                # temp_net = update_module(temp_net, updates=tuple(-lr*g for g in grad))
                epsilon = torch.randn_like(input_embs_param.data)

                input_embs_param = torch.nn.Parameter(
                    (input_embs_param.data + coef_ * sigma.mean().item() * epsilon).detach())

        sample[~partial_mask] = input_embs_param.data[~partial_mask]
        # sample = input_embs_param.data
        return sample

    def p_sample_infill_test_loop_accelerate(self, maml, num_step, x, x_mark, time_cond, adapt_grad=True, af_pred=None, **kwargs):
        model_adapt = maml.clone()
        x = x.to(torch.float32)
        x_mark = x_mark.to(torch.float32)
        x_feat = x[:, :self.feat_len, :].to(torch.float32)
        x_pred_mark = x_mark[:, self.feat_len:, :].to(torch.float32)
        x_feat_mark = x_mark[:, :self.feat_len, :].to(torch.float32)

        target = x[:, self.feat_len//2:, :]
        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(self.num_timesteps-self.pretrain_step-1,
                               self.num_timesteps - 1, steps=self.sampling_timesteps + 1)  # [201]

        times = list(reversed(times.int().tolist()))
        # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        time_pairs_ = list(zip(times[:-1], times[1:]))

        img = torch.randn(x[:, self.feat_len//2:, :].shape,
                          device=x.device)  # torch.Size([48, 144, 21])
        img_noise = img

        for time, time_next in time_pairs_:

            pred_noise, x_start, * \
                _ = model_adapt.model_predictions_inf(
                    x, x_mark, img, time_cond)

            if time_next < num_step:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = self.eta * ((1 - alpha / alpha_next) *
                                (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
            noise = torch.randn_like(img)

            img = pred_mean + sigma * noise
            img = img.detach()

        y_pred_raw = img

        # y_pred_raw,target = model_adapt.meta_pred(x,x_mark, t=time_cond, **kwargs)

        partial_mask_2 = torch.zeros(
            y_pred_raw[:, self.feat_len//2:, :].size(), dtype=torch.bool)
        partial_mask_1 = torch.ones(
            y_pred_raw[:, :self.feat_len//2, :].size(), dtype=torch.bool)
        partial_mask = torch.cat([partial_mask_1, partial_mask_2], dim=1)

        def get_sample_fast(n_step, cur_y, model_adapt, adapt_grad=True):

            n_count = 0
            times = torch.linspace(-1, self.num_timesteps -
                                   self.pretrain_step - 1, steps=num_step + 1)  # [201]
            times = list(reversed(times.int().tolist()))
            # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
            time_pairs = list(zip(times[:-1], times[1:]))

            for i in reversed(range(num_step)):

                t = time_pairs[n_count][0]
                t = torch.tensor([t]).to(x.device)

                x_start = self.output(
                    cur_y, x_feat, x_pred_mark, x_feat_mark, t)
                if time_pairs[n_count][1] < 0:
                    cur_y = x_start
                    continue
                maybe_clip = partial(torch.clamp, min=-1., max=1.)
                x_start = maybe_clip(x_start)
                pred_noise = self.predict_noise_from_start(cur_y, t, x_start)

                alpha = self.alphas_cumprod[time_pairs[n_count][0]]
                alpha_next = self.alphas_cumprod[time_pairs[n_count][1]]
                sigma = self.eta * ((1 - alpha / alpha_next)
                                    * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()
                pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
                noise = torch.randn_like(cur_y)
                pred_img = pred_mean + sigma * noise

                if adapt_grad:

                    cur_y = pred_img.detach()
                # else:

                #     cur_y = pred_img.detach()

                n_count += 1
                if n_count >= n_step:
                    if adapt_grad:
                        model_adapt = self.langevin_fn_train_fast(x=x, x_mark=x_mark, sample=pred_img, mean=pred_mean, sigma=sigma, t=t,
                                                                  tgt_embs=target, partial_mask=partial_mask, model_initializer=model_adapt)
                # else:
                #     print('haha', n_count, i)

                    break

            return cur_y, model_adapt

        def get_sample(n_step, cur_y, model_adapt, adapt_grad=True):
            n_count = 0

            for i in reversed(range(num_step)):

                t = torch.tensor([i]).to(x.device)

                x_start = self.output(
                    cur_y, x_feat, x_pred_mark, x_feat_mark, t)

                model_mean, _, posterior_log_variance = \
                    self.q_posterior(x_start=x_start, x_t=cur_y, t=t)
                noise = torch.randn_like(cur_y) if t > 0 else 0.
                sigma = (0.5 * posterior_log_variance).exp()
                pred_img = model_mean + sigma * noise

                if adapt_grad:

                    cur_y = pred_img.detach()

                n_count += 1
                if n_count == num_step:
                    cur_y = pred_img
                    break
                if n_count >= n_step:
                    if adapt_grad:
                        model_adapt = self.langevin_fn_train_fast(x=x, x_mark=x_mark, sample=pred_img, mean=model_mean, sigma=sigma, t=t,
                                                                  tgt_embs=target, partial_mask=partial_mask, model_initializer=model_adapt)
                # else:
                #     print('haha', n_count, i)

                    break
            return cur_y, model_adapt

        if self.pretrain_step == self.num_timesteps-num_step:

            cur_y, model_adapt = get_sample(
                1, y_pred_raw, model_adapt, adapt_grad=True)

            for k in range(2, num_step):
                # af

                img = img_noise
                for time, time_next in time_pairs_:

                    pred_noise, x_start, * \
                        _ = model_adapt.model_predictions_inf(
                            x, x_mark, img, time_cond)

                    if time_next < num_step:
                        img = x_start
                        continue

                    alpha = self.alphas_cumprod[time]
                    alpha_next = self.alphas_cumprod[time_next]
                    sigma = self.eta * \
                        ((1 - alpha / alpha_next) *
                         (1 - alpha_next) / (1 - alpha)).sqrt()
                    c = (1 - alpha_next - sigma ** 2).sqrt()
                    pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
                    noise = torch.randn_like(img)

                    img = pred_mean + sigma * noise
                    img = img.detach()
                    cur_y = img

                cur_y, model_adapt = get_sample(
                    k, cur_y, model_adapt, adapt_grad=True)

            img = img_noise
            for time, time_next in time_pairs_:
                pred_noise, x_start, * \
                    _ = model_adapt.model_predictions_inf(
                        x, x_mark, img, time_cond)

                if time_next < num_step:
                    img = x_start
                    continue

                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]
                sigma = self.eta * ((1 - alpha / alpha_next)
                                    * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()
                pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
                noise = torch.randn_like(img)

                img = pred_mean + sigma * noise

                y_pred_raw = img.detach()

            cur_y_new, model_adapt = get_sample(
                num_step, y_pred_raw, model_adapt, adapt_grad=False)

        else:
            cur_y, model_adapt = get_sample_fast(
                1, y_pred_raw, model_adapt, adapt_grad=True)

            for k in range(2, num_step+1):

                img = img_noise
                for time, time_next in time_pairs_:

                    pred_noise, x_start, * \
                        _ = model_adapt.model_predictions_inf(
                            x, x_mark, img, time_cond)

                    if time_next < num_step:
                        img = x_start
                        continue

                    alpha = self.alphas_cumprod[time]
                    alpha_next = self.alphas_cumprod[time_next]
                    sigma = self.eta * \
                        ((1 - alpha / alpha_next) *
                         (1 - alpha_next) / (1 - alpha)).sqrt()
                    c = (1 - alpha_next - sigma ** 2).sqrt()
                    pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
                    noise = torch.randn_like(img)

                    img = pred_mean + sigma * noise

                    cur_y = img.detach()

                cur_y, model_adapt = get_sample_fast(
                    k, y_pred_raw, model_adapt, adapt_grad=True)

            img = img_noise
            for time, time_next in time_pairs_:
                pred_noise, x_start, * \
                    _ = model_adapt.model_predictions_inf(
                        x, x_mark, img, time_cond)

                if time_next < num_step:
                    img = x_start
                    continue

                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]
                sigma = self.eta * ((1 - alpha / alpha_next)
                                    * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()
                pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
                noise = torch.randn_like(img)

                img = pred_mean + sigma * noise

                y_pred_raw = img.detach()

            cur_y_new, model_adapt = get_sample_fast(
                num_step+1, y_pred_raw, model_adapt, adapt_grad=False)

        print(cur_y_new)
        del model_adapt
        return cur_y_new

    def p_sample_infill_test_loop_accelerate_short_af(self, maml, num_step, x, x_mark, time_cond, adapt_grad=True, af_pred=None, **kwargs):
        model_adapt = maml.clone()
        x = x.to(torch.float32)
        x_mark = x_mark.to(torch.float32)
        x_feat = x[:, :self.feat_len, :].to(torch.float32)
        x_pred_mark = x_mark[:, self.feat_len:, :].to(torch.float32)
        x_feat_mark = x_mark[:, :self.feat_len, :].to(torch.float32)

        target = x[:, self.feat_len//2:, :]
        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(self.num_timesteps-self.pretrain_step-1,
                               self.num_timesteps - 1, steps=self.sampling_timesteps + 1)  # [201]

        times = list(reversed(times.int().tolist()))
        # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        time_pairs_ = list(zip(times[:-1], times[1:]))

        y_pred_raw = af_pred.to(x.device).detach()

        # y_pred_raw,target = model_adapt.meta_pred(x,x_mark, t=time_cond, **kwargs)

        partial_mask_2 = torch.zeros(
            y_pred_raw[:, self.feat_len//2:, :].size(), dtype=torch.bool)
        partial_mask_1 = torch.ones(
            y_pred_raw[:, :self.feat_len//2, :].size(), dtype=torch.bool)
        partial_mask = torch.cat([partial_mask_1, partial_mask_2], dim=1)

        def get_sample_fast(n_step, cur_y, model_adapt, adapt_grad=True):

            n_count = 0
            times = torch.linspace(-1, self.num_timesteps -
                                   self.pretrain_step - 1, steps=num_step + 1)  # [201]
            times = list(reversed(times.int().tolist()))
            # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
            time_pairs = list(zip(times[:-1], times[1:]))
            for i in reversed(range(num_step)):

                t = time_pairs[n_count][0]
                t = torch.tensor([t]).to(x.device)

                x_start = model_adapt.output(
                    cur_y, x_feat, x_pred_mark, x_feat_mark, t)

                if time_pairs[n_count][1] < 0:
                    cur_y = x_start
                    continue
                maybe_clip = partial(torch.clamp, min=-1., max=1.)
                x_start = maybe_clip(x_start)
                pred_noise = self.predict_noise_from_start(cur_y, t, x_start)

                alpha = self.alphas_cumprod[time_pairs[n_count][0]]
                alpha_next = self.alphas_cumprod[time_pairs[n_count][1]]
                sigma = self.eta * ((1 - alpha / alpha_next)
                                    * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()
                pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
                noise = torch.randn_like(cur_y)
                pred_img = pred_mean + sigma * noise

                if adapt_grad:
                    cur_y = pred_img.detach()

                n_count += 1
                if n_count >= n_step:
                    if adapt_grad:
                        model_adapt = self.langevin_fn_train_fast(x=x, x_mark=x_mark, sample=pred_img, mean=pred_mean, sigma=sigma, t=t,
                                                                  tgt_embs=target, partial_mask=partial_mask, model_initializer=model_adapt)

                    break
            return cur_y, model_adapt

        def get_sample(n_step, cur_y, model_adapt, adapt_grad=True):
            n_count = 0

            for i in reversed(range(num_step)):

                t = torch.tensor([i]).to(x.device)

                x_start = model_adapt.output(
                    cur_y, x_feat, x_pred_mark, x_feat_mark, t)

                model_mean, _, posterior_log_variance = \
                    self.q_posterior(x_start=x_start, x_t=cur_y, t=t)
                noise = torch.randn_like(cur_y) if t > 0 else 0.
                sigma = (0.5 * posterior_log_variance).exp()
                pred_img = model_mean + sigma * noise

                if adapt_grad:
                    cur_y = pred_img.detach()

                n_count += 1
                if n_count == num_step:
                    cur_y = pred_img
                    break
                if n_count >= n_step:
                    if adapt_grad:
                        model_adapt = self.langevin_fn_train_fast(x=x, x_mark=x_mark, sample=pred_img, mean=model_mean, sigma=sigma, t=t,
                                                                  tgt_embs=target, partial_mask=partial_mask, model_initializer=model_adapt)
                    break
            return cur_y, model_adapt

        def get_sample_fast_first(n_step, cur_y, model_adapt, adapt_grad=True):

            n_count = 0
            times = torch.linspace(-1, self.num_timesteps -
                                   self.pretrain_step - 1, steps=num_step + 1)  # [201]
            times = list(reversed(times.int().tolist()))
            # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
            time_pairs = list(zip(times[:-1], times[1:]))
            for i in reversed(range(num_step)):

                t = time_pairs[n_count][0]
                t = torch.tensor([t]).to(x.device)

                x_start = self.output(
                    cur_y, x_feat, x_pred_mark, x_feat_mark, t)

                if time_pairs[n_count][1] < 0:
                    cur_y = x_start
                    continue
                maybe_clip = partial(torch.clamp, min=-1., max=1.)
                x_start = maybe_clip(x_start)
                pred_noise = self.predict_noise_from_start(cur_y, t, x_start)

                alpha = self.alphas_cumprod[time_pairs[n_count][0]]
                alpha_next = self.alphas_cumprod[time_pairs[n_count][1]]
                sigma = self.eta * ((1 - alpha / alpha_next)
                                    * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()
                pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
                noise = torch.randn_like(cur_y)
                pred_img = pred_mean + sigma * noise

                if adapt_grad:
                    cur_y = pred_img.detach()

                n_count += 1
                if n_count >= n_step:
                    if adapt_grad:
                        model_adapt = self.langevin_fn_train_fast_first(x=x, x_mark=x_mark, sample=pred_img, mean=pred_mean, sigma=sigma, t=t,
                                                                        tgt_embs=target, partial_mask=partial_mask, model_initializer=model_adapt)
                # else:
                #     print('haha', n_count, i)

                    break
            return cur_y, model_adapt

        def get_sample_first(n_step, cur_y, model_adapt, adapt_grad=True):
            n_count = 0

            for i in reversed(range(num_step)):

                t = torch.tensor([i]).to(x.device)

                x_start = self.output(
                    cur_y, x_feat, x_pred_mark, x_feat_mark, t)

                model_mean, _, posterior_log_variance = \
                    self.q_posterior(x_start=x_start, x_t=cur_y, t=t)
                noise = torch.randn_like(cur_y) if t > 0 else 0.
                sigma = (0.5 * posterior_log_variance).exp()
                pred_img = model_mean + sigma * noise

                if adapt_grad:
                    cur_y = pred_img.detach()

                n_count += 1
                if n_count >= n_step:
                    if adapt_grad:
                        model_adapt = self.langevin_fn_train_fast_first(x=x, x_mark=x_mark, sample=pred_img, mean=model_mean, sigma=sigma, t=t,
                                                                        tgt_embs=target, partial_mask=partial_mask, model_initializer=model_adapt)
                    break
            return cur_y, model_adapt

        if self.pretrain_step == self.num_timesteps-num_step:

            cur_y, model_adapt = get_sample_first(
                1, y_pred_raw, model_adapt, adapt_grad=True)

            for k in range(2, num_step):

                cur_y, model_adapt = get_sample(
                    k, y_pred_raw, model_adapt, adapt_grad=True)

            cur_y_new, model_adapt = get_sample(
                num_step, y_pred_raw, model_adapt, adapt_grad=False)

        else:
            cur_y, model_adapt = get_sample_fast_first(
                1, y_pred_raw, model_adapt, adapt_grad=True)

            for k in range(2, num_step+1):

                cur_y, model_adapt = get_sample_fast(
                    k, y_pred_raw, model_adapt, adapt_grad=True)

            cur_y_new, model_adapt = get_sample_fast(
                num_step+1, y_pred_raw, model_adapt, adapt_grad=False)

        print(cur_y_new)
        del model_adapt
        return cur_y_new

    def p_sample_infill_test_loop_accelerate_short_af_incontext(self, maml, num_step, x, x_mark, x_incontext, x_in_mark, time_cond, adapt_grad=True,
                                                                af_pred=None, af_incontext=None, adapt_h=False, adapt_f=False, adapt_f_num=0, **kwargs):
        model_adapt = maml.clone()
        x = x.to(torch.float32)
        x_mark = x_mark.to(torch.float32)
        x_feat = x[:, :self.feat_len, :].to(torch.float32)
        x_pred_mark = x_mark[:, self.feat_len:, :].to(torch.float32)
        x_feat_mark = x_mark[:, :self.feat_len, :].to(torch.float32)

        target = x[:, self.feat_len//2:, :]
        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(self.num_timesteps-self.pretrain_step-1,
                               self.num_timesteps - 1, steps=self.sampling_timesteps + 1)  # [201]

        times = list(reversed(times.int().tolist()))
        # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        time_pairs_ = list(zip(times[:-1], times[1:]))

        y_pred_raw = af_pred.to(x.device).detach()

        y_pred_raw_incontext = af_incontext

        # y_pred_raw,target = model_adapt.meta_pred(x,x_mark, t=time_cond, **kwargs)

        partial_mask_2 = torch.zeros(
            y_pred_raw[:, self.feat_len//2:, :].size(), dtype=torch.bool)
        partial_mask_1 = torch.ones(
            y_pred_raw[:, :self.feat_len//2, :].size(), dtype=torch.bool)
        partial_mask = torch.cat([partial_mask_1, partial_mask_2], dim=1)

        def get_sample_fast(n_step, cur_y, x_incontext_, model_adapt, adapt_grad=True):

            n_count = 0
            times = torch.linspace(-1, self.num_timesteps -
                                   self.pretrain_step - 1, steps=num_step + 1)  # [201]
            times = list(reversed(times.int().tolist()))
            # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
            time_pairs = list(zip(times[:-1], times[1:]))
            for i in reversed(range(num_step)):

                t = time_pairs[n_count][0]
                t = torch.tensor([t]).to(x.device)

                x_start = model_adapt.output(
                    cur_y, x_feat, x_pred_mark, x_feat_mark, t)

                if time_pairs[n_count][1] < 0:
                    cur_y = x_start
                    continue
                maybe_clip = partial(torch.clamp, min=-1., max=1.)
                x_start = maybe_clip(x_start)
                pred_noise = self.predict_noise_from_start(cur_y, t, x_start)

                alpha = self.alphas_cumprod[time_pairs[n_count][0]]
                alpha_next = self.alphas_cumprod[time_pairs[n_count][1]]
                sigma = self.eta * ((1 - alpha / alpha_next)
                                    * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()
                pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
                noise = torch.randn_like(cur_y)
                pred_img = pred_mean + sigma * noise

                if adapt_grad:
                    cur_y = pred_img.detach()

                n_count += 1
                if n_count >= n_step:
                    if adapt_grad:
                        model_adapt = self.langevin_fn_train_fast(x=x, x_mark=x_mark, sample=pred_img, mean=pred_mean, sigma=sigma, t=t,
                                                                  tgt_embs=target, partial_mask=partial_mask, model_initializer=model_adapt)

                    break
            return cur_y, model_adapt

        def get_sample(n_step, cur_y, x_incontext_, model_adapt, adapt_grad=True):
            n_count = 0

            for i in reversed(range(num_step)):

                t = torch.tensor([i]).to(x.device)

                x_start = model_adapt.output(
                    cur_y, x_feat, x_pred_mark, x_feat_mark, t)

                model_mean, _, posterior_log_variance = \
                    self.q_posterior(x_start=x_start, x_t=cur_y, t=t)
                noise = torch.randn_like(cur_y) if t > 0 else 0.
                sigma = (0.5 * posterior_log_variance).exp()
                pred_img = model_mean + sigma * noise

                if adapt_grad:
                    cur_y = pred_img.detach()

                n_count += 1
                if n_count == num_step:
                    cur_y = pred_img
                    break
                if n_count >= n_step:
                    if adapt_grad:
                        model_adapt = self.langevin_fn_train_fast(x=x, x_mark=x_mark, sample=pred_img, mean=model_mean, sigma=sigma, t=t,
                                                                  tgt_embs=target, partial_mask=partial_mask, model_initializer=model_adapt)
                    break
            return cur_y, model_adapt

        def get_sample_incontext(n_step, cur_y, x_incontext_, model_adapt, adapt_grad=True, x_incontext=None):
            n_count = 0
            # if x_incontext is not None:
            #     cur_y, model_adapt = get_sample_history(y_pred_raw,model_adapt, adapt_grad=True,x_incontext=x_incontext)
            for i in reversed(range(num_step)):

                t = torch.tensor([i]).to(x.device)

                x_start = model_adapt.output(
                    cur_y, x_feat, x_pred_mark, x_feat_mark, t)

                model_mean, _, posterior_log_variance = \
                    self.q_posterior(x_start=x_start, x_t=cur_y, t=t)
                noise = torch.randn_like(cur_y) if t > 0 else 0.
                sigma = (0.5 * posterior_log_variance).exp()
                pred_img = model_mean + sigma * noise

                if adapt_grad:
                    cur_y = pred_img.detach()

                n_count += 1
                if n_count == num_step:
                    cur_y = pred_img
                    break
                if n_count >= n_step:
                    if adapt_grad:
                        if x_incontext is not None:
                            pred_img, model_adapt = get_sample_history(
                                cur_y, model_adapt, adapt_grad=True, x_incontext=x_incontext)

                        model_adapt = self.langevin_fn_train_fast_incontext(x=x, x_mark=x_mark, sample=pred_img, mean=model_mean, sigma=sigma, t=t,
                                                                            tgt_embs=x_incontext_, partial_mask=partial_mask, model_initializer=model_adapt)
                    break
            return cur_y, model_adapt

        def get_sample_fast_first(n_step, cur_y, x_incontext_, model_adapt, adapt_grad=True):

            n_count = 0
            times = torch.linspace(-1, self.num_timesteps -
                                   self.pretrain_step - 1, steps=num_step + 1)  # [201]
            times = list(reversed(times.int().tolist()))
            # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
            time_pairs = list(zip(times[:-1], times[1:]))
            for i in reversed(range(num_step)):

                t = time_pairs[n_count][0]
                t = torch.tensor([t]).to(x.device)

                x_start = self.output(
                    cur_y, x_feat, x_pred_mark, x_feat_mark, t)

                if time_pairs[n_count][1] < 0:
                    cur_y = x_start
                    continue
                maybe_clip = partial(torch.clamp, min=-1., max=1.)
                x_start = maybe_clip(x_start)
                pred_noise = self.predict_noise_from_start(cur_y, t, x_start)

                alpha = self.alphas_cumprod[time_pairs[n_count][0]]
                alpha_next = self.alphas_cumprod[time_pairs[n_count][1]]
                sigma = self.eta * ((1 - alpha / alpha_next)
                                    * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()
                pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
                noise = torch.randn_like(cur_y)
                pred_img = pred_mean + sigma * noise

                if adapt_grad:
                    cur_y = pred_img.detach()

                n_count += 1
                if n_count >= n_step:
                    if adapt_grad:
                        model_adapt = self.langevin_fn_train_fast_first(x=x, x_mark=x_mark, sample=pred_img, mean=pred_mean, sigma=sigma, t=t,
                                                                        tgt_embs=x_incontext_, partial_mask=partial_mask, model_initializer=model_adapt)
                # else:
                #     print('haha', n_count, i)

                    break
            return cur_y, model_adapt

        def get_sample_first(n_step, cur_y, model_adapt, adapt_grad=True):
            n_count = 0

            for i in reversed(range(num_step)):

                t = torch.tensor([i]).to(x.device)

                x_start = self.output(
                    cur_y, x_feat, x_pred_mark, x_feat_mark, t)

                model_mean, _, posterior_log_variance = \
                    self.q_posterior(x_start=x_start, x_t=cur_y, t=t)
                noise = torch.randn_like(cur_y) if t > 0 else 0.
                sigma = (0.5 * posterior_log_variance).exp()
                pred_img = model_mean + sigma * noise

                if adapt_grad:
                    cur_y = pred_img.detach()

                n_count += 1
                if n_count >= n_step:
                    if adapt_grad:
                        model_adapt = self.langevin_fn_train_fast_first(x=x, x_mark=x_mark, sample=pred_img, mean=model_mean, sigma=sigma, t=t,
                                                                        tgt_embs=target, partial_mask=partial_mask, model_initializer=model_adapt)
                    break
            return cur_y, model_adapt

        def get_sample_first_incontext(n_step, cur_y, model_adapt, adapt_grad=True, x_incontext=None):
            n_count = 0
            # if x_incontext is not None:

            #     cur_y, model_adapt = get_sample_history(y_pred_raw,model_adapt, adapt_grad=True,x_incontext=x_incontext)
            for i in reversed(range(num_step)):

                t = torch.tensor([i]).to(x.device)

                x_start = self.output(
                    cur_y, x_feat, x_pred_mark, x_feat_mark, t)

                model_mean, _, posterior_log_variance = \
                    self.q_posterior(x_start=x_start, x_t=cur_y, t=t)
                noise = torch.randn_like(cur_y) if t > 0 else 0.
                sigma = (0.5 * posterior_log_variance).exp()
                pred_img = model_mean + sigma * noise

                if adapt_grad:
                    cur_y = pred_img.detach()
                # print(x_start.shape) [64,42,17]
                # exit()
                n_count += 1
                if n_count >= n_step:
                    if adapt_grad:

                        model_adapt = self.langevin_fn_train_fast_first(x=x, x_mark=x_mark, sample=pred_img, mean=model_mean, sigma=sigma, t=t,
                                                                        tgt_embs=target, partial_mask=partial_mask, model_initializer=model_adapt)
                    break
            return cur_y, model_adapt

        def get_sample_history(cur_y, model_adapt, adapt_grad=True, x_incontext=x_incontext):
            n_count = 0
            his_adapt_step = x_incontext.shape[1]

            for i in range(his_adapt_step):

                t = torch.tensor([i]).to(x.device)

                x_start = self.output(
                    cur_y, x_feat, x_pred_mark, x_feat_mark, t)

                model_mean, _, posterior_log_variance = \
                    self.q_posterior(x_start=x_start, x_t=cur_y, t=t)
                noise = torch.randn_like(cur_y) if t > 0 else 0.
                sigma = (0.5 * posterior_log_variance).exp()
                pred_img = model_mean + sigma * noise

                if adapt_grad:
                    cur_y = pred_img.detach()
                # print(x_start.shape) [64,42,17]
                # exit()
                n_count += 1

                if adapt_grad:

                    model_adapt = self.langevin_fn_train_fast_his(x=x, x_mark=x_mark, sample=pred_img, mean=model_mean, sigma=sigma, t=t,
                                                                  tgt_embs=x_incontext[:, i, :, :], partial_mask=partial_mask, model_initializer=model_adapt)

            return cur_y, model_adapt
        # TODO:change -- incontext
        self.adapt_incontext = adapt_f
        self.iteration = 6
        self.adapt_num = adapt_f_num
        self.adapt_h = adapt_h
        # TODO:lr的下降
        if self.pretrain_step == self.num_timesteps-num_step:
            if self.adapt_h:
                for hh in range(x_incontext.shape[1]):
                    y_input = y_pred_raw_incontext[hh].to(x.device).detach()
                    cur_y, model_adapt = get_sample_first(
                        1, y_input, model_adapt, adapt_grad=True)

                    for k in range(2, num_step):

                        cur_y, model_adapt = get_sample(
                            k, y_input, x_incontext, model_adapt, adapt_grad=True)
                    cur_y_new, model_adapt = get_sample(
                        num_step, y_input, x_incontext, model_adapt, adapt_grad=False)
            if not self.adapt_incontext:
                # if self.adapt_h:
                #     ix = 0
                #     if x_incontext is not None:
                #         cur_y, model_adapt = get_sample_history(y_pred_raw,model_adapt, adapt_grad=True,x_incontext=x_incontext)
                #     cur_y, model_adapt = get_sample_first(1, y_pred_raw,model_adapt, adapt_grad=True)

                #     for k in range(2, num_step):
                #         ix+=1
                #         # if x_incontext is not None:
                #         #     cur_y, model_adapt = get_sample_history(y_pred_raw,model_adapt, adapt_grad=True,x_incontext=x_incontext)

                #         cur_y, model_adapt = get_sample(k, y_pred_raw, x_incontext,model_adapt, adapt_grad=True)
                #         if x_incontext is not None:
                #             cur_y, model_adapt = get_sample_history(y_pred_raw,model_adapt, adapt_grad=True,x_incontext=x_incontext)
                #     # if x_incontext is not None:
                #     #     cur_y, model_adapt = get_sample_history(y_pred_raw,model_adapt, adapt_grad=True,x_incontext=x_incontext)
                #     cur_y_new, model_adapt = get_sample(num_step, y_pred_raw, x_incontext,model_adapt, adapt_grad=False)
                #     if x_incontext is not None:
                #         cur_y, model_adapt = get_sample_history(y_pred_raw,model_adapt, adapt_grad=True,x_incontext=x_incontext)
                #     cur_y_new = cur_y_new[:,self.feat_len//2:,:]

                ix = 0
                cur_y, model_adapt = get_sample_first(
                    1, y_pred_raw, model_adapt, adapt_grad=True)

                for k in range(2, num_step):
                    ix += 1
                    cur_y, model_adapt = get_sample(
                        k, y_pred_raw, x_incontext, model_adapt, adapt_grad=True)

                cur_y_new, model_adapt = get_sample(
                    num_step, y_pred_raw, x_incontext, model_adapt, adapt_grad=False)
                cur_y_new = cur_y_new[:, self.feat_len//2:, :]
                
            else:

                for r in range(self.iteration):

                    adapt_output = []
                    ix = 0
                    cur_y, model_adapt = get_sample_first_incontext(
                        1, y_pred_raw, model_adapt, adapt_grad=True)
                    adapt_output.append(cur_y[:, -self.seq_length:, :])
                    for k in range(2, self.adapt_num):
                        ix += 1
                        cur_y, model_adapt = get_sample_incontext(
                            k, y_pred_raw, cur_y, model_adapt, adapt_grad=True)
                        adapt_output.append(cur_y[:, -self.seq_length:, :])
                    cur_y_new, model_adapt = get_sample_incontext(
                        self.adapt_num, y_pred_raw, cur_y, model_adapt, adapt_grad=False)
                    adapt_output.append(cur_y_new[:, -self.seq_length:, :])
                cur_y_new = torch.concat(adapt_output, dim=1)

        else:
            cur_y, model_adapt = get_sample_fast_first(
                1, y_pred_raw, model_adapt, adapt_grad=True)

            for k in range(2, num_step+1):

                cur_y, model_adapt = get_sample_fast(
                    k, y_pred_raw, model_adapt, adapt_grad=True)

            cur_y_new, model_adapt = get_sample_fast(
                num_step+1, y_pred_raw, model_adapt, adapt_grad=False)

        del model_adapt
        return cur_y_new


if __name__ == '__main__':
    pass
