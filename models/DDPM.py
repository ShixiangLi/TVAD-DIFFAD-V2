import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T

def get_beta_schedule(num_diffusion_steps, name="cosine"):
    betas = []
    if name == "cosine":
        max_beta = 0.999
        f = lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
        for i in range(num_diffusion_steps):
            t1 = i / num_diffusion_steps
            t2 = (i + 1) / num_diffusion_steps
            betas.append(min(1 - f(t2) / f(t1), max_beta))
        betas = np.array(betas)
    elif name == "linear":
        scale = 1000 / num_diffusion_steps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        betas = np.linspace(beta_start, beta_end, num_diffusion_steps, dtype=np.float64)
    else:
        raise NotImplementedError(f"unknown beta schedule: {name}")
    return betas


def extract(arr, timesteps, broadcast_shape, device):
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape).to(device)


def mean_flat(tensor):
    return torch.mean(tensor, dim=list(range(1, len(tensor.shape))))


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL Divergence between two gaussians

    :param mean1:
    :param logvar1:
    :param mean2:
    :param logvar2:
    :return: KL Divergence between N(mean1,logvar1^2) & N(mean2,logvar2^2))
    """
    return 0.5 * (-1 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2))


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretised_gaussian_log_likelihood(x, means, log_scales):
    """
        Compute the log-likelihood of a Gaussian distribution discretizing to a
        given image.
        :param x: the target images. It is assumed that this was uint8 values,
                  rescaled to the range [-1, 1].
        :param means: the Gaussian mean Tensor.
        :param log_scales: the Gaussian log stddev Tensor.
        :return: a tensor like x of log probabilities (in nats).
        """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)

    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)

    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))

    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
            x < -0.999,
            log_cdf_plus,
            torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
            )
    assert log_probs.shape == x.shape
    return log_probs




class GaussianDiffusionModel:
    def __init__(
            self,
            img_size,
            betas,
            img_channels=1,
            loss_type="l2",  # l2,l1 hybrid
            loss_weight='none',  # prop t / uniform / None
            noise="gauss",  # gauss / perlin / simplex
            ):
        super().__init__()

        if noise == "gauss":
            self.noise_fn = lambda x, t: torch.randn_like(x)
        # Add other noise types if needed (perlin, simplex)
        # else:
        #     raise NotImplementedError(f"Noise type {noise} not implemented")


        self.img_size = img_size
        self.img_channels = img_channels
        self.loss_type = loss_type
        self.num_timesteps = len(betas)

        if loss_weight == 'prop-t':
            self.weights = np.arange(self.num_timesteps, 0, -1).astype(np.float32) # Ensure float for division
        elif loss_weight == "uniform":
            self.weights = np.ones(self.num_timesteps, dtype=np.float32)
        # If 'none', self.weights is not set, handle in sample_t_with_weights

        self.loss_weight = loss_weight # Store the string name
        alphas = 1.0 - betas # Ensure float subtraction
        self.betas = betas
        self.sqrt_alphas = np.sqrt(alphas)
        self.sqrt_betas = np.sqrt(betas)

        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        # self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:],0.0) # Not used


        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1.0) # Ensure float

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
                betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
                np.maximum(self.posterior_variance, 1e-20) # Clip small values before log
                ) 
        # Original code appended self.posterior_variance[1] for the first element.
        # A common practice is to use the beta_0 for the first step if defined, or handle t=0 carefully.
        # For consistency with OpenAI's guide-diffusion, they often use:
        # self.posterior_log_variance_clipped = np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
        # Let's stick to the direct log of potentially clipped posterior_variance for now.


        self.posterior_mean_coef1 = (
                betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * np.sqrt(alphas) # This is sqrt_alphas
                / (1.0 - self.alphas_cumprod)
        )

        self.gauss_blur = T.GaussianBlur(kernel_size=31, sigma=3) # Unused in current DDPM logic shown


    def sample_t_with_weights(self, b_size, device):
        if self.loss_weight == 'none' or not hasattr(self, 'weights'): # Uniform sampling if no weights
            p = np.ones(self.num_timesteps, dtype=np.float32) / self.num_timesteps
        else:
            p = self.weights / np.sum(self.weights)
        
        indices_np = np.random.choice(len(p), size=b_size, p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        
        # weights_np = 1 / len(p) * p[indices_np] # This is not importance sampling weight, but probability
        # Importance sampling weights for loss re-weighting (if needed) would be 1 / (N * p_i)
        # For now, the 'weights' attribute seems to control the sampling distribution of t, not loss re-weighting directly here.
        # If loss re-weighting by 1/p(t) is intended, it should be applied in the loss calculation.
        # The original code returns 'weights' that look like 1 / (N * p_t) but N is len(p) here.
        # Let's return the sampled timesteps. If loss re-weighting is needed, it's usually handled outside or with a different mechanism.
        # For now, returning just indices. If `weights_for_loss` is needed, it should be 1.0 / (self.num_timesteps * p[indices_np])
        
        # Returning timesteps and placeholder weights if the original structure expected it
        # However, these returned 'weights' are not standard importance weights for loss.
        # Let's assume for now weights are for sampling distribution of t, and loss weighting is separate.
        # If the original 'weights' return was meant for loss weighting by 1/p(t):
        # effective_probs = p[indices_np]
        # importance_weights = 1.0 / (self.num_timesteps * effective_probs) # Assuming N = self.num_timesteps
        # importance_weights_tensor = torch.from_numpy(importance_weights).float().to(device)
        # return indices, importance_weights_tensor
        
        # Sticking to simpler return of indices, assuming loss weighting is handled if 'prop-t' or 'uniform' selected for actual loss value.
        return indices, torch.ones_like(indices, dtype=torch.float32) # Placeholder weights, not used in current loss calculation

    def predict_x_0_from_eps(self, x_t, t, eps):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape, x_t.device) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape, x_t.device) * eps)

    def predict_eps_from_x_0(self, x_t, t, pred_x_0):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape, x_t.device) * x_t
                - pred_x_0) \
               / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape, x_t.device)

    def q_mean_variance(self, x_0, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
                extract(self.sqrt_alphas_cumprod, t, x_0.shape, x_0.device) * x_0
        )
        variance = extract(1.0 - self.alphas_cumprod, t, x_0.shape, x_0.device)
        log_variance = extract(
                self.log_one_minus_alphas_cumprod, t, x_0.shape, x_0.device
                )
        return mean, variance, log_variance

    def q_posterior_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        posterior_mean = (extract(self.posterior_mean_coef1, t, x_t.shape, x_t.device) * x_0
                          + extract(self.posterior_mean_coef2, t, x_t.shape, x_t.device) * x_t)
        posterior_var = extract(self.posterior_variance, t, x_t.shape, x_t.device)
        posterior_log_var_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape, x_t.device)
        return posterior_mean, posterior_var, posterior_log_var_clipped

    def p_mean_variance(self, model, x_t, t, current_features, estimate_noise=None): # Added current_features
        """
        Finds the mean & variance from N(x_{t-1}; mu_theta(x_t,t), sigma_theta (x_t,t))
        """
        if estimate_noise is None: # Pass current_features to model
            estimate_noise = model(x_t, t, current_features)


        # Model variance can be fixed or learned. Here it's fixed.
        # Commonly, model_var is either self.betas or self.posterior_variance
        # The original code uses: np.append(self.posterior_variance[1], self.betas[1:]) which is a bit unusual.
        # Let's use self.posterior_variance as it's related to q(x_{t-1}|x_t, x_0) or simply self.betas.
        # Using self.betas for model_var_values (fixed variance to \beta_t)
        model_var_values = self.betas
        model_log_var_values = np.log(np.maximum(model_var_values, 1e-20)) # Clip for log

        model_var = extract(model_var_values, t, x_t.shape, x_t.device)
        model_logvar = extract(model_log_var_values, t, x_t.shape, x_t.device)

        pred_x_0 = self.predict_x_0_from_eps(x_t, t, estimate_noise).clamp(-1, 1)
        model_mean, _, _ = self.q_posterior_mean_variance(
                pred_x_0, x_t, t
                )
        return {
            "mean":         model_mean,
            "variance":     model_var,
            "log_variance": model_logvar,
            "pred_x_0":     pred_x_0,
            }

    def sample_p(self, model, x_t, t, current_features, denoise_fn_name="gauss"): # Added current_features, renamed denoise_fn
        out = self.p_mean_variance(model, x_t, t, current_features) # Pass current_features
        
        if denoise_fn_name == "gauss": # Assuming denoise_fn_name refers to the noise sampling strategy
            noise = torch.randn_like(x_t) 
        else:
            # If other denoise_fn (like learned noise) are intended, they need to be implemented
            # noise = self.noise_fn(x_t, t) # This was original self.noise_fn for forward, not for p_sample noise
            raise NotImplementedError(f"denoise_fn {denoise_fn_name} not implemented for p_sample")


        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))) # Mask for t=0 where noise is not added
        )

        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_x_0": out["pred_x_0"]}

    def forward_backward( # This method seems for full sampling, ensure current_features are handled if used.
            self, model, x, current_features, see_whole_sequence="half", t_distance=None, denoise_fn_name="gauss",
            ):
        assert see_whole_sequence in ["whole", "half", None]

        if t_distance == 0:
            return x.detach()

        if t_distance is None:
            t_distance = self.num_timesteps # Default to full num_timesteps
        
        # Ensure current_features are correctly shaped for batch if x is a batch
        # Assuming current_features is already batched [B, 3] matching x [B, C, H, W]

        img_seq = [x.cpu().detach()] # Store sequence of images

        # Forward process (noising)
        if see_whole_sequence == "whole": # Diffuse step-by-step
            current_x_t = x
            for t_step in range(t_distance):
                t_batch = torch.tensor([t_step], device=x.device).repeat(x.shape[0])
                noise = self.noise_fn(current_x_t, t_batch).float() # Use noise_fn for forward diffusion
                current_x_t = self.sample_q_gradual(current_x_t, t_batch, noise) # q(x_t | x_{t-1})
                img_seq.append(current_x_t.cpu().detach())
            x_noised = current_x_t
        else: # Diffuse in one step to t_distance
            t_tensor = torch.tensor([t_distance - 1], device=x.device).repeat(x.shape[0])
            noise = self.noise_fn(x, t_tensor).float()
            x_noised = self.sample_q(x, t_tensor, noise) # q(x_t | x_0)
            if see_whole_sequence == "half":
                img_seq.append(x_noised.cpu().detach())

        # Backward process (denoising)
        current_x_t = x_noised
        for t_step in range(t_distance - 1, -1, -1): # From T-1 down to 0
            t_batch = torch.tensor([t_step], device=x.device).repeat(x.shape[0])
            with torch.no_grad():
                out = self.sample_p(model, current_x_t, t_batch, current_features, denoise_fn_name=denoise_fn_name)
                current_x_t = out["sample"]
            if see_whole_sequence: # "whole" or "half" (if half, only last step is added here)
                img_seq.append(current_x_t.cpu().detach())
        
        # If not seeing whole sequence, return only the final denoised sample
        return current_x_t.detach() if not see_whole_sequence else img_seq


    def sample_q(self, x_0, t, noise):
        """
            q (x_t | x_0 )
        """
        return (extract(self.sqrt_alphas_cumprod, t, x_0.shape, x_0.device) * x_0 +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape, x_0.device) * noise)

    def sample_q_gradual(self, x_prev_t, t, noise): # Renamed x_t to x_prev_t for clarity
        """
        q (x_t | x_{t-1})
        :param x_prev_t: image at t-1
        :param t: current timestep t
        :param noise:
        :return: image at t
        """
        # Note: t here should be the target t, so coefficients are for x_t from x_{t-1}
        # sqrt_alphas[t] and sqrt_betas[t] are correct.
        return (extract(self.sqrt_alphas, t, x_prev_t.shape, x_prev_t.device) * x_prev_t +
                extract(self.sqrt_betas, t, x_prev_t.shape, x_prev_t.device) * noise)

    def calc_vlb_xt(self, model, x_0, x_t, t, current_features, estimate_noise=None): # Added current_features
        # find KL divergence at t
        true_mean, _, true_log_var = self.q_posterior_mean_variance(x_0, x_t, t)
        output = self.p_mean_variance(model, x_t, t, current_features, estimate_noise) # Pass current_features
        kl = normal_kl(true_mean, true_log_var, output["mean"], output["log_variance"])
        kl = mean_flat(kl) / np.log(2.0) # Convert to bits

        # NLL for x_0 given the predicted distribution for x_0 at t=0
        decoder_nll = -discretised_gaussian_log_likelihood(
                x_0, means=output["mean"], log_scales=0.5 * output["log_variance"] # x_0, predicted mean of x_{t-1}, predicted var of x_{t-1}
                )
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # Choose KL for t > 0 and NLL for t = 0
        nll = torch.where((t == 0), decoder_nll, kl)
        return {"output": nll, "pred_x_0": output["pred_x_0"]}

    def calc_loss(self, model, x_0, t, current_features): # Added current_features
        noise = self.noise_fn(x_0, t).float() # Sample noise ε
        x_t = self.sample_q(x_0, t, noise)    # Get x_t = sqrt(α_bar_t)x_0 + sqrt(1-α_bar_t)ε
        
        # Predict noise using the model (U-Net)
        estimate_noise = model(x_t, t, current_features) # Pass current_features
        
        loss_dict = {}
        if self.loss_type == "l1":
            loss_val = mean_flat((estimate_noise - noise).abs())
        elif self.loss_type == "l2":
            loss_val = mean_flat((estimate_noise - noise).square())
        elif self.loss_type == "hybrid": # L2 + VLB
            vlb_output = self.calc_vlb_xt(model, x_0, x_t, t, current_features, estimate_noise=estimate_noise) # Pass current_features
            loss_dict["vlb"] = vlb_output["output"]
            loss_val = loss_dict["vlb"] + mean_flat((estimate_noise - noise).square()) # Should be weighted sum if needed
        else: # Default to l2
            loss_val = mean_flat((estimate_noise - noise).square())
        
        loss_dict["loss"] = loss_val
        return loss_dict, x_t, estimate_noise # Return loss_dict, x_t, and predicted noise

   
    def norm_guided_one_step_denoising(self, model, x_0, anomaly_label, args, current_features): # Added current_features
        # two-scale t
        # Ensure current_features batch size matches x_0.shape[0]
        batch_size = x_0.shape[0]
        
        # Sample timesteps
        normal_t = torch.randint(0, args["less_t_range"], (batch_size,), device=x_0.device)
        noisier_t = torch.randint(args["less_t_range"], self.num_timesteps, (batch_size,), device=x_0.device)
        
        # Calculate losses and get intermediate states
        # Assuming current_features are the same for both normal_t and noisier_t steps for a given x_0
        normal_loss_dict, x_normal_t, estimate_noise_normal = self.calc_loss(model, x_0, normal_t, current_features)
        noisier_loss_dict, x_noiser_t, estimate_noise_noisier = self.calc_loss(model, x_0, noisier_t, current_features)
        
        pred_x_0_noisier = self.predict_x_0_from_eps(x_noiser_t, noisier_t, estimate_noise_noisier).clamp(-1, 1)
        # To sample x_t from a predicted x_0, we need noise. 
        # The original used estimate_noise_normal. This implies using the *model's prediction* of noise at normal_t as if it were true noise.
        # This is a specific choice for constructing pred_x_t_noisier.
        # Let's assume the intention is to use a fresh random noise if estimate_noise_normal is not directly suitable as a "true" noise sample here.
        # However, sticking to original logic:
        noise_for_pred_x_t_noisier = estimate_noise_normal # This is unusual, typically you'd use fresh noise.
                                                          # If this is model's *output* (predicted noise), using it as *input* noise needs care.
                                                          # Let's assume it's a shorthand for some desired conditioning.
                                                          # A safer bet, if standard diffusion sampling is intended from pred_x_0_noisier to normal_t:
                                                          # fresh_noise_at_normal_t = self.noise_fn(pred_x_0_noisier, normal_t).float()
                                                          # pred_x_t_noisier = self.sample_q(pred_x_0_noisier, normal_t, fresh_noise_at_normal_t)

        # Sticking to the paper's apparent structure if `estimate_noise_normal` is used as noise input for `sample_q`
        pred_x_t_noisier = self.sample_q(pred_x_0_noisier, normal_t, noise_for_pred_x_t_noisier)

        # Only calculate the noise loss of normal samples according to formula 9.
        # The loss is a sum of losses from two different t scales.
        combined_loss = normal_loss_dict["loss"] + noisier_loss_dict["loss"]
        
        # Filter for normal samples (anomaly_label == 0)
        # Ensure anomaly_label is correctly shaped for broadcasting if needed, or use boolean indexing.
        # anomaly_label should be [batch_size]
        if anomaly_label.ndim > 1: # E.g. [batch_size, 1]
            anomaly_label_flat = anomaly_label.squeeze()
        else:
            anomaly_label_flat = anomaly_label

        loss = combined_loss[anomaly_label_flat == 0].mean()
        
        if torch.isnan(loss): # If batch contained only anomalies, mean results in NaN
            loss = torch.tensor(0.0, device=x_0.device, requires_grad=True) # Ensure it's a tensor that requires grad if it's part of loss computation

        # Guidance term (Formula 10 in DiffAD paper implies estimate_noise_hat is for epsilon_theta_hat)
        # The term (pred_x_t_noisier - x_normal_t) is the difference guiding the noise prediction.
        # This difference is scaled by sqrt(1 - alpha_bar_t) and condition_w.
        # estimate_noise_hat = predicted_noise_at_normal_t - guidance_term
        guidance_scale = extract(self.sqrt_one_minus_alphas_cumprod, normal_t, x_normal_t.shape, x_0.device) * args["condition_w"]
        guidance_term = guidance_scale * (pred_x_t_noisier - x_normal_t)
        
        estimate_noise_hat = estimate_noise_normal - guidance_term # This is ε̂_θ(x_normal_t, t_normal, c)
        
        pred_x_0_norm_guided = self.predict_x_0_from_eps(x_normal_t, normal_t, estimate_noise_hat).clamp(-1, 1)

        return loss, pred_x_0_norm_guided, normal_t, x_normal_t, x_noiser_t # Returning scalar loss


    def norm_guided_one_step_denoising_eval(self, model, x_0, normal_t, noisier_t, args, current_features): # Added current_features
        # normal_t and noisier_t are tensors here, already batched.
        batch_size = x_0.shape[0]

        # Assuming current_features are passed and match batch_size
        normal_loss_dict, x_normal_t, estimate_noise_normal = self.calc_loss(model, x_0, normal_t, current_features)
        noisier_loss_dict, x_noiser_t, estimate_noise_noisier = self.calc_loss(model, x_0, noisier_t, current_features)

        pred_x_0_noisier = self.predict_x_0_from_eps(x_noiser_t, noisier_t, estimate_noise_noisier).clamp(-1, 1)
        
        # See note in norm_guided_one_step_denoising about using estimate_noise_normal here
        noise_for_pred_x_t_noisier = estimate_noise_normal 
        pred_x_t_noisier = self.sample_q(pred_x_0_noisier, normal_t, noise_for_pred_x_t_noisier)    

        # In evaluation, loss is typically not filtered by anomaly_label.
        loss = (normal_loss_dict["loss"] + noisier_loss_dict["loss"]).mean() # Average over batch

        # Reconstruction without guidance (standard DDPM prediction from x_normal_t)
        pred_x_0_normal = self.predict_x_0_from_eps(x_normal_t, normal_t, estimate_noise_normal).clamp(-1, 1)
        
        # Guided reconstruction
        guidance_scale = extract(self.sqrt_one_minus_alphas_cumprod, normal_t, x_0.shape, x_0.device) * args["condition_w"]
        guidance_term = guidance_scale * (pred_x_t_noisier - x_normal_t)
        estimate_noise_hat = estimate_noise_normal - guidance_term
        
        pred_x_0_norm_guided = self.predict_x_0_from_eps(x_normal_t, normal_t, estimate_noise_hat).clamp(-1, 1)
        
        return loss, pred_x_0_norm_guided, pred_x_0_normal, pred_x_0_noisier, x_normal_t, x_noiser_t, pred_x_t_noisier  
    

    def noise_t(self, model, x_0, t, args, current_features): # Added current_features and args (if needed by calc_loss indirectly)
        # This method seems simpler, directly calculating loss and predicting x0 from a single t
        loss_dict, x_t, estimate_noise = self.calc_loss(model, x_0, t, current_features) # Pass current_features
        
        loss = (loss_dict["loss"]).mean() # Average loss over batch
        
        pred_x_0 = self.predict_x_0_from_eps(x_t, t, estimate_noise).clamp(-1, 1)
        return loss, pred_x_0, x_t