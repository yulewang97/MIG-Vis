import torch
import numpy as np
from diffusion_model.denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer, normalize_to_01
from diffusion_model.denoising_diffusion_pytorch import mig_vis_p_sample
import argparse
from utils.utils_torch import get_logger


logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with customizable parameters.")
    parser.add_argument("--generation_index", type=int, default=0)
    parser.add_argument("--sample_index_x", type=int, default=0)
    parser.add_argument("--sample_index_z", type=int, default=0)
    parser.add_argument("--group_index", type=int, default=0)
    parser.add_argument("--guidance_scale", type=float, default=100.)
    parser.add_argument("--edit_t", type=int, default=135)
    return parser.parse_args()

args = parse_args()

logger.info("Training Configuration:")
for arg in vars(args):
    logger.info(f"{arg}: {getattr(args, arg)}")

config = vars(args)

sample_index_x = config["sample_index_x"]
sample_index_z_pos = config["sample_index_z"]
group_index = config["group_index"]
generation_index = config["generation_index"]
guidance_scale = config["guidance_scale"]
edit_t = config["edit_t"]

stimulus_data = np.load('datasets/stimulus_data_cleaned.npy')
stimulus_data_normalized, data_min, data_max = normalize_to_01(torch.Tensor(stimulus_data))

stimulus_data_normalized = stimulus_data_normalized.reshape(-1, 128, 128)
stimulus_data_normalized = np.expand_dims(stimulus_data_normalized, axis=1)

logger.info(f"stimulus_data_normalized shape: {stimulus_data_normalized.shape}")


diffusion_timesteps = 150

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = False
)

diffusion = GaussianDiffusion(
    model,
    height = 128,
    width = 128,
    timesteps = diffusion_timesteps,    # number of steps
    guidance_scale = guidance_scale,
    group_index = group_index,
)

trainer = Trainer(
    diffusion,
    stimulus_data_normalized,
    train_batch_size = 64,
    train_lr = 2e-4,
    train_num_steps = 20000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.98,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = False,              # whether to calculate fid during training
)

trainer.load("20")

x_start = torch.Tensor(stimulus_data_normalized[sample_index_x]).unsqueeze(0)
x_start = x_start.repeat(15, 1, 1, 1)  # repeat the image to match the batch size
x_start = x_start.to(diffusion.device)


# sampling with guidance phase
sampled_image, xt = mig_vis_p_sample(diffusion, x_start, sample_index_z_pos, edit_t=edit_t)

sampled_image = sampled_image.reshape(-1, 128, 128)
x_start = x_start.reshape(-1, 128, 128)

sampled_images_denorm = sampled_image * (data_max - data_min) + data_min
xt_denorm = xt * (data_max - data_min) + data_min


torch.save(sampled_images_denorm, f'samples/clg_x_{sample_index_x}_z_{sample_index_z_pos}_gid_{group_index}_timestep_{edit_t}_gscale_{guidance_scale}.pt')