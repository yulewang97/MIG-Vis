import random
import numpy as np
import torch
import math
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import r2_score, mean_squared_error

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True  # Ensure reproducibility
    torch.backends.cudnn.benchmark = False  # May slow down training but ensures determinism


def zscore_2d(arr, axis=1):
    # Along the time dimension 
    means = np.mean(arr, axis=(0), keepdims=False)
    stds = np.std(arr, axis=(0), keepdims=False)
    
    # Perform Z-score normalization
    zscored_arr = (arr - means) / stds
    
    return zscored_arr


def rescale_to_01(sample):
    min_val = sample.min()
    max_val = sample.max()
    return (sample - min_val) / (max_val - min_val)  # Rescale data to [0, 1]

def rescale_to_minus1_1(sample):
    min_val = sample.min()
    max_val = sample.max()
    return 2 * (sample - min_val) / (max_val - min_val) - 1  # Rescale to [-1, 1]

def zscore_by_column(data):
    """
    
    Args:
        data (numpy.ndarray): 2D array of shape [num_samples, num_features].

    Returns:
        numpy.ndarray: Normalized data, with mean 0 and standard deviation 1.
    """
    mean = data.mean(axis=0)  # Calculate mean by column
    std = data.std(axis=0)    # Calculate standard deviation by column
    std[std == 0] = 1         # Prevent division by zero, set std to 1
    return (data - mean) / std


def normalize_to_01(tensor):
    """
    Normalize a tensor to [0, 1] range using min-max normalization.
    
    Args:
        tensor (torch.Tensor): Input tensor to normalize.
    
    Returns:
        normalized_tensor (torch.Tensor): Tensor scaled to [0, 1].
        data_min (float): Minimum value in the original tensor.
        data_max (float): Maximum value in the original tensor.
    """
    data_min = np.min(tensor)
    data_max = np.max(tensor)

    normalized_tensor = (tensor - data_min) / (data_max - data_min + 1e-8)  # Add epsilon to avoid division by zero
    return normalized_tensor, data_min, data_max


class StimulusDataset(Dataset):
    def __init__(self, stimulus_data, transform=None, apply_transform=False):
        self.stimulus_data = stimulus_data
        self.transform = transform
        self.apply_transform = apply_transform  # New parameter to control whether to apply transform

    def __len__(self):
        return len(self.stimulus_data)

    def __getitem__(self, idx):
        stimulus = self.stimulus_data[idx]

        if self.transform is not None and self.apply_transform:
            stimulus = self.transform(stimulus)
        return stimulus

class PairwiseDataset(Dataset):
    def __init__(self, neural_data, stimulus_data, transform=None, apply_transform=False):
        self.neural_data = neural_data
        self.stimulus_data = stimulus_data
        self.transform = transform
        self.apply_transform = apply_transform  # New parameter to control whether to apply transform

    def __len__(self):
        return len(self.stimulus_data)

    def __getitem__(self, idx):
        neural = self.neural_data[idx]
        stimulus = self.stimulus_data[idx]

        if self.transform is not None and self.apply_transform:
            stimulus = self.transform(stimulus)
        return neural, stimulus

class PairwiseImageDataset(Dataset):
    def __init__(self, stimulus_data, label_data, stimulus_transform=None, label_transform=None, apply_transform=False):
        self.stimulus_data = stimulus_data
        self.label_data = label_data
        self.stimulus_transform = stimulus_transform
        self.label_transform = label_transform
        self.apply_transform = apply_transform

    def __len__(self):
        return len(self.stimulus_data)

    def __getitem__(self, idx):
        stimulus = self.stimulus_data[idx]
        label = self.label_data[idx]

        if self.stimulus_transform is not None and self.apply_transform:
            stimulus = self.stimulus_transform(stimulus)
        if self.label_transform is not None and self.apply_transform:
            label = self.label_transform(label)

        return stimulus, label


def warmup_then_decay_lr(current_step, warmup_steps, total_steps, portion = 0.4):
    if current_step < warmup_steps:
        return portion + (1 - portion) * math.sin(0.5 * math.pi * current_step / warmup_steps)
    else:
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        return  0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265359)))


def plotting_random_images(input_matrice):

    # Assuming augmented_images is a numpy array of shape (1200, 256, 256)
    # Randomly select 9 indices from the images
    sample_indices = np.random.choice(len(input_matrice), 9, replace=False)
    # print(sample_indices)

    # Create a 3x3 grid to display the 9 randomly sampled images
    plt.figure(figsize=(4, 4))
    for idx, i in enumerate(sample_indices):
        plt.subplot(3, 3, idx + 1)
        # print(downs_augmented_images[i].shape)
        plt.imshow(input_matrice[i], cmap='gray')  # Display as grayscale image
        plt.axis('off')  # Hide axes

    plt.tight_layout()
    plt.show()
    

def plotting_ordering_images(input_matrice):

    # Assuming augmented_images is a numpy array of shape (1200, 256, 256)
    # Randomly select 9 indices from the first 400 images
    # sample_indices = np.random.choice(len(input_matrice), 9, replace=False)
    sample_indices = np.arange(9)

    # Create a 3x3 grid to display the 9 sampled images
    plt.figure(figsize=(8, 8))
    for idx, i in enumerate(sample_indices):
        plt.subplot(3, 3, idx + 1)
        # print(downs_augmented_images[i].shape)
        plt.imshow(input_matrice[i], cmap='gray')  # Display as grayscale image
        plt.axis('off')  # Hide axes

    plt.tight_layout()
    plt.show()

def get_latents_and_recons(data_loader, model):
    device = next(model.parameters()).device

    latents_list = list()
    recons_list = list()
    
    with torch.no_grad():  
        for i, (images, _) in enumerate(data_loader):
            # latent_vectors = latent_vectors.to(device)
            images = images.unsqueeze(1).to(device)
            
            reconstructed_images, _, _, mu, z, logvar = model(images)

            # Calculate the VAE loss
            latents_list.append(mu.cpu())
            recons_list.append(reconstructed_images.cpu())
            # loss = vae_loss(reconstructed_images, images, mu, logvar)
    
    return latents_list, recons_list

def get_latents_and_recons_ae(data_loader, model):
    device = next(model.parameters()).device

    latents_list = list()
    recons_list = list()
    
    with torch.no_grad():  
        for i, images in enumerate(data_loader):
            # latent_vectors = latent_vectors.to(device)
            images = images.unsqueeze(1).to(device)
            
            reconstructed_images, mu = model(images)

            # Calculate the VAE loss
            latents_list.append(mu.cpu())
            recons_list.append(reconstructed_images.cpu())
            # loss = vae_loss(reconstructed_images, images, mu, logvar)
    
    return latents_list, recons_list


def reshape_outputs(full_latents, full_reconstructed_images, latent_size):
    full_reconstructed_images = np.array(np.concatenate(full_reconstructed_images,axis=0))

    _, channel, height, width = full_reconstructed_images.shape

    full_latents_reshaped = np.array(np.concatenate(full_latents,axis=0)).reshape(-1, latent_size)
    full_reconstructed_images = np.array(full_reconstructed_images)

    # Reshape, combining the first two dimensions A and B into one
    full_reconstructed_images_reshaped = full_reconstructed_images.reshape(-1, height, width)
    return full_latents_reshaped, full_reconstructed_images_reshaped

def get_dataloaders(X_train_torch, y_train_torch, X_test_torch, y_test_torch, batch_size=64):
    """
    Create DataLoader objects for training and testing datasets.
    
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training labels.
        X_test (numpy.ndarray): Testing features.
        y_test (numpy.ndarray): Testing labels.
        batch_size (int, optional): Batch size for the DataLoader. Default is 64.

    Returns:
        train_loader (DataLoader): DataLoader for training set.
        test_loader (DataLoader): DataLoader for test set.
    """

    # Create dataset objects
    train_dataset = TensorDataset(X_train_torch, y_train_torch)
    test_dataset = TensorDataset(X_test_torch, y_test_torch)

    # Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def evaluate_vae_on_loader(model, dataloader, mode="train", device="cuda"):
    """
    Evaluate a VAE model on a dataloader and compute R² scores for both
    neural and stimulus reconstructions.

    Args:
        model (nn.Module): Trained VAE model.
        dataloader (DataLoader): PyTorch DataLoader (train or test).
        mode (str): "train" or "test", just used for logging.
        device (str): "cuda" or "cpu".

    Returns:
        dict: R² scores for neural and stimulus reconstructions.
    """

    model.eval()
    all_recon_neural, all_recon_stimulus = [], []
    all_neural_data, all_stimulus_data = [], []

    with torch.no_grad():
        for neural_batch, stimulus_batch in dataloader:
            neural_batch = neural_batch.to(device)
            stimulus_batch = stimulus_batch.to(device)

            # Forward pass through VAE
            recon_neural, recon_stimulus, _, _, _, _ = model(neural_batch)

            # Accumulate results
            all_recon_neural.append(recon_neural.cpu())
            all_neural_data.append(neural_batch.cpu())

            all_recon_stimulus.append(recon_stimulus.cpu())
            all_stimulus_data.append(stimulus_batch.cpu())

    # Stack all results
    all_neural_data = np.vstack(all_neural_data)
    all_recon_neural = np.vstack(all_recon_neural)
    all_stimulus_data = np.vstack(all_stimulus_data)
    all_recon_stimulus = np.vstack(all_recon_stimulus)

    # Compute R² scores
    neural_r2 = r2_score(all_neural_data, all_recon_neural)
    stimulus_r2 = r2_score(all_stimulus_data[:,:-1], all_recon_stimulus)


    # RMSE scores
    neural_rmse = np.sqrt(mean_squared_error(all_neural_data, all_recon_neural))
    stimulus_rmse = np.sqrt(mean_squared_error(all_stimulus_data[:, :-1], all_recon_stimulus))


    # Print
    print(f"R² Score [{mode}] - Neural: {neural_r2:.4f} | Stimulus: {stimulus_r2:.4f}")
    print(f"RMSE Score [{mode}] - Neural: {neural_rmse:.4f} | Stimulus: {stimulus_rmse:.4f}")

    return neural_r2, stimulus_r2, neural_rmse, stimulus_rmse


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with customizable parameters.")
    
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--latent_size", type=int, default=16, help="Latent vector size.")
    parser.add_argument("--group_rank", type=int, default=1, help="Group rank size.")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer.")
    parser.add_argument("--kl_weight", type=float, default=0.002, help="KL divergence weight.")
    parser.add_argument("--guidance_weight", type=float, default=0.0, help="Guidance weight for the loss function.")
    parser.add_argument("--tc_weight", type=float, default=0.0, help="TC weight.")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed for reproducibility.")

    return parser.parse_args()


