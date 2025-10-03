from models.neural_vae_model import NeuralVAE
from utils_scripts.utils_torch import set_random_seed, zscore_2d, PairwiseDataset, rescale_to_01, rescale_to_minus1_1, zscore_by_column, warmup_then_decay_lr
from utils_scripts.utils_torch import evaluate_vae_on_loader, parse_args
from utils_scripts.disentangle_metrics import factorvae_score, compute_unsupervised_sap, compute_mig

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import wandb


args = parse_args()

print("Training Configuration:")

config = vars(args)
print(config)

print(f"\n--- Device Check ---")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"[GPU {i}] {torch.cuda.get_device_name(i)}")
else:
    print("No GPU available. Using CPU.")


neural_data_name = 'sorted_it_avg_data'

stimulus_file = 'datasets/stimulus_pose_features.npy'
stimulus_category_file = 'datasets/stimulus_category_ids.npy'
neural_data_file = f"datasets/{neural_data_name}.npy"

stimulus_data_pos = np.load(stimulus_file).astype(np.float32)
stimulus_data_category = np.expand_dims(np.load(stimulus_category_file).astype(np.float32), axis=1)
stimulus_data = np.concatenate((stimulus_data_pos, stimulus_data_category), axis=1)
neural_data = np.load(neural_data_file).astype(np.float32)[:,:58]
neural_dim, stimulus_dim = neural_data.shape[-1], stimulus_data.shape[-1]

print(f"neural_data shape: {neural_data.shape}")
print(f"stimulus_data shape: {stimulus_data.shape}")


def train_vae(model, train_dataloader, optimizer, num_epochs=200, device='cpu', record_training = False):
    model = model.to(device)
    pre_test_loss = 1e8

    for epoch in range(num_epochs):
        model.train()
        training_epoch_loss = 0.0
        recon_epoch_loss, label_epoch_loss = 0.0, 0.0
        kl_epoch_loss, tc_epoch_loss = 0.0, 0.0

        # scheduler = LambdaLR(optimizer, lr_lambda=lambda step: warmup_then_decay_lr(step, warmup_steps, total_steps))
        scheduler = optimizer
    
        for i, (neural_batch, stimulus_batch) in enumerate(train_dataloader):
            # latent_vectors = latent_vectors.to(device)
            neural_batch = neural_batch.to(device)
            stimulus_batch = stimulus_batch.to(device)
            
            recon_x, recon_y_pose, recon_y_category, mu, z, logvar = model(neural_batch)

            # if i == 0:
                # print(f"reconstructed_images testue: {reconstructed_images[0]}")
            # Calculate the VAE loss
            loss, recon_loss, label_loss, kl_loss, tc_loss = model.guide_vae_loss(recon_x, neural_batch, recon_y_pose, recon_y_category, stimulus_batch, z, mu, logvar)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            training_epoch_loss += loss.item()
            recon_epoch_loss += recon_loss.item()
            label_epoch_loss += label_loss.item()
            kl_epoch_loss += kl_loss.item()
            tc_epoch_loss += tc_loss.item()

        scheduler.step()  
        
        testing_epoch_loss, testing_label_epoch_loss = 0.0, 0.0
        
        with torch.no_grad():  
            for (neural_batch, stimulus_batch) in test_dataloader:
                neural_batch = neural_batch.to(device)
                stimulus_batch = stimulus_batch.to(device)
                
                recon_x, recon_y_pose, recon_y_category, mu, z, logvar = model(neural_batch)

                # Calculate the VAE loss
                loss, recon_loss, label_loss, _, _ = model.guide_vae_loss(recon_x, neural_batch, recon_y_pose, recon_y_category, stimulus_batch, z, mu, logvar)
                
                testing_epoch_loss += loss.item()

                testing_label_epoch_loss += label_loss.item()


        if epoch % 5 ==0:
            if pre_test_loss > testing_epoch_loss:
                pre_test_loss = testing_epoch_loss
                torch.save(model.state_dict(), f'model_checkpoints/{exp_name}.pth')

            if record_training:
                wandb.log({
                    # "epoch": epoch,
                    "train_loss": training_epoch_loss,
                    "test_loss": testing_epoch_loss,
                    "test_label_loss": testing_label_epoch_loss,
                    "recon_epoch_loss": recon_epoch_loss,
                    "label_epoch_loss": label_epoch_loss,
                    "kl_epoch_loss": kl_epoch_loss,
                    "tc_epoch_loss": tc_epoch_loss
                }, step=epoch)

        if epoch == 0 or (epoch + 1) % 25 == 0:
            # print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
            # print(f"Epoch [{epoch+1}/{num_epochs}], Recon Loss: {t_recon_loss:.4f}, Test Loss: {test_loss:.4f}")
            print("Test Evaluation at epoch {}:".format(epoch + 1))
            neural_r2, stimulus_r2, neural_rmse, stimulus_rmse = evaluate_vae_on_loader(model, test_dataloader, mode="test", device=device)
            if record_training:
                wandb.log({
                        "neural_r2": neural_r2,
                        "stimulus_r2": stimulus_r2,
                        "neural_rmse": neural_rmse,
                        "stimulus_rmse": stimulus_rmse
                    }, step=epoch)

    print(f"Training Finished with seed {seed}")
    return model


# Training Hyper-Parameter
batch_size = config["batch_size"]
latent_size = config["latent_size"]
group_rank = config["group_rank"]
num_epochs = config["num_epochs"]
kl_weight = config["kl_weight"]
tc_weight = config["tc_weight"]
guidance_weight = config["guidance_weight"]
learning_rate = config["learning_rate"]
random_seeds = np.arange(2024, 2025)
# random_seeds = [2024]

neural_data_standardized = zscore_by_column(neural_data)
stimulus_data_standardized = (stimulus_data)

neural_train, neural_test, stimulus_train, stimulus_test = train_test_split(
    neural_data_standardized, stimulus_data_standardized, test_size=0.2, random_state=2024
)

train_dataset = PairwiseDataset(neural_train, stimulus_train, transform=zscore_by_column, apply_transform=False)
test_dataset = PairwiseDataset(neural_test, stimulus_test, transform=zscore_by_column, apply_transform=False)
neural_dataset = PairwiseDataset(neural_data_standardized, stimulus_data_standardized, transform=zscore_by_column, apply_transform=False)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
neural_dataloader = DataLoader(neural_dataset, batch_size=batch_size, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_r2_neural_results, test_r2_neural_results = list(), list()
train_r2_stimulus_results, test_r2_stimulus_results = list(), list()

for id, seed in enumerate(random_seeds):
    # print(f"\nTraining with seed: {type(seed)}\n")
    
    set_random_seed(int(seed))

    exp_name = f'{neural_data_name}_epoch_{num_epochs}_b_{batch_size}_d_{latent_size}_kl_{kl_weight}_gr_{group_rank}_sd_{seed}_trimmed'


    model = NeuralVAE(input_dim=neural_dim, latent_size=latent_size, group_rank=group_rank, label_size = stimulus_dim, guidance_weight=guidance_weight, kl_weight=kl_weight, tc_weight=tc_weight).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    if id == 0:
        wandb.init(project="neural_vae_training", name=exp_name, config={
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "latent_size": latent_size,
        })
        record_training = True
    else:
        record_training = False

    model = train_vae(model, train_dataloader, optimizer, num_epochs=num_epochs, device=device, record_training=record_training)

    wandb.finish()

    train_metrics = evaluate_vae_on_loader(model, train_dataloader, mode="train", device=device)
    test_metrics = evaluate_vae_on_loader(model, test_dataloader, mode="test", device=device)

    train_r2_neural_results.append(train_metrics[0])
    test_r2_neural_results.append(test_metrics[0])

    train_r2_stimulus_results.append(train_metrics[1])
    test_r2_stimulus_results.append(test_metrics[1])



full_latents, full_neural_recons, full_pose_recons, full_category_recons = list(), list(), list(), list()

with torch.no_grad():  
    for i, (neural_batch, stimulus_batch) in enumerate(neural_dataloader):
        # latent_vectors = latent_vectors.to(device)
        neural_batch = neural_batch.to(device)
        
        recon_neural_batch, recon_pose_batch, recon_category_batch, mu, z, logvar = model(neural_batch)

        # Calculate the VAE loss
        full_latents.append(mu.cpu())
        full_neural_recons.append(recon_neural_batch.cpu())
        full_pose_recons.append(recon_pose_batch.cpu())
        full_category_recons.append(recon_category_batch.cpu())
        # loss = vae_loss(reconstructed_images, images, mu, logvar)

full_latents = np.vstack(full_latents)
full_neural_recons = np.vstack(full_neural_recons)
full_pose_recons = np.vstack(full_pose_recons)
full_category_recons = np.vstack(full_category_recons)


