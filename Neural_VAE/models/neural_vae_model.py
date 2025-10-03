import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_size=16):
        super(Encoder, self).__init__()
        # Fully connected layers for encoding
        self.fc1 = nn.Linear(input_dim, 64)
        # self.fc2 = nn.Linear(64, 32)
        self.fc_mu = nn.Linear(64, latent_size)       # Mean of latent distribution
        self.fc_logvar = nn.Linear(64, latent_size)  # Log variance of latent distribution

    def forward(self, x):
        # Pass through the network with ReLU activations
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # Output mean and log variance
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, input_dim, label_dim, latent_size=16):
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.label_dim = label_dim
        self.hidden_dim_1 = 32
        self.group_rank = 6
        # Fully connected layers for decoding
        # self.fc1 = nn.Linear(self.latent_size, 64)
        # self.fc2 = nn.Linear(32, 64)
        self.fc_x = nn.Linear(self.latent_size, input_dim)
        self.fc_category = nn.Sequential(
            nn.Linear(self.group_rank, 8),
            # nn.ReLU(),
            # nn.Linear(self.hidden_dim_1, self.hidden_dim_1),
            # nn.ReLU(),
            # nn.Linear(self.hidden_dim_1, self.label_dim)
        )
        # il is short for the image latents
        self.il_pointwise = nn.Parameter(torch.Tensor(self.label_dim-1))
        init.normal_(self.il_pointwise, mean=0.0, std=0.01)


    def forward(self, z):
        # Pass through the network with ReLU activations

        # x = F.relu(self.fc1(z))
        # x = F.relu(self.fc2(x))
        # Use sigmoid for output activation if input data is normalized to [0, 1]
        # recon_x = (self.fc3(z))
        recon_x = self.fc_x(z)

        # Predicting the labels
        recon_y_pose = z[:, :self.label_dim - 1].mul(self.il_pointwise)
        recon_y_category = self.fc_category(z[:, 6:12])
        return recon_x, recon_y_pose, recon_y_category


class NeuralVAE(nn.Module):
    def __init__(self, input_dim, latent_size=24, group_rank = 1, label_size = 5, guidance_weight=0.2, kl_weight=0.002, tc_weight=4e-7):
        super(NeuralVAE, self).__init__()
        self.latent_size = latent_size
        self.label_size = label_size
        self.group_rank = group_rank
        self.kl_weight = kl_weight
        self.tc_weight = tc_weight
        self.guidance_weight = guidance_weight
        self.group_rank = self.group_rank
        self.n_groups = self.latent_size // self.group_rank
        self.category_criterion = nn.CrossEntropyLoss()
        self.encoder = Encoder(input_dim=input_dim, latent_size=latent_size)
        self.decoder = Decoder(input_dim=input_dim, label_dim=label_size, latent_size=latent_size)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample z ~ N(mu, sigma^2)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode the input to get mu and logvar
        mu, logvar = self.encoder(x)
        # Reparameterization trick to sample latent vector
        z = self.reparameterize(mu, logvar)
        # Decode the latent vector to reconstruct the input
        recon_x, y_hat_pose, y_hat_category = self.decoder(z)
        return recon_x, y_hat_pose, y_hat_category, mu, z, logvar

    def aggregated_posterior(
            self, z_pred_mean: torch.Tensor, z: torch.Tensor, z_pred_log_std: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Aggregated posterior jointly and dimension-wise.
            Parameters
            ----------
            z_pred_mean : torch.Tensor of shape (batch_size, n_components)
                The predicted mean of the latent variable.
            z : torch.Tensor of shape (batch_size, n_components)
                The sampled latent variable.
            z_pred_log_std : torch.Tensor of shape (batch_size, n_components)
                The predicted log standard deviation of the latent variable.
            Returns
            -------
            ln_q_z : torch.Tensor of shape (batch_size,)
                The joint aggregated posterior.
            ln_prod_q_zi : torch.Tensor of shape (batch_size,)
                The dimension-wise aggregated posterior.
            """
            batch_size, n_components = z.shape
            # if self.n_total_samples is None:
            n_total_samples = batch_size
            # else:
                # n_total_samples = self.n_total_samples
            mat_ln_q_z = -F.gaussian_nll_loss(
                z_pred_mean.view((1, batch_size, self.n_groups, self.group_rank)),
                z.view((batch_size, 1, self.n_groups, self.group_rank)),
                (z_pred_log_std.exp() ** 2).view(
                    (1, batch_size, self.n_groups, self.group_rank)
                ),
                full=True,
                reduction="none",
            )  # (n_monte_carlo = batch_size, batch_size, n_groups, group_rank)

            # print(f'mat_ln_q_z shape: {mat_ln_q_z.shape}')

            reweights = (
                torch.ones(batch_size, batch_size, device=z.device)
                / (batch_size - 1)
                * (n_total_samples - 1)
            )
            reweights[torch.arange(batch_size), torch.arange(batch_size)] = 1
            reweights = reweights.log()

            # print(f'reweights shape: {reweights.shape}')

            ln_q_z = torch.logsumexp(
                mat_ln_q_z.sum(dim=(2, 3)) + reweights, dim=1
            ) - np.log(n_total_samples)

            # print(f'ln_q_z shape: {ln_q_z.shape}')

            ln_prod_q_zi = (
                torch.logsumexp(mat_ln_q_z.sum(dim=3) + reweights[:, :, None], dim=1)
                - np.log(n_total_samples)
            ).sum(dim=1)

            # print(f'ln_prod_q_zi shape: {ln_prod_q_zi.shape}')


            # mat_ln_q_z shape: torch.Size([64, 64, 16, 1])
            # reweights shape: torch.Size([64, 64])
            # ln_q_z shape: torch.Size([64])
            # ln_prod_q_zi shape: torch.Size([64])

            return ln_q_z, ln_prod_q_zi

    def partial_correlation(
        self, z_pred_mean: torch.Tensor, z: torch.Tensor, z_pred_log_std: torch.Tensor
    ) -> torch.Tensor:
        ln_q_z, ln_prod_q_zi = self.aggregated_posterior(z_pred_mean, z, z_pred_log_std)
        return (ln_q_z - ln_prod_q_zi).mean()

    def ss_vae_loss(self, recon_x, x, mu, logvar):
        """VAE loss function: reconstruction loss + KL divergence."""
        # Reconstruction loss (Mean Squared Error or Binary Cross-Entropy)
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')

        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        # Weighted total loss
        total_loss = recon_loss + self.kl_weight * kl_loss
        return total_loss, recon_loss, kl_loss

    def guide_vae_loss(self, recon_x, x, y_hat_pose, y_hat_category, y, z, mu, logvar):
        """VAE loss function: reconstruction loss + KL divergence."""
        # Reconstruction loss (Mean Squared Error or Binary Cross-Entropy)
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')

        label_pose_loss = F.mse_loss(y_hat_pose, y[:,:6], reduction='mean')
        # print("min:", y[:,6].min().item(), "max:", y[:,6].max().item())
        # print("y_hat_category shape:", y_hat_category.shape)
        # target = y[:,6].to(torch.long)
        # logits = y_hat_category
        # n_classes = logits.shape[1]

        # assert (target >= 0).all(), "Negative labels!"
        # assert (target < n_classes).all(), f"Label out of range! Got max {target.max()}, but only {n_classes} classes."

        label_category_loss = self.category_criterion(y_hat_category, y[:,6].to(torch.long))
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        tc_loss = self.partial_correlation(mu, z, logvar)

        # Weighted total loss
        total_loss = recon_loss + self.guidance_weight * (label_pose_loss + 0.1 * label_category_loss) + self.kl_weight * kl_loss + self.tc_weight * tc_loss
        return total_loss, recon_loss, self.guidance_weight * (label_pose_loss + 0.1 * label_category_loss), self.kl_weight * kl_loss, self.tc_weight * tc_loss