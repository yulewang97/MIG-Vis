## MIG-Vis: Uncovering Semantic Selectivity of Latent Groups in Higher Visual Cortex with Mutual Information-Guided Diffusion

### News
___

* [2025/12] The abstract version of **[MIG-Vis](https://arxiv.org/abs/2510.02182)** has been accepted as a poster presentation at **[COSYNE 2026](https://www.cosyne.org/)** in Lisbon, Portugal!

Official code for **[MIG-Vis](https://arxiv.org/abs/2510.02182)** (ICLR 2026 submission): a framework to visualize and interpret neural latent groups via mutual-information guided diffusion.


### Overview of the Approach
___

MIG‑Vis is a method to discover and visualize semantically meaningful latent subspaces encoded in neural activity from higher visual cortex recordings. It combines a group‑wise disentangled VAE for inferring structured neural latent subspaces, and a mutual information (MI)‑guided diffusion image synthesis procedure to visualize what each latent group encodes.


The full code and installation & experiment instructions will be available soon.


#### Repository Structure

```
MIG‑Vis/
├── neural_vae/                  # VAE module for inferring neural latent groups
├── mig_diffusion/               # MI‑guided diffusion synthesis procedures
├── diffusion_model/             # Image diffusion function modules
├── utils_scripts/               # Utility scripts e.g., data loading, training helpers
```