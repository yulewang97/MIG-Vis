import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer

def create_latent_pairs(latents, num_samples=10000):
    """
    Create latent pairs (z1, z2) that differ by exactly one latent dimension.
    Return:
        - z_diff_pairs: Tensor of shape (num_samples, latent_dim)
        - labels: which dimension was changed
    """
    latent_dim = latents.shape[1]
    z_diff_pairs = []
    labels = []

    for _ in range(num_samples):
        idx1, idx2 = np.random.choice(len(latents), 2, replace=False)
        z1 = latents[idx1].copy()
        z2 = z1.copy()

        # randomly choose a dimension to change
        dim = np.random.randint(latent_dim)
        z2[dim] = latents[idx2][dim]  # replace only that dimension

        z_diff_pairs.append(np.abs(z1 - z2))
        labels.append(dim)

    return np.array(z_diff_pairs), np.array(labels)


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.net(x)


def factorvae_score(latents, num_samples=10000, num_epochs=10, batch_size=256):
    z_diff_pairs, labels = create_latent_pairs(latents, num_samples)

    dataset = TensorDataset(
        torch.tensor(z_diff_pairs, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    classifier = SimpleClassifier(input_dim=latents.shape[1])
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Training
    classifier.train()
    for epoch in range(num_epochs):
        for x_batch, y_batch in loader:
            logits = classifier(x_batch)
            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluation
    classifier.eval()
    with torch.no_grad():
        logits = classifier(torch.tensor(z_diff_pairs, dtype=torch.float32))
        preds = torch.argmax(logits, dim=1).numpy()
        acc = accuracy_score(labels, preds)

    return acc


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
import numpy as np


def get_pseudo_factors(stimulus_data, num_clusters=10):
    # PCA 降维 + KMeans 聚类
    pca = PCA(n_components=min(10, stimulus_data.shape[1]))
    reduced = pca.fit_transform(stimulus_data)
    clusters = KMeans(n_clusters=num_clusters, random_state=0).fit_predict(reduced)
    return clusters


def compute_unsupervised_sap(latents, stimulus_data, num_clusters=10):
    pseudo_factors = get_pseudo_factors(stimulus_data, num_clusters=num_clusters)

    latent_dim = latents.shape[1]
    sap_diffs = []

    for factor in np.unique(pseudo_factors):
        # Create binary label for this pseudo-class
        binary_labels = (pseudo_factors == factor).astype(int)

        scores = []
        for i in range(latent_dim):
            z_i = latents[:, i].reshape(-1, 1)

            clf = RidgeClassifier()
            clf.fit(z_i, binary_labels)
            preds = clf.predict(z_i)
            acc = accuracy_score(binary_labels, preds)
            scores.append(acc)

        sorted_scores = np.sort(scores)[::-1]  # descending
        if len(sorted_scores) >= 2:
            sap_diffs.append(sorted_scores[0] - sorted_scores[1])
        else:
            sap_diffs.append(0.0)

    return np.mean(sap_diffs)



def discretize_latents(latents, n_bins=20):
    """对 continuous latent 进行分箱离散化"""
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    return est.fit_transform(latents)

def compute_mutual_info(x, y):
    """计算互信息 MI(x, y)"""
    return mutual_info_score(x, y)

def compute_entropy(x):
    """计算熵 H(x)"""
    return mutual_info_score(x, x)

def compute_mig(latents, factors, n_bins=20):
    """
    latents: [N, D] - continuous or discrete latent variables
    factors: [N, K] - discrete ground-truth factors
    """
    # 离散化 latents
    discretized_latents = discretize_latents(latents, n_bins=n_bins)
    
    D = discretized_latents.shape[1]
    K = factors.shape[1]

    mijs = np.zeros((D, K))
    entropies = np.zeros(K)

    for k in range(K):
        y = factors[:, k]
        entropies[k] = compute_entropy(y)

        for d in range(D):
            x = discretized_latents[:, d]
            mijs[d, k] = compute_mutual_info(x, y)

    # 对每个 factor，计算前两大的 MI 差值除以 H(y_k)
    sorted_mijs = np.sort(mijs, axis=0)[::-1]
    migs = (sorted_mijs[0] - sorted_mijs[1]) / (entropies + 1e-10)  # 避免除0

    return np.mean(migs)
