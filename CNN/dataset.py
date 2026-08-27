import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import numpy as np
import glob

class PMUDataset(Dataset):
    def __init__(self, X, y, mean = None, std = None):
        self.X = X.float()
        self.y = y.float()

        if mean is None or std is None:

            self.mean = self.X.mean(dim=1, keepdim=True)
            self.std = self.X.std(dim=1, keepdim=True) + 1e-5

        else:
            self.mean = mean
            self.std = std

        self.X = (self.X - self.mean) / self.std


    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def telecharge_dataset(dossier, taille_batch, taux_test = 0.2, taux_val = 0.1, shuffle=True):

    doss_anomal = os.path.join(dossier, 'anomal')
    doss_normal = os.path.join(dossier, 'normal')

    fich_anomal = sorted(glob.glob(os.path.join(doss_anomal, '*.csv')))
    fich_normal = sorted(glob.glob(os.path.join(doss_normal, '*.csv')))

    X_liste = []
    y_liste = []

    for fpath in fich_normal:

        df = pd.read_csv(fpath)
        arr = df.values[:,1:]

        if arr.shape != (200, 8):
            raise ValueError(f"shape inattendue dans {fpath}: {arr.shape}")

        X_liste.append(arr)
        y_liste.append(0)

    for fpath in fich_anomal:

        df = pd.read_csv(fpath)
        arr = df.values[:,1:]

        if arr.shape != (200, 8):
            raise ValueError(f"shape inattendue dans {fpath}: {arr.shape}")

        X_liste.append(arr)
        y_liste.append(1)

    X_complet = np.stack(X_liste).astype(np.float32)
    y_complet = np.stack(y_liste).astype(np.int32)

    X_complet = torch.from_numpy(X_complet)
    y_complet = torch.from_numpy(y_complet)

    N = X_complet.shape[0]
    indices = np.arange(N)

    if shuffle:
        np.random.shuffle(indices)

    n_test = int(N * taux_test)
    n_val = int(N * taux_val)
    n_train = N - n_test - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]

    X_train, y_train = X_complet[train_idx], y_complet[train_idx]
    X_val, y_val = X_complet[val_idx], y_complet[val_idx]
    X_test, y_test = X_complet[test_idx], y_complet[test_idx]

    train_dataset = PMUDataset(X_train, y_train)
    val_dataset = PMUDataset(X_val, y_val)
    test_dataset = PMUDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=taille_batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=taille_batch, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=taille_batch, shuffle=False)

    return train_loader, val_loader, test_loader








