from mvae import MVAE
from datasets import PMUDataset
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.decomposition import PCA
from wavelet import Meig_Zero

def PMU_sim(T):

    f0 = 50.0
    fs = 50.0
    N = int(T*fs)
    t = np.linspace(0,T,N,endpoint=False)

    f = f0 + 0.0 * np.sin(2 * np.pi * 0.5 * t)

    phase = 2 * np.pi * (f0 * t - 0.0 / (2 * np.pi) * np.cos(2 * np.pi * 0.5 * t))
    phase += np.random.rand(N)


    A = 100.0 + 0.0005 * np.sin(2 * np.pi * 5 * t)
    SNR_amplitude = - 40.0
    noise_power = 100**2 * 10**(SNR_amplitude/10)
    noise =  np.sqrt(noise_power)* np.random.randn(N, 3)

    va = (A + noise[:, 0]) * np.exp(phase * 1j)
    vb = (A + noise[:, 1]) * np.exp((phase - 2 * np.pi / 3) * 1j)
    vc = (A + noise[:, 2]) * np.exp((phase + 2 * np.pi / 3) * 1j)


    v_abc = np.stack([va, vb, vc],axis = 1).astype("complex")

    a = np.exp(1j * 2 * np.pi / 3)

    # Matrice de transformation de Fortescue
    A = (1 / 3) * np.array([
        [1, 1, 1],
        [1, a, a ** 2],
        [1, a ** 2, a]
    ])
    # Transformation de Fortescue
    v_fortescue = np.zeros((N, 3), dtype=complex)
    for i in range(N):
        v_fortescue[i, :] = np.dot(A,v_abc[i, :])

    v0, v1, v2 = v_fortescue[:, 0], v_fortescue[:, 1], v_fortescue[:, 2]

    return np.concatenate(([np.abs(v0)], [np.unwrap(np.angle(v0))], [np.abs(v1)], [np.unwrap(np.angle(v1))], [np.abs(v2)], [np.unwrap(np.angle(v2))], [f]))




def main():

    input_dim = 7
    time_dim = 250
    batch_size = 32
    latent_dim = 2




    per = 3600
    PMU_data = PMU_sim(per).T


    PMU_data = torch.from_numpy(PMU_data).view(per//(time_dim//50), time_dim, input_dim)
    taux = 0.8


    for i in range(3):
        plt.subplot(3,1,i+1)
        plt.plot(PMU_data[0, :,i+2], label="Original")
        plt.title(f"Grandeur {i+1}")
        plt.legend()




    scaler = StandardScaler().fit(PMU_data.reshape(-1, input_dim)[:int(taux * per*50), :])
    PMU_data_norm = torch.from_numpy(scaler.transform(PMU_data.reshape(-1, input_dim))).reshape(per//(time_dim//50), time_dim, input_dim)


    dataset = PMUDataset(PMU_data_norm)


    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])



    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle= False)

    all_data = []
    for batch_data in dataset:
        all_data.append(batch_data.numpy())
    all_data = np.concatenate(all_data, axis=0)

    pca = PCA(n_components=7)
    principal_components = pca.fit_transform(all_data)
    print(pca.singular_values_)


    std_dev_a = Meig_Zero(principal_components[:,0].T)
    std_dev_b = Meig_Zero(principal_components[:,1].T)
    eps = torch.from_numpy(0.8 * std_dev_a + 0.2 * std_dev_b).to(torch.float32)


    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")


    device = torch.device("cude" if torch.cuda.is_available() else "cpu")

    model = MVAE(
        input_dim = input_dim,
        time_dim = time_dim,
        batch_size = batch_size,
        latent_dim = latent_dim,
        eps=eps
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    n_epochs = 50

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")

        for batch in progress_bar:

            batch = batch.to(device)
            optimizer.zero_grad()
            x_decoded, z_mean, z_log_var = model(batch)
            loss = model.loss_function(batch, x_decoded, z_mean, z_log_var)['loss']
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                x_decoded, z_mean, z_log_var = model(batch)
                loss = model.loss_function(batch, x_decoded, z_mean, z_log_var)['loss']
                test_loss += loss.item()

        print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, Test Loss: {test_loss/len(test_loader):.4f}")


    test_window = test_dataset[6].unsqueeze(0).to(device)
    x_decoded, _, _ = model(test_window)



    test_window_denorm = scaler.inverse_transform(test_window[0].cpu().numpy())
    x_decoded_denorm = scaler.inverse_transform(x_decoded[0].detach().cpu().numpy())



    plt.figure(figsize=(12,6))

    for i in range(3):
        plt.subplot(3,1,i+1)
        plt.plot(test_window_denorm[:,i+2], label="Original")
        plt.plot(x_decoded_denorm[:,i+2], label="Reconstructed", linestyle="--")
        plt.title(f"Grandeur {i+1}")
        plt.legend()

    plt.tight_layout()
    plt.show()












if __name__ == "__main__":
    main()