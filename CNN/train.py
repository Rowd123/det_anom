import torch
import torch.nn as nn

from archi import Modele
from dataset import telecharge_dataset

from tqdm.auto import tqdm

def train_one_epoch(model, loader, optimizer, critere, device, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    n_samples = 0


    pbar = tqdm(loader, desc=f"Train epoch {epoch+1}/{num_epochs}", leave=False)

    for X_batch, y_batch in pbar:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).view(-1,1).float()

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = critere(logits, y_batch)
        loss.backward()
        optimizer.step()

        taille_batch = X_batch.size(0)
        running_loss += loss.item() * taille_batch
        n_samples += taille_batch

        avg_loss = running_loss / n_samples
        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    return running_loss / n_samples

def evaluate(model, loader, citere, device, epoch, num_epochs, phase="val"):

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"{phase} epoch {epoch+1}/{num_epochs}", leave=False)

    with torch.no_grad():
        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).view(-1,1).float()

            logits = model(X_batch)
            loss = citere(logits, y_batch)

            running_loss += loss.item() * X_batch.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

            avg_loss = running_loss / total
            acc = correct / total
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{acc:.4f}"})

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


def main():

    donnees = "/home/taklarid/Projects/code/CNN/datasets"

    # Hyperparamètres
    taille_batch = 32
    num_epochs = 10
    lr = 1e-3


    # Chargement des données

    train_loader, val_loader, test_loader = telecharge_dataset(donnees, taille_batch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Modele()
    critere = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, critere, device, epoch, num_epochs)
        val_loss, val_acc = evaluate(model, val_loader, critere, device, epoch, num_epochs)


        print(f"[Epoch {epoch+1:02d}/{num_epochs}] "
              f"train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | "
              f"val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")


    # Évaluation finale sur le test
    print("\nÉvaluation sur le jeu de test avec le meilleur modèle :")
    model.load_state_dict(torch.load("best_model.pt"))
    test_loss, test_acc = evaluate(
        model, test_loader, critere, device,
        epoch=num_epochs-1, num_epochs=num_epochs, phase="Test"
    )
    print(f"[Test] loss={test_loss:.4f} | acc={test_acc:.4f}")

if __name__ == "__main__":
    main()


