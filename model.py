import torch
import torchvision
import wandb
import time

from torchmetrics import Accuracy
from tqdm import tqdm
from data import data_loaders

train_dataloader, test_dataloader = data_loaders('Floreview_split', batch_size=24, num_workers=4)

print(f"Total training batches: {len(train_dataloader)}")

class Model2(torch.nn.Module):
    def _init_(self, outputs = 39, pretrained = True, freeze = False):
      super()._init_()

      efficientnet =  torchvision.models.efficientnet_v2_m(pretrained = pretrained)
      self.efficientnet = torch.nn.Sequential(*list(efficientnet.children())[:-1])

      self.fc = torch.nn.Linear(1280, 512)
      self.fc1 = torch.nn.Linear(512, 256)
      self.fc2 = torch.nn.Linear(256, outputs)

    def forward(self, batch):
      batch = self.efficientnet(batch)
      batch = batch.view(batch.shape[0], -1)
      batch = self.fc(batch)
      batch = self.fc1(batch)
      batch = self.fc2(batch)
      return batch



class Model(torch.nn.Module):
  # se congelan las capas convolucionales y los pesos pre entrenados se mantienen
    def _init_(self, outputs = 39, pretrained = True, freeze = False):
      super()._init_()

      resnet50 = torchvision.models.resnet50(pretrained = pretrained)
      # se descargó resnet50 pre-entrenado
      self.resnet50 = torch.nn.Sequential(*list(resnet50.children())[:-1])

      if freeze:
        for param in self.resnet50.parameters():
          param.requires_grad=False
      self.fc = torch.nn.Linear(2048, 512)
      self.fc1 = torch.nn.Linear(512, 256)
      self.fc2 = torch.nn.Linear(256, outputs)

    def forward(self, batch):
      batch = self.resnet50(batch)
      batch = batch.view(batch.shape[0], -1)
      batch = self.fc(batch)
      batch = self.fc1(batch)
      batch = self.fc2(batch)
      return batch

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    total_norm = 0.0

    for batch_idx, (x, y) in enumerate(tqdm(dataloader, desc=f"Train Epoch {epoch}")):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()

        y_hat = model(x)
        #print("Predictions: ", y_hat)
        #print("Shape of predictions: ", y_hat.shape)
        #print("Groundtruth: ", y)
        #print("Shape of groundtruth: ", y.shape)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()

        # gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        running_loss += loss.item()
        wandb.log({
            "train/batch_loss": loss.item(),
            "train/grad_norm": total_norm,
            "train/step": epoch * len(dataloader) + batch_idx
        })

    avg_loss = running_loss / len(dataloader)
    return avg_loss

def validate_one_epoch(model, dataloader, criterion, device, epoch):
    accuracy = Accuracy(num_classes=39, average='micro', task='multiclass').to(device)
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(tqdm(dataloader)):
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            accuracy.update(y_hat, y)  # Proper accumulation
            running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    avg_acc = accuracy.compute()  # Correct final accuracy

    wandb.log({
        "val/loss": avg_loss,
        "val/epoch": epoch,
        "val/accuracy": avg_acc
    })
    accuracy.reset()  # Important for next epoch
    return avg_loss

def fit(model, trainloader, validloader, epochs, lr, num_classes=39):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    # metrics
    accuracy_micro = Accuracy(num_classes=num_classes, average='micro', task='multiclass').to(device)
    accuracy_per_class = Accuracy(num_classes=num_classes, average=None, task='multiclass').to(device)

    # best-model tracking
    best_val_acc = 0.0
    best_model_path = "best_model.pth"
    patience, no_improve = 10, 0
    best_val_loss = float("inf")

    wandb.login(key="8b67cfcbdea25a891a0c70382c955f441f82941b")

    wandb.init(
    project="camera-identification",
    name="finetuning_resnet50",
    config={
        "dataset": "FloreView",
        "architecture": "Resnet50",
        "batch_size": 64,
        "epochs": 200,
        "optimizer": "Adam",
        "learning_rate": 3e-3
    })

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, trainloader, criterion, optimizer, device, epoch)
        wandb.log({"train/loss": train_loss, "train/epoch": epoch})

        val_loss = validate_one_epoch(model, validloader, criterion, device, epoch)

        # compute validation accuracy on last batch
        #x_val, y_val = next(iter(validloader))
        #x_val, y_val = x_val.to(device), y_val.to(device)
        #y_hat = model(x_val)
        #acc = accuracy_micro(y_hat, y_val)
        #wandb.log({"val/accuracy_micro": acc, "val/epoch": epoch})

        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), best_model_path)  # Save best model
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered!")
                break
        """
        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(model.state_dict(), best_model_path)
            print(f"→ New best model saved (acc {acc:.4f})")
        if no_improve >= patience:
            print("Convergence reached. Stopping early.")
            break
        """
    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.1f}s")
    wandb.finish()

if _name_ == '_main_':
    model = Model2()
    fit(model, train_dataloader, test_dataloader, epochs=200, lr=1e-3)
    print("Training complete.")
    print("Model saved as 'best_model.pth'.")