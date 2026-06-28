import torch, torch.nn as nn
import torchvision
from app.data.data_preparation import transform_raw_data
from tqdm import tqdm
import torchmetrics
from environs import Env

env = Env()
env._load_dotenv('/Users/joniq/Documents/grocery_cv/.env')
device = env('DEVICE')

model = torchvision.models.resnet18(weights = torchvision.models.ResNet18_Weights.DEFAULT)

for params in model.parameters():
     params.requires_grad = False

for params in model.layer4:
     params.requires_grad = True

model.fc = nn.Linear(in_features = 512, out_features = 36)
model.to(device)

dataloader = transform_raw_data('archive/train', 'archive/validation')
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss = nn.CrossEntropyLoss().to(device)
n_epochs = 50
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,
    factor=0.1,
    threshold=0.0015,
    patience=5
)

metric = torchmetrics.Accuracy(task='multiclass', num_classes=36).to(device)
history = {'epochs': [], 'lr': []}
def training(model, optimizer, n_epochs, dataloader_train, dataloader_val, loss, scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau, metric, history):
        model.train()
        best_val_loss = float('inf')
        for epoch in (range(n_epochs)):
            history['epochs'].append(epoch+1)
            total_loss_train = 0
            model.train()
            for X_batch, y_batch in tqdm(dataloader_train):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                cur_loss = loss(preds, y_batch)
                cur_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss_train += cur_loss.item()
            model.eval()
            total_loss_val = 0
            for X_val, y_val in tqdm(dataloader_val):
                X_val, y_val = X_val.to(device), y_val.to(device)
                preds = model(X_val)
                metric.update(preds, y_val)
                cur_loss = loss(preds, y_val)
                total_loss_val += cur_loss.item()
            history['lr'].append(scheduler.get_last_lr())
            print(f'Current LR - {scheduler.get_last_lr()}')
            scheduler.step(total_loss_val / len(dataloader_val))
            print(f'Epoch - {epoch+1}, Loss - {total_loss_train / len(dataloader_train)}, Val_Loss - {total_loss_val / len(dataloader_val)}, ACC - {metric.compute()}')
            if total_loss_val < best_val_loss:
                best_val_loss = total_loss_val
                torch.save(model.state_dict(), f'best_model_now')
        
        torch.save(model.state_dict(), 'model')



try:
  training(model, optimizer=optimizer, n_epochs=n_epochs, dataloader_train=dataloader['train'][0], dataloader_val=dataloader['validation'][0], loss=loss, scheduler=lr_scheduler, metric=metric, history=history)
except KeyboardInterrupt:
  torch.save(model.state_dict(), 'model_interrupted')
  print('Interrupted')


print(history)