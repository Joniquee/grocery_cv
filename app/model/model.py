import torch.nn as nn
import torch
from app.data.data_preparation import transform_raw_data

class ImageClassifier(nn.Module):
    def __init__(self, n_out, n_out2, n_out3):
        super().__init__()
        self.first_layers_channels = 64
        self.model = nn.Sequential(
           #Первый слой -------------------------- 
            nn.Dropout2d(p=0.2),
            nn.Conv2d(in_channels=3, out_channels=n_out, kernel_size=5),
            nn.BatchNorm2d(n_out),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(in_channels=n_out, out_channels=n_out2, kernel_size=3),
            nn.BatchNorm2d(n_out2),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(in_channels=n_out2, out_channels=n_out3, kernel_size=3),
            nn.BatchNorm2d(n_out3),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.Flatten(),
            nn.Linear(184512, 36)
        )
    def forward(self, X):
        return self.model(X)

if torch.backends.mps.is_available():
    device=torch.device('mps')
else:
    device = torch.device('cpu')

print(device)


if __name__ == '__main__':
    dataloaders = transform_raw_data('./archive/test', './archive/train', './archive/validation')
        




    model = ImageClassifier(10, 5, 3).to(device)


    def training(model, optimizer, n_epochs, dataloader_train, dataloader_val, loss, scheduler):
        model.train()
        best_val_loss = float('inf')
        for epoch in range(n_epochs):
            total_loss_train = 0
            model.train()
            for X_batch, y_batch in dataloader_train:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                cur_loss = loss(preds, y_batch)
                cur_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss_train += cur_loss.item()
            model.eval()
            total_loss_val = 0
            for X_val, y_val in dataloader_val:
                X_val, y_val = X_val.to(device), y_val.to(device)
                preds = model(X_val)
                cur_loss = loss(preds, y_val)
                total_loss_val += cur_loss.item()
            if total_loss_val < best_val_loss:
                best_val_loss = total_loss_val
                torch.save(model.state_dict(), f'my_best_model_epoch_{epoch+1}')
            lr_scheduler.step()
            print(f'Epoch - {epoch+1}, Loss - {total_loss_train / len(dataloader_train)}, Val_Loss - {total_loss_val / len(dataloader_val)}')
        
        torch.save(model.state_dict(), 'model_my')


    loss = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)

    training(model, optimizer, 10, dataloaders['train'][0], dataloaders['validation'][0], loss, lr_scheduler)