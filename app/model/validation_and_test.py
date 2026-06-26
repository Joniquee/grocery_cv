import torch
import torchmetrics
from app.data.data_preparation import transform_raw_data
from app.model.model import ImageClassifier, device

print('now it is test')



model = ImageClassifier(10,5,3).to(device)
model.load_state_dict(torch.load('best_model_epoch_4'))

dataloaders = transform_raw_data('./archive/test', './archive/train', './archive/validation')

acc_1 = torchmetrics.Accuracy('multiclass', num_classes=36).to(device)

@torch.no_grad()
def test_model(model, dataloader, metric):
    model.eval()
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        preds = model(X_batch)
        metric.update(preds, y_batch)

    return metric.compute()


acc = test_model(model, dataloaders['test'][0], acc_1)
print(acc)