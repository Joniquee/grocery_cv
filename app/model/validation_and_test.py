import torch, torch.nn as nn
import torchmetrics
import torchvision
from torchvision.datasets import ImageFolder
from app.data.data_preparation import transform_raw_data
from app.model.model import ImageClassifier
from environs import Env
from tqdm import tqdm

env = Env()
env._load_dotenv('/Users/joniq/Documents/grocery_cv/.env')
device = env('DEVICE')

print('now it is test')

dataloaders = transform_raw_data('archive/test')

acc_1 = torchmetrics.Accuracy('multiclass', num_classes=36).to(device)
model_test = torchvision.models.resnet18(weights = None)
model_test.fc = nn.Linear(in_features = 512, out_features = 36)
model_test.load_state_dict(torch.load('app/model/model_resnet_2.0', map_location=device))
model_test.to(device)

@torch.no_grad()
def test_model(model, dataloader, metric):
    model.eval()
    for X_batch, y_batch in tqdm(dataloader):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        preds = model(X_batch)
        metric.update(preds, y_batch)

    return metric.compute()


acc = test_model(model_test, dataloaders['test'][0], acc_1)
print(acc)