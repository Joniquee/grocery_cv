import torch
#from app.data.data_preparation import transform_single_image
import torchvision
#from torchvision.datasets import ImageFolder
from environs import Env
import json
import os

env = Env()
env.read_env()
device = os.environ.get('DEVICE', 'cpu')

#dataset = ImageFolder('./archive/test')
#class_names = dataset.classes

with open('classes.json', 'r') as f:
    class_names_dict = json.load(f)

class_names = class_names_dict['class']



model = torchvision.models.resnet18(weights=None)
model.fc = torch.nn.Linear(512, 36)
model.load_state_dict(torch.load('app/model/model_resnet_2.0', map_location=device))
model.to(device)

#data = transform_for_predict('./predict_images', batch_size=1)



@torch.no_grad()
def predict(model, data):
    model.eval()
    preds_probas = []
    preds_names = []
    for img in data:
        img = img.to(device)
        preds = model(img)
        preds_probas.append( torch.nn.functional.softmax(preds, dim=1))
        probas_list = list(preds_probas[0].reshape(36))
        probas_list = [(x[0], x[1].item()) for x in sorted(enumerate(probas_list), key=lambda x: x[1].item())]
        top_probas = [probas_list[i] for i in range(len(probas_list)-1, -1,  -1) if probas_list[i][1] >=0.1]
        for pair in top_probas:
            label = pair[0]
            preds_names.append(class_names[label])

        

    return  top_probas, preds_names


#predictions = predict(model, data)
#print(predictions)