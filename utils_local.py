import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

import IPython
from IPython.display import display

import copy
import pickle
import time
import random, os

from PIL import Image
from io import BytesIO

from tqdm.autonotebook import tqdm, trange

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,models
from torchmetrics import F1Score
from torchmetrics.classification import MultilabelF1Score
import wandb

import timm 



BATCH = 64
IMG_SIZE = 224
EPOCHS = 5
LR = 0.001

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    
def pil_loader(img):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    img = Image.open(img) 
    return img.convert('RGB')

# class CustomData(Dataset):
#     def __init__(self, byte_img, Labels, Transform):
#         self.byte_img = byte_img
#         self.transform = Transform
#         self.labels = Labels         
        
#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, index):       
#         image = pil_loader(BytesIO(self.byte_img[index]))
#         label = self.labels[index]
#         if self.transform[label] is not None:
#             image = self.transform[label](image)
#         return image, label

# class CustomData(Dataset):
#     def __init__(self, byte_img, Labels, Transform, test = False):
#         self.byte_img = byte_img
#         self.transform = Transform
#         self.labels = Labels
#         self.test = test
        
#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, index):       
#         image = pil_loader(BytesIO(self.byte_img[index]))
#         label = self.labels[index]
#         if not self.test:         
#             return self.transform[label](image), label
#         else:
#             return self.transform(image), label

class CustomData(Dataset):
    def __init__(self, byte_img, Labels, Transform):
        self.byte_img = byte_img
        self.transform = Transform
        self.labels = Labels
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):       
        image = pil_loader(BytesIO(self.byte_img[index]))
        return self.transform(image), self.labels[index]

    
def log_image_table(images, predicted, labels, probs, num_cls):
    "Log a wandb.Table with (img, pred, target, scores)"
    # 🐝 Create a wandb Table to log images, labels and predictions to
    table = wandb.Table(columns=["image", "pred", "target"]+[f"score_{i}" for i in range(num_cls)])
    for img, pred, targ, prob in zip(images.to("cpu"), predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")):
        table.add_data(wandb.Image(img[0].detach().numpy()*255), pred, targ, *prob.detach().numpy())
    wandb.log({"predictions_table":table}, commit=False)

    
def train_model(model, criterion, optimizer, dataloaders,
                device, model_name, d_loader_name,
                wandb, loss_weighted=True,
                num_cls=9, num_epochs=25):
    
    wandb.watch(model, log="all")
#     f1 = F1Score(num_classes=num_cls, average="macro", ).to(device) #change this
    f1 = MultilabelF1Score(num_labels=num_cls).to(device)
    weighted = 'weighted' if loss_weighted else 'noweighted'
    since = time.time()
    best_model_wts = model.state_dict()
    best_f1 = 0.0
    losses = {'train': [], 'test': []}
    f1_macro = {'train': [], 'test': []}
    pbar = trange(num_epochs, desc='Epoch:')

    for epoch in pbar:
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train(True)
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            true_labels = []
            pred_labels = []
            batch_f1_score = []
            for data in tqdm(dataloaders[phase], leave=False, desc=f'{phase} iter:'):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels.to(torch.float32)
                if phase == 'train':
                    optimizer.zero_grad()
                if phase == 'test':
                    with torch.no_grad():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)
                preds = torch.argmax(outputs, -1)
                loss = criterion(outputs, labels)
                true_labels += labels.tolist()
                pred_labels += outputs.tolist()
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item()
                #multilb
#                 print(f'Outputs:{outputs}, labels:{labels}')
                batch_f1_score.append(f1(outputs, labels).cpu().numpy())#change this
#                 print(f'batch_f1_score: {batch_f1_score}')
            f1_score_batch = np.mean(batch_f1_score)
            epoch_loss = running_loss / len(dataloaders[phase])
#             print(f'pred_labels:{pred_labels}, true_labels: {true_labels}')
            f1_score = f1(torch.tensor(pred_labels).to(device), 
                             torch.tensor(true_labels).to(device)).to(device)
#             f1_score = f1(torch.tensor(pred_labels).to(device), torch.tensor(true_labels).to(device))#change this
            losses[phase].append(epoch_loss)
            f1_macro[phase].append(f1_score)
#             if f1_score != f1_score_ep:
#                 print('________DIFFERENCE DETECTED________')
            pbar.set_description('{} Loss: {:.4f} F1_batch: {:.4f} F1_ep: {:.4f}'.format(
                phase, epoch_loss, f1_score_batch, f1_score
            ))
            ###
            wandb.log({ f"{phase} F1": f1_score,  # still not quite shure about f1
                    f"{phase} Loss": epoch_loss})
            ###
            if (epoch+1)%10==0 and phase == 'test':
                #print(f'Saving model:{model_name} after epoch{epoch+1}')
                #PATH = f'models/dct_{model_name}_{d_loader_name}_{weighted}_epoch_{epoch+1}_f1_{f1_score}.pt'
                #torch.save(model.state_dict(), PATH)
                ###
                log_image_table(images = inputs, predicted=preds, labels=labels,
                                probs=outputs.softmax(dim=1), num_cls=num_cls)
                ###
            if phase == 'test' and f1_score > best_f1:
                print(f'now we have wts with f1_score={f1_score}')
                best_f1 = f1_score
                best_model_wts = copy.deepcopy(model.state_dict())    
    PATH = f'models/best_model_{model_name}_{d_loader_name}_{weighted}_epoch_{epoch+1}_f1_{best_f1}.pt'
    torch.save(best_model_wts, PATH)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test f1_macro: {:4f}'.format(best_f1))

    return model, losses, f1_macro


def debarcle_layers(model, num_debarcle):
    '''Debarcle From the last [-1]layer to the [-num_debarcle] layers, 
    approximately(for there is Conv2d which has only weight parameter)'''
    num_debarcle *= 2
    param_name = [name for name,_ in model.named_parameters()]
    param_debarcle = param_name[-num_debarcle:]
    if param_debarcle[0].split('.')[-1] == 'bias':
        param_debarcle = param_name[-(num_debarcle + 1):]
    for name, param in model.named_parameters():
        param.requires_grad = True if name in param_debarcle else False
        

def choose_model(model_name, freeze, num_cls, debarcle):
    # Chose model
    resnet_timm = [ 'resnet50d',
                     'resnet101',
                     'resnet101d',
                     'resnet152',
                     'resnet152d',
                     'resnet200d',
                     'resnetrs200',
                     'resnetrs270',
                     'resnetrs350',
                     'resnetrs420',
                     'resnetv2_50',
                     'resnetv2_50d_evos',
                     'resnetv2_50d_gn',
                     'resnetv2_50x1_bit_distilled',
                     'resnetv2_50x1_bitm',
                     'resnetv2_50x1_bitm_in21k',
                     'resnetv2_50x3_bitm',
                     'resnetv2_50x3_bitm_in21k',
                     'resnetv2_101',
                     'resnetv2_101x1_bitm',
                     'resnetv2_101x1_bitm_in21k',
                     'resnetv2_101x3_bitm',
                     'resnetv2_101x3_bitm_in21k',
                     'resnetv2_152x2_bit_teacher',
                     'resnetv2_152x2_bit_teacher_384',
                     'resnetv2_152x2_bitm',
                     'resnetv2_152x2_bitm_in21k',
                     'resnetv2_152x4_bitm',
                     'resnetv2_152x4_bitm_in21k',
                     'seresnet50',
                     'wide_resnet50_2',
                     'wide_resnet101_2']
    
    timm_clip = ['vit_base_patch32_224_clip_laion2b',
                 'vit_giant_patch14_224_clip_laion2b',
                 'vit_huge_patch14_224_clip_laion2b',
                 'vit_large_patch14_224_clip_laion2b']
    
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True, progress=True)
        in_feats = model.fc.in_features
        
    elif model_name in resnet_timm:
        model = timm.create_model(model_name, pretrained = True)
        if model.default_cfg['classifier'] == 'fc':
            in_feats = model.fc.in_features
        elif model.default_cfg['classifier'] == 'head.fc':
            in_feats = model.head.fc.in_channels
        
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=True, progress=True)
        in_feats = model.classifier[0].in_features

    elif model_name.startswith('tf_efficientnet') or model_name.startswith('efficientnet'):
        model = timm.create_model(model_name, pretrained = True)
        in_feats = model.classifier.in_features
        
    elif model_name =='swin_large_patch4_window12_384_in22k':
        model = timm.create_model('swin_large_patch4_window12_384_in22k', pretrained = True)
        in_feats = model.head.in_features
        
    elif model_name in timm_clip:
        model = timm.create_model(model_name, pretrained = True)
        in_feats = model.head.in_features
        
    elif model_name =='convnext_base_384_in22ft1k':
        model = timm.create_model('convnext_base_384_in22ft1k', pretrained = True)
        in_feats = model.head.fc.in_features
            
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=True)
        in_feats = 1280

    elif model_name == 'inception_v3':
        model = models.inception_v3(pretrained=True, progress=True)
        in_feats = model.fc.in_features

    if freeze:
        for param in model.parameters():
            param.requires_grad = False
            
    if debarcle > 0:
        debarcle_layers(model, num_debarcle=debarcle)


    # Chose model
    if model_name == 'resnet50':
        model.fc = nn.Sequential(nn.Linear(in_feats, 1024),
                            nn.ReLU(inplace=True),
                            nn.Linear(1024, 512),
                            nn.ReLU(inplace=True),
                            nn.Linear(512, num_cls))
        
    elif model_name in resnet_timm:
        if model.default_cfg['classifier'] == 'fc':
            model.fc = nn.Sequential(nn.Linear(in_feats, 1024),
                            nn.ReLU(inplace=True),
                            nn.Linear(1024, 512),
                            nn.ReLU(inplace=True),
                            nn.Linear(512, num_cls))
        elif model.default_cfg['classifier'] == 'head.fc':
            model.head.fc = nn.Sequential(nn.Linear(in_feats, 1024),
                            nn.ReLU(inplace=True),
                            nn.Linear(1024, 512),
                            nn.ReLU(inplace=True),
                            nn.Linear(512, num_cls))

    elif model_name.startswith('tf_efficientnet') or model_name.startswith('efficientnet'):
        model.classifier = nn.Sequential(nn.Linear(in_feats, 1024),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(1024, 512),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(512, num_cls))
    
        
    elif model_name =='swin_large_patch4_window12_384_in22k':
        model.head = nn.Sequential(nn.Linear(in_feats, 1024),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(1024, 512),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(512, num_cls))
        
    elif model_name =='convnext_base_384_in22ft1k':
        model.head.fc = nn.Sequential(nn.Linear(in_feats, 1024),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(1024, 512),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(512, num_cls))
        
    elif model_name in timm_clip:
        model.head = nn.Sequential(nn.Linear(in_feats, 1024),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(1024, 512),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(512, num_cls))
    elif model_name == 'mobilenet_v2':
        model.classifier = nn.Sequential(nn.Linear(in_feats, 1024),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(1024, 512),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(512, num_cls))
        
    elif model_name == 'inception_v3':
        model.fc = nn.Sequential(nn.Linear(in_feats, 1024),
                            nn.ReLU(inplace=True),
                            nn.Linear(1024, 512),
                            nn.ReLU(inplace=True),
                            nn.Linear(512, num_cls))
        
    return model

class Model_multilb(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.sigm = nn.Sigmoid()
 
    def forward(self, x):
        return self.sigm(self.model(x))
    
    
def calculate_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    return {
#             'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
#             'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
#             'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),
#             'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),
#             'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),
#             'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),
#             'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples'),
#             'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples'),
            'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples'),
            }