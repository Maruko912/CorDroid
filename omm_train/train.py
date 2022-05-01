import os
import random
import shutil
import time
import warnings
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data as data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
from MyDataset import *
from utils import *
# from efficientnet_pytorch import EfficientNet
from CNN_Model import *
from LabelSmoothSoftmaxCE import *


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

Config = {
    'learning_rate' : 0.0005,
    'batch_size' : 16,
    'accumulation_steps' : 1,
    'epoches' : 50,
    'data_local' : './dataset8407',
    'device' : torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'num_classes' : 2,
    'save_folder' : './model',
    'train_txt' : 'train.txt',
    'val_txt' : 'data_split\test.txt',
}


train_dataset = Mydataset(load_data(Config['train_txt']))
val_dataset = Mydataset(load_data(Config['val_txt']))

train_loader = data.DataLoader(train_dataset, batch_size=Config['batch_size'], shuffle=True, pin_memory=True)
val_loader = data.DataLoader(val_dataset, batch_size=Config['batch_size'], shuffle=True, pin_memory=True)

device = Config['device']

cnn_model = CNN_Model(model_name = model_name, num_classes=Config['num_classes'])
cnn_model = cnn_model.to(device)

loss_func = LabelSmoothSoftmaxCE(lb_pos=0.95, lb_neg=0.0014)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, cnn_model.parameters()),lr=Config['learning_rate'], weight_decay=0.0005)

n_epochs = Config['epoches']
accumulation_steps = Config['accumulation_steps']

def train_model(Config, bool_continue, load_epoch):
    if os.path.exists(Config['save_folder']+"/"+model_name+"_"+str(load_epoch).zfill(4)+".pkl") and bool_continue:
        checkpoint = torch.load(Config['save_folder']+"/"+model_name+"_"+str(load_epoch).zfill(4)+".pkl")
        cnn_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # start_epoch = checkpoint['epoch']
        start_epoch = load_epoch
        print('loading epoch {} succeeded！'.format(start_epoch))
    else:
        start_epoch = 0
        if not os.path.exists(Config['save_folder']):
            os.makedirs(Config['save_folder'])
        print("The model doesn't exist, it will be trained from scratch!")

    for epoch in range(start_epoch+1, n_epochs+1):
        now_lr = adjust_learning_rate(optimizer, epoch, Config['learning_rate'])
        print("Epoch  {}/{}, now learning rate is : {}".format(epoch, n_epochs, now_lr))
        f = open('./train_record.txt','a+')
        f.write("Epoch:"+str(epoch)+"\n")
        avg_train_loss, avg_train_acc, loss = train(f)
        avg_val_loss, avg_val_accuracy = validation()
        print("Avg Train Loss is :{:.6f}, Avg Train Accuracy is:{:.6f}%, Avg Test Loss is:{:.6f}, Avg Test Accuracy is:{:.6f}%".format(avg_train_loss, avg_train_acc, avg_val_loss, avg_val_accuracy))
        f.write("Avg Train Loss is :{:.6f}, Avg Train Accuracy is:{:.6f}%, Avg Test Loss is:{:.6f}, Avg Test Accuracy is:{:.6f}%".format(avg_train_loss, avg_train_acc, avg_val_loss, avg_val_accuracy))
        f.write("\n")
        f.close()
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': cnn_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, Config['save_folder']+"/"+model_name+"_"+str(epoch).zfill(4)+".pkl")
        # torch.save(cnn_model.state_dict(),Config['save_folder']+"/s_"+str(epoch).zfill(4)+".pkl")



def train(f):
    cnn_model.train()
    train_loss = 0.0
    train_correct = 0.0
    for i,(X_train, y_train) in enumerate(train_loader):
        X_train, y_train = Variable(X_train.to(device)), Variable(y_train.to(device))
        outputs = cnn_model(X_train)
        _, pred = torch.max(outputs.data, 1)
        loss = loss_func(outputs, y_train)
        loss.backward()
        train_loss += loss.item()
        train_correct += torch.sum(pred == y_train.data)
        if((i+1)%accumulation_steps)==0:
            optimizer.step()
            optimizer.zero_grad()
            print("Train set " + str(i+1) + "/" + str(len(train_loader)) + " Avg Loss:",train_loss/((i+1)*Config['batch_size']),
        " Avg Acc:", train_correct.item()/((i+1)*Config['batch_size']))

    avg_train_loss = train_loss / len(train_dataset)
    avg_train_acc = 100. * float(train_correct) / float(len(train_dataset))
    return  avg_train_loss, avg_train_acc, loss

def validation():
    cnn_model.eval()
    val_loss = 0
    val_correct = 0
    with torch.no_grad():
        for i, (X_val, y_val) in enumerate(val_loader):
            X_val, y_val = Variable(X_val.to(device)), Variable(y_val.to(device))
            outputs = cnn_model(X_val)
            _, pred = torch.max(outputs.data, 1)
            val_correct += torch.sum(pred == y_val.data)
            loss = loss_func(outputs, y_val)
            val_loss += loss.item()
        val_loss /= len(val_dataset)
        val_accuracy = 100. * float(val_correct) / float(len(val_dataset))
        print('Test set: Average loss: {:.6f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
            val_loss, val_correct, len(val_dataset), 100. * val_correct / len(val_dataset)))
        return val_loss, val_accuracy

if __name__ == "__main__":
    bool_continue = False
    load_epoch = 20
    train_model(Config, bool_continue, load_epoch)
