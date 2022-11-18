import dgl as dgl
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
from utils import *
from LabelSmoothSoftmaxCE import *
import numpy as np
from mydataset import *
from torch.autograd import Variable
import time
from GCN_Model import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

Config = {
    'learning_rate' : 0.01,
    'batch_size' : 4,
    'accumulation_steps' : 16,
    'epoches' : 50,
    # 'device' : torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'device' : torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'save_folder' : './model',
    'input_dim' : 100,
    'output_dim1' : 256,
    'output_dim2' : 512,
    'output_dim3' : 256,
    'output_dim4' : 128,
    'num_classes' : 55,
    'train_txt' : 'train.txt',
    'val_txt' : 'test.txt',
}


trainset = Mydataset(load_data(Config['train_txt']))
valset = Mydataset(load_data(Config['val_txt']))

train_loader = data.DataLoader(trainset, batch_size=Config['batch_size'], shuffle=True, pin_memory=True, collate_fn=collate, drop_last=False)
val_loader = data.DataLoader(valset, batch_size=Config['batch_size'], shuffle=True, pin_memory=True, collate_fn=collate, drop_last=False)

device = Config['device']
model = GCN_Model(Config['input_dim'], Config['output_dim1'], Config['output_dim2'], Config['output_dim3'], Config['output_dim4'], Config['num_classes'])
# model = nn.DataParallel(model,device_ids=[0,1],output_device=[0])
model.to(device)


loss_func = LabelSmoothSoftmaxCE(lb_pos=0.95, lb_neg=0.0014)
optimizer = torch.optim.Adam(model.parameters(),lr=Config['learning_rate'], weight_decay=0.0005)
n_epochs = Config['epoches']
accumulation_steps = Config['accumulation_steps']

def train_model(Config, bool_continue, load_epoch):
    if os.path.exists(Config['save_folder'] + "/GCN_model_" + str(load_epoch).zfill(4) + ".pkl") and bool_continue:
        checkpoint = torch.load(Config['save_folder'] + "/GCN_model_" + str(load_epoch).zfill(4) + ".pkl")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # start_epoch = checkpoint['epoch']
        start_epoch = load_epoch
        print('loading epoch {} succeeded！'.format(start_epoch))
    else:
        start_epoch = 0
        if not os.path.exists(Config['save_folder']):
            os.makedirs(Config['save_folder'])
        print("The model doesn't exist, it will be trained from scratch!")
    begin_time = time.time()
    for epoch in range(start_epoch+1, n_epochs+1):
        now_lr = adjust_learning_rate(optimizer, epoch, Config['learning_rate'])
        f = open('./train_record.txt', 'a+')
        f.write("Epoch:" + str(epoch) + " now learning rate : " + str(now_lr) + "\n")
        print("Epoch  {}/{} now learning rate : {}".format(epoch, n_epochs, now_lr))
        avg_train_loss, avg_train_acc, loss = train(f, begin_time)
        avg_val_loss, avg_val_accuracy = validation()
        print(
            "Avg Train Loss is :{:.6f}, Avg Train Accuracy is:{:.6f}%, Avg Test Loss is:{:.6f}, Avg Test Accuracy is:{:.6f}%".format(
                avg_train_loss, avg_train_acc, avg_val_loss, avg_val_accuracy))
        f.write(
            "Avg Train Loss is :{:.6f}, Avg Train Accuracy is:{:.6f}%, Avg Test Loss is:{:.6f}, Avg Test Accuracy is:{:.6f}%".format(
                avg_train_loss, avg_train_acc, avg_val_loss, avg_val_accuracy))
        f.write("\n")
        f.close()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, Config['save_folder'] + "/GCN_model_" + str(epoch).zfill(4) + ".pkl")

def train(f, begin_time):
    model.train()
    train_loss = 0.0
    train_correct = 0.0
    for i, (batchg, label) in enumerate(train_loader):

        batchg, label = batchg.to(device), label.to(device)
        outputs = model(batchg)
        _, pred = torch.max(outputs.data, 1)

        try:
            loss = loss_func(outputs, label)
            loss.backward()
        except:
            print("batchg, error", batchg)
            continue
        train_loss += loss.item()
        train_correct += torch.sum(pred == label.data)
        if ((i + 1) % accumulation_steps) == 0:
            optimizer.step()
            optimizer.zero_grad()
            print("Train set " + str(i + 1) + "/" + str(len(train_loader)),
                  str(train_correct.item()) + "/" + str((i + 1) * Config['batch_size']), " Avg Loss:",
                  train_loss / ((i + 1) * Config['batch_size']),
                  " Avg Acc:", train_correct.item() / ((i + 1) * Config['batch_size']),
                  "time:{:.4f}".format(time.time() - begin_time))
        del batchg
        del label
    avg_train_loss = train_loss / len(trainset)
    avg_train_acc = 100. * float(train_correct) / float(len(trainset))
    return  avg_train_loss, avg_train_acc, loss



def validation():
    model.eval()
    val_loss = 0
    val_correct = 0
    with torch.no_grad():
        for i, (X_val, y_val) in enumerate(val_loader):
            X_val, y_val = X_val.to(device), y_val.to(device)
            outputs = model(X_val)
            _, pred = torch.max(outputs.data, 1)
            val_correct += torch.sum(pred == y_val.data)
            loss = loss_func(outputs, y_val)
            val_loss += loss.item()
            del X_val
            del y_val
        val_loss /= len(valset)
        val_accuracy = 100. * float(val_correct) / float(len(valset))
        print('Test set: Average loss: {:.6f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
            val_loss, val_correct, len(valset), 100. * val_correct / len(valset)))
        return val_loss, val_accuracy



if __name__ == "__main__":
    bool_continue = False
    load_epoch = 30

    train_model(Config, bool_continue, load_epoch)