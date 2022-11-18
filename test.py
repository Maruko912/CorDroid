import dgl as dgl
import os
from sklearn import metrics
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
from utils import *
import numpy as np
from MyDataset import *
from torch.autograd import Variable
import time
from omm.CNN_Model import CNN_Model
from sfcg.GCN_Model import GCN_Model
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

Config = {
    'batch_size' : 8,
    'num_classes' : 36,
    # 'device' : torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'device' : torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'save_folder' : './model',
    'gcn_model' :"model/GCN_model.pkl",
    'cnn_model' :"model/CNN_model.pkl",
    "classes_name_txt" : "data/FC_1/classes_name.txt",
    "test_data_txt" : "data/FC_1/test.txt"
}

target_names = load_data(Config['classes_name_txt'])
testset = Mydataset(load_data(Config['test_data_txt']))
test_loader = data.DataLoader(testset, batch_size=Config['batch_size'], shuffle=True, pin_memory=True, collate_fn=collate)

device = Config['device']
model_sfcg = GCN_Model(100, 256, 512, 256, 128, Config['num_classes'])
model_sfcg.to(device)
model_omm = CNN_Model(num_classes=Config['num_classes'])
model_omm.to(device)


def test(test_report_save):
    checkpoint1 = torch.load(Config['gcn_model'])
    model_sfcg.load_state_dict(checkpoint1['model_state_dict'])
    checkpoint2 = torch.load(Config['cnn_model'])
    model_omm.load_state_dict(checkpoint2['model_state_dict'])

    model_sfcg.eval()
    model_omm.eval()
    with torch.no_grad():
        test_y =[]
        test_result_fcg = []
        test_result_omm = []
        test_result_des = []
        for i, (batchg, img, label) in enumerate(test_loader):
            batchg, img, label = batchg.to(device), img.to(device), label.to(device)
            gcn_outputs = model_sfcg(batchg)
            cnn_outputs = model_omm(img)
            _, gcn_pred = torch.max(gcn_outputs.data, 1)
            test_result_fcg.extend(gcn_pred.cpu())

            _, cnn_pred = torch.max(cnn_outputs.data, 1)
            test_result_omm.extend(cnn_pred.cpu())

            w_gcn = 0.5
            w_omm = 0.5

            outputs = gcn_outputs*w_gcn + cnn_outputs*w_omm
            _, des_pred = torch.max(outputs.data, 1)
            test_result_des.extend(des_pred.cpu())

            test_y.extend(label.data.cpu())
            del batchg
            del img
            del label
        test_report_fcg = metrics.classification_report(test_y, test_result_fcg,target_names=target_names, digits=6)
        test_report_omm = metrics.classification_report(test_y, test_result_omm,target_names=target_names, digits=6)
        test_report_des = metrics.classification_report(test_y, test_result_des,target_names=target_names, digits=6)
        # print(test_report_fcg,test_report_omm,test_report_des)
        print(test_report_des)
        with open("report/des_"+test_report_save,'w') as f:
            f.write("fcg\n"+test_report_fcg+'\nomm\n'+test_report_omm+'\ndes\n'+test_report_des)
if __name__ == "__main__":
    test_report_save = "report.txt"
    test(test_report_save)