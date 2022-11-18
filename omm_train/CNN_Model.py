import math
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
# https://github.com/yueruchen/sppnet-pytorch/blob/master/spp_layer.py

model_name = 'efficientnet-b0'

class CNN_Model(torch.nn.Module):
    def __init__(self, num_classes, model_name = 'efficientnet-b0'):
        super(CNN_Model, self).__init__()
        self.efficientNet = EfficientNet.from_pretrained(model_name, in_channels = 1, drop_connect_rate=0.05, dropout_rate=0.2, num_classes=num_classes)
        # self.efficientNet._avg_pooling = self.spatial_pyramid_pool(out_pool_size=[4,2,1])
        self.dense1 = torch.nn.Sequential(
            torch.nn.Linear(5 * 1280, 512, bias=True),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU())
        self.dense2 = torch.nn.Sequential(
            torch.nn.Linear(512, 256, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_classes))


    def spatial_pyramid_pool(self, previous_conv, num_sample, previous_conv_size, out_pool_size):
        '''
        previous_conv: a tensor vector of previous convolution layer
        num_sample: an int number of image in the batch
        previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
        out_pool_size: a int vector of expected output size of max pooling layer

        returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
        '''
        # print(previous_conv.size())
        for i in range(len(out_pool_size)):
            # print(previous_conv_size)
            h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
            w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
            h_pad = int((h_wid * out_pool_size[i] - previous_conv_size[0] + 1) / 2)
            w_pad = int((w_wid * out_pool_size[i] - previous_conv_size[1] + 1) / 2)
            maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
            x = maxpool(previous_conv)
            if (i == 0):
                spp = x.view(num_sample, -1)
            else:
                spp = torch.cat((spp, x.view(num_sample, -1)), 1)
        return spp

    def m_extract_feature(self, x):
        x1 = self.efficientNet.extract_features(x)
        x2 = self.spatial_pyramid_pool(x1, int(x1.size(0)),[int(x1.size(2)),int(x1.size(3))],[2, 1])
        x3 = self.dense1(x2)
        return x3
    def forward(self, x):
        x = self.m_extract_feature(x)
        x = self.dense2(x)
        return x
