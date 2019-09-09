import torch.nn as nn
from torchvision import models
from util import * 
from Identity import *
from tempPooling import *
import torch

class ViolenceModelResNet(nn.Module):
    def __init__(self, seqLen, model_name, joinType ,feature_extract):
        super(ViolenceModelResNet, self).__init__()
        self.seqLen = seqLen
        self.joinType = joinType
        if model_name == 'resnet18':
            self.model_ft = models.resnet18(pretrained=True)
        elif model_name == 'resnet34':
            self.model_ft = models.resnet34(pretrained=True)
        self.num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = Identity()
        self.convLayers = nn.Sequential(*list(self.model_ft.children())[:-2]) # to tempooling

        set_parameter_requires_grad(self.model_ft, feature_extract)
        set_parameter_requires_grad(self.convLayers, feature_extract)
        if self.joinType == 'cat':
            self.linear = nn.Linear(self.num_ftrs*self.seqLen,2)
        elif self.joinType == 'tempMaxPool':
            self.linear = nn.Linear(512*7*7,2)

        # self.conv1 = self.model_ft.conv1
        # self.bn1 = self.model_ft.bn1
        # self.relu = self.model_ft.relu
        # self.maxpool = self.model_ft.maxpool
        
        # self.l1 = nn.Sequential(*list(self.model_ft.layer1.children()))
        # self.l2 = nn.Sequential(*list(self.model_ft.layer2.children()))
        # self.l3 = nn.Sequential(*list(self.model_ft.layer3.children()))
        # self.l4 = nn.Sequential(*list(self.model_ft.layer4.children()))

        # self.avgpool = self.model_ft.avgpool

        # set_parameter_requires_grad(self.conv1, feature_extract)
        # set_parameter_requires_grad(self.bn1, feature_extract)
        # set_parameter_requires_grad(self.relu, feature_extract)
        # set_parameter_requires_grad(self.maxpool, feature_extract)
        # set_parameter_requires_grad(self.l1, feature_extract)
        # set_parameter_requires_grad(self.l2, feature_extract)
        # set_parameter_requires_grad(self.l3, feature_extract)
        # set_parameter_requires_grad(self.l4, feature_extract)
        # set_parameter_requires_grad(self.avgpool, feature_extract)
        
        # self.num_ftrs = self.model_ft.fc.in_features
        # self.model_ft = None
        # self.fc = nn.Linear(self.seqLen*self.num_ftrs, 2)
    
    def forward(self, x):
        # print('forward input size:',x.size())
        if self.joinType == 'cat':
            x = self.getFeatureVectorCat(x)
            # print('cat input size:',x.size())
        elif self.joinType == 'tempMaxPool':
            x = self.getFeatureVectorTempPool(x)
            # print('tempPooling input size:',x.size())
        # print('linear input size:',x.size())
        x = self.linear(x)
        return x
        # lista = []
        # for dimage in range(0, self.seqLen):
        #     feature = self.conv1(x[dimage])
        #     feature = self.bn1(feature)
        #     feature = self.relu(feature)
        #     feature = self.maxpool(feature)
        #     feature = self.l1(feature)
        #     feature = self.l2(feature)
        #     feature = self.l3(feature)
        #     feature = self.l4(feature) ##torch.Size([64, 512, 7, 7])
        #     feature = self.avgpool(feature)
        #     feature = torch.flatten(feature, 1)
        #     feature = feature.view(feature.size(0), self.num_ftrs)
        #     lista.append(feature)
        # x = torch.cat(lista, dim=1)  
        # x = self.fc(x)
        # return x
    def getFeatureVectorTempPool(self, x):
        lista = []
        for dimage in range(0, self.seqLen):
            feature = self.convLayers(x[dimage])
            lista.append(feature)

        minibatch = torch.stack(lista, 0)
        minibatch = minibatch.permute(1, 0, 2, 3, 4)
        num_dynamic_images = self.seqLen
        tmppool = nn.MaxPool2d((num_dynamic_images, 1))
        lista_minibatch = []
        for idx in range(minibatch.size()[0]):
            out = tempMaxPooling(minibatch[idx], tmppool)
            lista_minibatch.append(out)

        feature = torch.stack(lista_minibatch, 0)
        feature = torch.flatten(feature, 1)
        return feature
    
    def getFeatureVectorCat(self, x):
        lista = []
        for dimage in range(0, self.seqLen):
            feature = self.model_ft(x[dimage])
            # feature = torch.flatten(feature, 1)
            # feature = feature.view(feature.size(0), self.num_ftrs)
            lista.append(feature)
        x = torch.cat(lista, dim=1)
        return x