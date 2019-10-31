import torch.nn as nn
from torchvision import models
from util import * 
from Identity import *
from tempPooling import *
import torch

class ViolenceModelResNet(nn.Module):
    def __init__(self, num_classes, numDiPerVideos, model_name, joinType ,feature_extract):
        super(ViolenceModelResNet, self).__init__()
        self.numDiPerVideos = numDiPerVideos
        self.num_classes = num_classes
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
            self.linear = nn.Linear(self.num_ftrs*self.numDiPerVideos,self.num_classes)
        elif self.joinType == 'tempMaxPool':
            self.linear = nn.Linear(512, self.num_classes)
            # if model_name == 'resnet18' else nn.Linear(512*7*7,self.num_classes)
    
    def forward(self, x):
        # print('forward input size:',x.size())
        shape = x.size()
        if len(shape) == 4:
            x = self.model_ft(x)
            x = torch.flatten(x, 1)
            # print('x: ',x.size())
        else:
            if self.joinType == 'cat':
                x = self.getFeatureVectorCat(x)
                # print('cat input size:',x.size())
            elif self.joinType == 'tempMaxPool':
                x = self.getFeatureVectorTempPool(x)
            # elif self.joinType == 'tempMaxPool_list':
            #     x = self.getFeatureVectorTempPool2(x)
        x = self.linear(x)
        return x
    
    def getFeatureVectorTempPool2(self, x):
        lista = []
        seqLen = len(x) #batch size
        # print(seqLen)
        for dimage in range(0, seqLen):
            feature = self.convLayers(x[dimage])
            lista.append(feature)

        # minibatch = torch.stack(lista, 0)
        # minibatch = minibatch.permute(1, 0, 2, 3, 4)
        num_dynamic_images = seqLen
        tmppool = nn.MaxPool2d((num_dynamic_images, 1))
        lista_minibatch = []
        for idx in range(len(lista)):
            out = tempMaxPooling2(lista[idx], tmppool)
            print('out', out.size())
            lista_minibatch.append(out)
        print('lista_minibatch: ',len(lista_minibatch))
        feature = torch.stack(lista_minibatch, 0)
        feature = torch.flatten(feature, 1)
        return feature

    def getFeatureVectorTempPool(self, x):
        lista = []
        seqLen = self.numDiPerVideos
        # print(seqLen)
        for dimage in range(0, seqLen):
            feature = self.convLayers(x[dimage])
            lista.append(feature)

        minibatch = torch.stack(lista, 0)
        minibatch = minibatch.permute(1, 0, 2, 3, 4)
        num_dynamic_images = self.numDiPerVideos
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
        for dimage in range(0, self.numDiPerVideos):
            feature = self.model_ft(x[dimage])
            # feature = torch.flatten(feature, 1)
            # feature = feature.view(feature.size(0), self.num_ftrs)
            lista.append(feature)
        x = torch.cat(lista, dim=1)
        return x