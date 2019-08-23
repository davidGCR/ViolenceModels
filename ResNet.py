import torch.nn as nn
from torchvision import models
from util import * 
import torch

class ViolenceModelResNet(nn.Module):
    def __init__(self, seqLen, feature_extract=True):
        super(ViolenceModelResNet, self).__init__()
        self.seqLen = seqLen

        self.model_ft = models.resnet18(pretrained=True)

        self.conv1 = self.model_ft.conv1
        self.bn1 = self.model_ft.bn1
        self.relu = self.model_ft.relu
        self.maxpool = self.model_ft.maxpool
        
        self.l1 = nn.Sequential(*list(self.model_ft.layer1.children()))
        self.l2 = nn.Sequential(*list(self.model_ft.layer2.children()))
        self.l3 = nn.Sequential(*list(self.model_ft.layer3.children()))
        self.l4 = nn.Sequential(*list(self.model_ft.layer4.children()))

        self.avgpool = self.model_ft.avgpool

        

        set_parameter_requires_grad(self.conv1, feature_extract)
        set_parameter_requires_grad(self.bn1, feature_extract)
        set_parameter_requires_grad(self.relu, feature_extract)
        set_parameter_requires_grad(self.maxpool, feature_extract)
        set_parameter_requires_grad(self.l1, feature_extract)
        set_parameter_requires_grad(self.l2, feature_extract)
        set_parameter_requires_grad(self.l3, feature_extract)
        set_parameter_requires_grad(self.l4, feature_extract)
        set_parameter_requires_grad(self.avgpool, feature_extract)
        
        self.num_ftrs = self.model_ft.fc.in_features
        self.model_ft = None
        self.fc = nn.Linear(self.seqLen*self.num_ftrs, 2)
    
    def forward(self, x):
        lista = []

        for dimage in range(0, self.seqLen):
            feature = self.conv1(x[dimage])
            feature = self.bn1(feature)
            feature = self.relu(feature)
            feature = self.maxpool(feature)
            feature = self.l1(feature)
            feature = self.l2(feature)
            feature = self.l3(feature)
            feature = self.l4(feature)

            feature = self.avgpool(feature)
            feature = torch.flatten(feature, 1)
            feature = self.fc(feature)
            # feature = feature.view(feature.size(0), self.num_ftrs)
            # lista.append(feature)
            print('--->feature fc output size: ',feature.size())
        
        # x = torch.cat(lista, dim=1)  
    #     print('x cat: ',x.size())
        # x = self.fc(x)
        
    #     print('x classifier: ',x.size())
        
    #       print('feature (CNN output)size: ',feature.size())
            
        return x