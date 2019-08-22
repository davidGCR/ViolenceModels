import torch.nn as nn
from torchvision import models
from util import * 
import torch

class ViolenceModelResNet(nn.Module):
    def __init__(self, seqLen, feature_extract=True):
        super(ViolenceModelResNet, self).__init__()
        self.seqLen = seqLen
        self.model_ft = models.resnet18(pretrained=True)

        self.set_parameter_requires_grad(self.model_ft, feature_extract)
        
        self.num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(self.seqLen*self.num_ftrs, 2)
    
    def forward(self, x):
        lista = []
        for dimage in range(0, self.seqLen):
            feature = self.convNet(x[dimage])
        #       print('--->feature  (CNN output) size: ',feature.size())
            feature = feature.view(feature.size(0), 256 * 6 * 6)
            lista.append(feature)
    #       print('--->feature VIEW (CNN output) size: ',feature.size())
        
        x = torch.cat(lista, dim=1)  
    #     print('x cat: ',x.size())
        x = self.linear(x)
        
    #     print('x classifier: ',x.size())
        
    #       print('feature (CNN output)size: ',feature.size())
            
        return x