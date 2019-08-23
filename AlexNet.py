import torch.nn as nn
from torchvision import models
from util import * 
import torch


class ViolenceModelAlexNetV1(nn.Module): ##ViolenceModel
  def __init__(self, seqLen, feature_extract):
      super(ViolenceModelAlexNetV1, self).__init__()
      self.seqLen = seqLen
      self.alexnet = models.alexnet(pretrained=True)
      self.feature_extract = feature_extract
      set_parameter_requires_grad(self.alexnet, feature_extract)
      
      self.convNet = nn.Sequential(*list(self.alexnet.features.children()))
      self.linear = nn.Linear(256*6*6*seqLen,2)
      self.alexnet = None

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
############################################################################

class ViolenceModelAlexNetV2(nn.Module): ##ViolenceModel2
  def __init__(self, seqLen, feature_extract= True):
      super(ViolenceModelAlexNetV2, self).__init__()
      self.seqLen = seqLen
      self.alexnet = models.alexnet(pretrained=True)
      
      self.convNet = nn.Sequential(*list(self.alexnet.features.children()))
      
      self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
      self.classifier = nn.Sequential(
          nn.Dropout(),
          nn.Linear(256 * 6 * 6, 4096),
          nn.ReLU(inplace=True),
          nn.Dropout(),
          nn.Linear(4096, 4096),
          nn.ReLU(inplace=True),
      )
      self.feature_extract = feature_extract
      
      set_parameter_requires_grad(self.alexnet, feature_extract)
      set_parameter_requires_grad(self.avgpool, feature_extract)
      set_parameter_requires_grad(self.classifier, feature_extract)
      
#       self.linear = nn.Linear(256*6*6*seqLen,2)
      self.linear = nn.Linear(4096*seqLen,2)
      self.alexnet = None

  def forward(self, x):
    lista = []
    for dimage in range(0, self.seqLen):
      feature = self.convNet(x[dimage])
      
      feature = self.avgpool(feature)
      feature = torch.flatten(feature, 1)
      feature = self.classifier(feature)
      
#       print('--->feature  (CNN output) size: ',feature.size())
#       feature = feature.view(feature.size(0), 256 * 6 * 6)
      feature = feature.view(feature.size(0), 4096)
      lista.append(feature)
#       print('--->feature VIEW (CNN output) size: ',feature.size())
      
    x = torch.cat(lista, dim=1)  
#     print('x cat: ',x.size())
    x = self.linear(x)
    
#     print('x classifier: ',x.size())
    
#       print('feature (CNN output)size: ',feature.size())
        
    return x