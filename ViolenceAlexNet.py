import torch.nn as nn
from torchvision import models
from util import * 
import torch


class ViolenceModel(nn.Module):
  def __init__(self, seqLen):
      super(ViolenceModel, self).__init__()
      self.seqLen = seqLen
      self.alexnet = models.alexnet(pretrained=True)
      feature_extract = True
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