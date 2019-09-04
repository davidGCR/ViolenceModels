import torch.nn as nn
from torchvision import models
from util import *
from tempPooling import *
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
############################################################################
############################################################################

class ViolenceModelAlexNetV2(nn.Module): ##ViolenceModel2
  def __init__(self, seqLen, joinType, feature_extract= True):
      super(ViolenceModelAlexNetV2, self).__init__()
      self.seqLen = seqLen
      self.joinType = joinType
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
      
      if self.joinType == 'cat':
        self.linear = nn.Linear(4096*seqLen,2)
      elif self.joinType == 'tempMaxPool':
        self.linear = nn.Linear(4096,2)
      self.alexnet = None

  def getFeatureVector(self, x):
    lista = []
    for dimage in range(0, self.seqLen):
      feature = self.convNet(x[dimage])
      feature = self.avgpool(feature)
      feature = torch.flatten(feature, 1)
      feature = self.classifier(feature)
      feature = feature.view(feature.size(0), 4096)
      lista.append(feature)
    x = torch.cat(lista, dim=1) 
    # if self.joinType == 'cat':
    #   x = self.catType(x)
    # elif self.joinType == 'tempMaxPool':
    #   x = self.tempMaxPoolingType(x)
    #   x = self.classifier(x)
    return x

  def catType(self, x):
    lista = []
    for dimage in range(0, self.seqLen):
      feature = self.convNet(x[dimage])
      feature = self.avgpool(feature)
      feature = torch.flatten(feature, 1)
      feature = self.classifier(feature)
      feature = feature.view(feature.size(0), 4096)
      lista.append(feature)
    x = torch.cat(lista, dim=1)  
    return x

  def tempMaxPoolingType(self, x):
    lista = []
    for dimage in range(0, self.seqLen):
      feature = self.convNet(x[dimage])
      feature = self.avgpool(feature)
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
    
    return x
  
  def forward(self, x):
    # if self.joinType == 'cat':
    #   x = self.catType(x)
    #   x = self.linear(x)
    # elif self.joinType == 'tempMaxPool':
    #   x = self.tempMaxPoolingType(x)
    #   x = self.classifier(x)
    #   x = self.linear(x)
    lista = []
    # x = x.permute(1, 0, 2, 3, 4)
    # print('X size: ',x.size())
    for dimage in range(0, self.seqLen):
      feature = self.convNet(x[dimage])
      feature = self.avgpool(feature)
      lista.append(feature)

    minibatch = torch.stack(lista, 0)
    minibatch = minibatch.permute(1, 0, 2, 3, 4)
    # print('minibatch size: ', minibatch.size())
    
    num_dynamic_images = self.seqLen
    tmppool = nn.MaxPool2d((num_dynamic_images, 1))
    lista_minibatch = []
    for idx in range(minibatch.size()[0]):
        out = tempMaxPooling(minibatch[idx], tmppool)
        lista_minibatch.append(out)
        # print('out size: ',out.size())
    feature = torch.stack(lista_minibatch, 0)
    # print('minibatch size: ', feature.size())
    feature = torch.flatten(feature, 1)
    feature = self.classifier(feature)
    # view = feature.view(feature.size(0), 4096)
    # x = torch.cat(lista, dim=1)
    # print('forward view: ',feature.size(), view.size())
    x = self.linear(feature)
    return x