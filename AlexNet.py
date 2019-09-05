import torch.nn as nn
from torchvision import models
from util import *
from tempPooling import *
import torch


class ViolenceModelAlexNetV1(nn.Module): ##ViolenceModel
  def __init__(self, seqLen, joinType,feature_extract):
      super(ViolenceModelAlexNetV1, self).__init__()
      self.seqLen = seqLen
      self.joinType = joinType
      self.alexnet = models.alexnet(pretrained=True)
      self.feature_extract = feature_extract
      set_parameter_requires_grad(self.alexnet, feature_extract)
      
      self.convNet = nn.Sequential(*list(self.alexnet.features.children()))

      if self.joinType == 'cat':
        self.linear = nn.Linear(256 * 6 * 6*seqLen,2)
      elif self.joinType == 'tempMaxPool':
        self.linear = nn.Linear(256 * 6 * 6,2)
      self.alexnet = None

      # self.linear = nn.Linear(256*6*6*seqLen,2)
      # self.alexnet = None

  def forward(self, x):
    if self.joinType == 'cat':
      x = self.getFeatureVectorCat(x)  
    elif self.joinType == 'tempMaxPool':
      x = self.getFeatureVectorTempPool(x)  

    # lista = []
    # for dimage in range(0, self.seqLen):
    #   feature = self.convNet(x[dimage])
    #   feature = feature.view(feature.size(0), 256 * 6 * 6)
    #   lista.append(feature)
    # x = torch.cat(lista, dim=1)  
    x = self.linear(x)
    return x
  
  def getFeatureVectorTempPool(self, x):
    lista = []
    for dimage in range(0, self.seqLen):
      feature = self.convNet(x[dimage])
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
      feature = self.convNet(x[dimage])
      feature = feature.view(feature.size(0), 256 * 6 * 6)
      lista.append(feature)
    x = torch.cat(lista, dim=1) 
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
  
  def forward(self, x):
    if self.joinType == 'cat':
      x = self.getFeatureVectorCat(x)
    elif self.joinType == 'tempMaxPool':
      x = self.getFeatureVectorTempPool(x)
    
    x = self.linear(x)
    return x
  
  def getFeatureVectorCat(self, x):
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

  def getFeatureVectorTempPool(self, x):
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
    feature = self.classifier(feature)
    return feature