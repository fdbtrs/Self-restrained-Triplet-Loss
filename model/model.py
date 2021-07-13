import torch.nn as nn
import torch

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output


class LinearHeader(torch.nn.Module):
 """ Linear Header class"""

 def __init__(self, in_features, out_features):
  super(LinearHeader, self).__init__()

  self.in_features = in_features
  self.out_features = out_features

  self.linear = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=False)

 def forward(self, input, label):
  return self.linear(input)





class SingleLayerModel(nn.Module):
 def __init__(self, embedding_size=512):
  super(SingleLayerModel, self).__init__()
  self.fc1 = nn.Linear(embedding_size, embedding_size)
  self.actvation1= nn.LeakyReLU(negative_slope=0.2)
  self.bn1 = nn.BatchNorm1d(embedding_size)
  self.dropout1 = nn.Dropout(0.1)

  self.fc2 = nn.Linear(embedding_size, embedding_size)
  self.actvation2=nn.LeakyReLU(negative_slope=0.2)
  self.bn2 = nn.BatchNorm1d(embedding_size)
  self.dropout2 = nn.Dropout(0.1)

  self.fc3 = nn.Linear(embedding_size, embedding_size)
  self.actvation3 = nn.LeakyReLU(negative_slope=0.2)
  self.bn3 = nn.BatchNorm1d(embedding_size)
  self.dropout3 = nn.Dropout(0.1)


  self.fc4 = nn.Linear(embedding_size, embedding_size)
  self.bn4 = nn.BatchNorm1d(embedding_size)
 def forward(self, x):
  x = self.fc1(x)
  x = self.bn1(x)
  x=self.actvation1(x)

  x = self.fc2(x)
  x = self.bn2(x)
  x=self.actvation2(x)

  x = self.fc3(x)
  x = self.bn3(x)
  x=self.actvation3(x)

  x = self.fc4(x)
  x = self.bn4(x)
  return x

