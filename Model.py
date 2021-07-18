import torch as th
import torch.nn as nn
import torch.nn.functional as F
import Configure as Con

class network(nn.Module):
    def __init__(self):
        super(network,self).__init__()
        # layerlist = nn.ModuleList()
        layerlist=[];
        self.drop = nn.Dropout(Con.drop)
        In = Con.Inp
        for lay in Con.layers:
            layerlist.append(nn.Conv2d(In,lay,3,1,1))
            layerlist.append(nn.ReLU(True))
            layerlist.append(nn.MaxPool2d(2))
            layerlist.append(nn.BatchNorm2d(lay))
            In =lay
        self.CNN = nn.Sequential(*layerlist)
        self.fc1 = nn.Linear(128,2048)
        print(self.CNN);
        self.fc2 = nn.Linear(2048,10)
    def forward(self,x):
        x = self.CNN(x)
        x =th.flatten(x,1)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x =self.drop(x)
        x = F.log_softmax(self.fc2(x))
        return x


