import torch
from torch import nn
from torchvision.models import resnet18,ResNet18_Weights
class Gate(nn.Module):
    def __init__(self,c=512):
        super().__init__(); h=max(c//16,8); self.pool=nn.AdaptiveAvgPool2d(1); self.fc=nn.Sequential(nn.Linear(c,h),nn.ReLU(True),nn.Linear(h,c),nn.Sigmoid())
    def forward(self,x):
        b,c,_,_=x.shape; w=self.fc(self.pool(x).view(b,c)).view(b,c,1,1); return x*w
class OSCARNet(nn.Module):
    def __init__(self,n,pretrained=True):
        super().__init__(); b=resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.features=nn.Sequential(b.conv1,b.bn1,b.relu,b.maxpool,b.layer1,b.layer2,b.layer3,b.layer4)
        self.gate=Gate(); self.pool=nn.AdaptiveAvgPool2d(1); self.dropout=nn.Dropout(.25); self.fc=nn.Linear(512,n)
    def forward(self,x):
        x=self.gate(self.features(x)); return self.fc(self.dropout(self.pool(x).flatten(1)))
def mc_dropout(model):
    for m in model.modules():
        if isinstance(m,nn.Dropout): m.train()
