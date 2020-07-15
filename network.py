import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from BASNet import BASNet

class seblock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(seblock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
        
class residualse(nn.Module):
  def __init__(self, channel_num):
    super(residualse, self).__init__()
    self.conv_block1 = nn.Sequential(
			nn.Conv2d(channel_num, channel_num, 3, padding=1),
			nn.BatchNorm2d(channel_num),
			nn.ReLU(),
		) 
    self.conv_block2 = nn.Sequential(
			nn.Conv2d(channel_num, channel_num, 3, padding=1),
			nn.BatchNorm2d(channel_num),
		)
    self.relu = nn.ReLU()
    self.se=seblock(channel_num)
  def forward(self, x):
    x = self.conv_block1(x)
    x = self.conv_block2(x)
    residual = x
    x = self.se(x)
    x = x + residual
    out = self.relu(x)
    return out
    
class ressenet(nn.Module):
  def __init__(self):
    super(ressenet,self).__init__()
    #in block
    
    self.inc1=nn.Conv2d(4,64,kernel_size=3,stride=1,padding=1)
    self.inr1=residualse(64)

    #eblock1
    self.e1c1=nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1)
    self.e1r1=residualse(128)

    #eblock2
    self.e2c1=nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1)
    self.e2r1=residualse(256)

    #bottleneck
    self.br1=residualse(256)
    self.br2=residualse(256)
    self.br3=residualse(256)

    #dblock1
    self.d1r1=residualse(256)
    self.d1c1=nn.Conv2d(256,128,kernel_size=3,stride=1,padding=1)
    self.d1p1=nn.PixelShuffle(2)

    #dblock2
    self.d2r1=residualse(96)#
    self.d2c1=nn.Conv2d(96,64,kernel_size=3,stride=1,padding=1)
    self.d2p1=nn.PixelShuffle(2)

    #outblock
    self.or1=residualse(144)#
    self.oc1=nn.Conv2d(144,3,kernel_size=3,stride=1,padding=1)
    self.tanh=nn.Tanh()

    #pixel-wise
    self.pwc1=nn.Conv2d(64,64,kernel_size=1,stride=2,padding=0)
    self.pwc2=nn.Conv2d(128,128,kernel_size=1,stride=1,padding=0)

    #negative residual?
    
  def forward(self,x,att):
    xin=torch.cat([x,att],1)
    #print(xin.shape)
    in1=self.inr1(self.inc1(xin))
    in1=F.interpolate(in1,scale_factor=0.5)
    e1=self.e1r1(self.e1c1(in1))
    e1=F.interpolate(e1,scale_factor=0.5)
    e2=self.e2r1(self.e2c1(e1))
    b=self.br3(self.br2(self.br1(e2)))
    d1=self.d1p1(self.d1c1(self.d1r1(b)))

    inp=self.pwc1(in1)
    e1p=self.pwc2(e1)

    dc=torch.cat([inp,d1],1)
    d2=self.d2p1(self.d2c1(self.d2r1(dc)))
    d2=F.interpolate(d2,scale_factor=0.5)

    dc2=torch.cat([e1p,d2],1)
    d3=self.oc1(self.or1(dc2))
    d3=F.interpolate(d3,scale_factor=4)
    d3=self.tanh(d3+x)
    return d3

class totalnet(nn.Module):
  def __init__(self):
    super(totalnet,self).__init__()
    self.bas=BASNet()
    #weights=torch.load("/public/zebanghe2/derain/reimplement/residualse/basnet_bsi_itr_100000_train_1.641742_tar_0.068001.pth")
    #preweight=weights.state_dict()
    #self.bas.load_state_dict(preweight)
    #for p in self.parameters():
       #p.requires_grad = False
    self.senet=ressenet()
    #weights2=torch.load("/public/zebanghe2/derain/reimplement/residualse/generator_ 90_ori.pth")
    #print(weights2)
    #prew2=weights2.state_dict()
    #self.senet.load_state_dict(prew2)
  def forward(self,x):
    dout,d1,d2,d3,d4,d5,d6,db=self.bas(x)
    att=F.sigmoid(dout+d1+d2+d3+d4+d5+d6+db)
    att=F.interpolate(att,size=(384,384))
    amean=torch.mean(att)
    att2=att.ge(amean)
    
    trans=self.senet(x,att2)
    return att2,trans,dout,d1,d2,d3,d4,d5,d6,db
    
    
    
    