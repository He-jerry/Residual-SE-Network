#! /usr/bin/env python


import sys
sys.path.insert(0, '/public/zebanghe2/derain/reimplement/residualse/')

import argparse
import os
import numpy as np
import math
import torch
import torch.nn as nn
import torchvision
from dataset import ImageDataset
from tqdm import tqdm
from PIL import Image
from network import ressenet
from BASNet import BASNet
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
from torchvision.utils import save_image
from loss import VGGLoss

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn.functional as F
import pytorch_ssim
import pytorch_iou

bce_loss = nn.BCELoss(size_average=True)
ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)

def bce_ssim_loss(pred,target):

	bce_out = bce_loss(pred,target)
	ssim_out = 1 - ssim_loss(pred,target)
	iou_out = iou_loss(pred,target)

	loss = bce_out + ssim_out + iou_out

	return loss


def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, d7, labels_v):

	loss0 = bce_ssim_loss(d0,labels_v)
	loss1 = bce_ssim_loss(d1,labels_v)
	loss2 = bce_ssim_loss(d2,labels_v)
	loss3 = bce_ssim_loss(d3,labels_v)
	loss4 = bce_ssim_loss(d4,labels_v)
	loss5 = bce_ssim_loss(d5,labels_v)
	loss6 = bce_ssim_loss(d6,labels_v)
	loss7 = bce_ssim_loss(d7,labels_v)
	#ssim0 = 1 - ssim_loss(d0,labels_v)

	# iou0 = iou_loss(d0,labels_v)
	#loss = torch.pow(torch.mean(torch.abs(labels_v-d0)),2)*(5.0*loss0 + loss1 + loss2 + loss3 + loss4 + loss5) #+ 5.0*lossa
	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7#+ 5.0*lossa
	print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.item(),loss1.item(),loss2.item(),loss3.item(),loss4.item(),loss5.item(),loss6.item()))
	# print("BCE: l1:%3f, l2:%3f, l3:%3f, l4:%3f, l5:%3f, la:%3f, all:%3f\n"%(loss1.data[0],loss2.data[0],loss3.data[0],loss4.data[0],loss5.data[0],lossa.data[0],loss.data[0]))

	return loss

criterion_GAN= torch.nn.MSELoss().cuda()
criterion_bce=torch.nn.BCEWithLogitsLoss().cuda()
criterion_vgg=VGGLoss().cuda()

gen1 = BASNet()
gen2=ressenet()
writer = SummaryWriter(log_dir="/public/zebanghe2/derain/reimplement/residualse/modeltest/", comment="ResSENet")

gen1 = gen1.cuda()
gen2 = gen2.cuda()
criterion_GAN.cuda()
criterion_bce.cuda()
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
])
optimizer_G1 = torch.optim.Adam(gen1.parameters(), lr=0.002, betas=(0.5, 0.999))
optimizer_G2 = torch.optim.Adam(gen2.parameters(), lr=0.002, betas=(0.5, 0.999))
#optimizer_T = torch.optim.Adam(netparam, lr=0.00005, betas=(0.5, 0.999))
# Configure dataloaders


trainloader = DataLoader(
    ImageDataset(transforms_=None),
    batch_size=2,
    shuffle=False,drop_last=True
)
print("data length:",len(trainloader))
Tensor = torch.cuda.FloatTensor
eopchnum=50+50+50
print("start training")
totalloss_g1=0
totalloss_g2=0
#gen=torch.load("/public/zebanghe2/derain/reimplement/residualse/modeltest/ressenet_try1_25.pth")
gen1=torch.load("/public/zebanghe2/derain/reimplement/residualse/basnet_try1_3stg_49.pth")
gen2=torch.load("/public/zebanghe2/derain/reimplement/residualse/ressenet_try1_3stg_49.pth")
for epoch in range(51, eopchnum+1):
  print("epoch:",epoch)
  iteration=0
  gen1.train()
  gen2.train()
  #train
  train_iterator = tqdm(trainloader, total=len(trainloader))
  for total in train_iterator:
    
    iteration=iteration+1
    optimizer_G1.zero_grad()
    optimizer_G2.zero_grad()
    # Model inputs
    real_img = total["img"]
    real_trans = total["trans"]
    real_mask = total["mask"]
    real_img=real_img.cuda()
    real_mask=real_mask.cuda()
    real_trans=real_trans.cuda()
    real_img=Variable(real_img,requires_grad=False)
    real_mask=Variable(real_mask,requires_grad=False)
    real_trans=Variable(real_trans,requires_grad=False)
    #print(torch.mean(real_mask))
    if epoch>0 and epoch<=50:
     dout,d1,d2,d3,d4,d5,d6,db=gen1(real_img)
     lossg1=1
     dout=F.sigmoid(dout)
     d1=F.sigmoid(d1)
     d2=F.sigmoid(d2)
     d3=F.sigmoid(d3)
     d4=F.sigmoid(d4)
     d5=F.sigmoid(d5)
     d6=F.sigmoid(d6)
     db=F.sigmoid(db)
     lossg2=muti_bce_loss_fusion(dout,d1,d2,d3,d4,d5,d6,db,real_mask)
     lossg=lossg2
     lossg.backward()
     #for name, weight in gen1.named_parameters():
           #if weight.requires_grad:	
		          #train_iterator.set_description(name,str(parms.requires_grad),str(parms.grad))
             #print("name",name)
             #print(name,weight.grad)
     optimizer_G1.step()
     train_iterator.set_description("batch:%3d,iteration:%3d,loss_g2:%3f,loss_total:%3f"%(epoch+1,iteration,lossg2.item(),lossg.item()))
     
    if epoch>50 and epoch<=100:
     for p in gen1.parameters():
       p.requires_grad=False
     
     real_D=gen2(real_img,real_mask)
     lossg1=criterion_GAN(real_D,real_trans)
     lossg2=1
     lossg=lossg1
     lossg.backward()
     #for name, weight in gen2.named_parameters():
           #if weight.requires_grad:	
		          #train_iterator.set_description(name,str(parms.requires_grad),str(parms.grad))
             #print("name",name)
             #print(name,weight.grad)
     optimizer_G2.step()
     train_iterator.set_description("batch:%3d,iteration:%3d,loss_g1:%3f,loss_total:%3f"%(epoch+1,iteration,lossg1.item(),lossg.item()))
    
    if epoch>100:
     with torch.no_grad():
       dout,d1,d2,d3,d4,d5,d6,db=gen1(real_img.detach())
     outmap=F.sigmoid(dout+d1+d2+d3+d4+d5+d6+db)
     real_D=gen2(real_img,outmap.detach())
     lossg1=criterion_GAN(real_D,real_trans)
     with torch.no_grad():
       qout,q1,q2,q3,q4,q5,q6,qb=gen1(real_D.detach())
     qout=F.sigmoid(qout)
     q1=F.sigmoid(q1)
     q2=F.sigmoid(q2)
     q3=F.sigmoid(q3)
     q4=F.sigmoid(q4)
     q5=F.sigmoid(q5)
     q6=F.sigmoid(q6)
     qb=F.sigmoid(qb)
     lossg2=muti_bce_loss_fusion(qout,q1,q2,q3,q4,q5,q6,qb,real_mask)
     lossg3=criterion_vgg(real_D,real_trans)
     lossg=lossg1+lossg2+lossg3
     #lossg=Variable(lossg,requires_grad=True)
     lossg.backward()
     optimizer_G2.step()
     train_iterator.set_description("batch:%3d,iteration:%3d,loss_g1:%3f,loss_g2:%3f,loss_g3:%3f"%(epoch+1,iteration,lossg1.item(),lossg2.item(),lossg3.item()))
    
    if iteration % 5000 == 0:	
      writer.add_scalar('lossg1', lossg1, iteration)	
      writer.add_scalar('lossg2', lossg2, iteration)
    
    #print("batch:%3d,iteration:%3d,loss_g1:%3f,loss_g2:%3f"%(epoch,iteration,lossg1.item(),lossg2.item()))
    
    del lossg1,lossg2,lossg
  
  if(epoch%10==9):
    torch.save(gen1,"/public/zebanghe2/derain/reimplement/residualse/basnet_try1_3stg_%s.pth"%epoch)
    torch.save(gen2,"/public/zebanghe2/derain/reimplement/residualse/ressenet_try1_3stg_%s.pth"%epoch)



    