"""Conditional residual U-Net and the fixed DDPM noise schedule."""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self,cin,cout,embedding):
        super().__init__()
        self.norm1=nn.GroupNorm(4,cin);self.conv1=nn.Conv2d(cin,cout,3,padding=1)
        self.embed=nn.Linear(embedding,cout)
        self.norm2=nn.GroupNorm(4,cout);self.conv2=nn.Conv2d(cout,cout,3,padding=1)
        self.skip=nn.Identity() if cin==cout else nn.Conv2d(cin,cout,1)

    def forward(self,x,emb):
        h=self.conv1(F.silu(self.norm1(x)))+self.embed(F.silu(emb))[:,:,None,None]
        return self.skip(x)+self.conv2(F.silu(self.norm2(h)))


class Attention(nn.Module):
    def __init__(self,channels):
        super().__init__();self.norm=nn.GroupNorm(4,channels)
        self.attn=nn.MultiheadAttention(channels,4,batch_first=True)

    def forward(self,x):
        b,c,h,w=x.shape;q=self.norm(x).flatten(2).transpose(1,2)
        a,_=self.attn(q,q,q,need_weights=False)
        return x+a.transpose(1,2).reshape(b,c,h,w)


class ConditionalUNet(nn.Module):
    def __init__(self,residual=False,base=36):
        super().__init__();self.residual=residual;self.base=base;embedding=base*4
        self.time_mlp=nn.Sequential(nn.Linear(128,embedding),nn.SiLU(),nn.Linear(embedding,embedding))
        self.design_mlp=nn.Sequential(nn.Linear(4,embedding),nn.SiLU(),nn.Linear(embedding,embedding))
        channels=[base,base*2,base*4,base*4]
        self.input=nn.Conv2d(4 if residual else 2,base,3,padding=1)
        self.down=nn.ModuleList();self.downsample=nn.ModuleList();current=base
        for i,c in enumerate(channels):
            self.down.append(nn.ModuleList([ResBlock(current,c,embedding),ResBlock(c,c,embedding)]));current=c
            if i<3:self.downsample.append(nn.Conv2d(c,c,3,stride=2,padding=1))
        self.mid1=ResBlock(current,current,embedding);self.attn=Attention(current);self.mid2=ResBlock(current,current,embedding)
        self.up=nn.ModuleList();self.upsample=nn.ModuleList()
        for i in reversed(range(4)):
            c=channels[i];self.up.append(nn.ModuleList([ResBlock(current+c,c,embedding),ResBlock(c,c,embedding)]));current=c
            if i>0:self.upsample.append(nn.Conv2d(current,channels[i-1],3,padding=1));current=channels[i-1]
        self.output=nn.Sequential(nn.GroupNorm(4,base),nn.SiLU(),nn.Conv2d(base,2,3,padding=1))
        nn.init.zeros_(self.output[-1].weight);nn.init.zeros_(self.output[-1].bias)

    def forward(self,noisy,t,design,condition=None):
        frequency=torch.exp(-math.log(10000)*torch.arange(64,device=t.device,dtype=torch.float32)/63)
        phase=t.float()[:,None]*frequency[None]
        emb=self.time_mlp(torch.cat([phase.sin(),phase.cos()],dim=1))+self.design_mlp(design)
        if self.residual:
            if condition is None:raise ValueError('Residual diffusion requires the frozen regression prediction')
            noisy=torch.cat([noisy,condition],dim=1)
        h=self.input(noisy);skips=[]
        for i,blocks in enumerate(self.down):
            h=blocks[1](blocks[0](h,emb),emb);skips.append(h)
            if i<3:h=self.downsample[i](h)
        h=self.mid2(self.attn(self.mid1(h,emb)),emb)
        for i,blocks in enumerate(self.up):
            h=torch.cat([h,skips.pop()],dim=1);h=blocks[1](blocks[0](h,emb),emb)
            if i<3:h=self.upsample[i](F.interpolate(h,scale_factor=2,mode='nearest'))
        return self.output(h)


def schedule(device,steps=1000):
    beta=torch.linspace(1e-4,0.02,steps,device=device,dtype=torch.float32)
    return torch.cumprod(1-beta,dim=0)


def masked_noise_loss(prediction,noise,mask):
    valid=(~mask)[:,None]
    return ((prediction.float()-noise.float()).square()*valid).sum()/(valid.sum()*2).clamp_min(1)


def ddim_update(noisy,epsilon,alpha,previous_alpha,mask,clean_transform=None):
    clean=(noisy-(1-alpha).sqrt()*epsilon)/alpha.sqrt()
    if clean_transform is not None:clean=clean_transform(clean)
    updated=previous_alpha.sqrt()*clean+(1-previous_alpha).sqrt()*epsilon
    return updated.masked_fill(mask[:,None],0)
