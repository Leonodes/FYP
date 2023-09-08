import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear, ComplexDropout
from complexFunctions import complex_relu, complex_max_pool2d

class ComplexNet32(nn.Module):
    
    def __init__(self):
        super(ComplexNet32, self).__init__()
        self.fc1 = ComplexLinear(256*256, 32*32)
             
    def forward(self,x):
        xr = x
        # imaginary part to zero
        xi = torch.zeros(xr.shape, dtype = xr.dtype, device = xr.device)
                
        xr = xr.view(-1, 256*256)
        xi = xi.view(-1, 256*256)

        xr,xi = self.fc1(xr,xi)

        # take the absolute value as output
        x = torch.sqrt(torch.pow(xr,2)+torch.pow(xi,2))
        # x = torch.pow(xr,2)+torch.pow(xi,2)
        return x.view(-1, 1, 32,32)

class ComplexNet64(nn.Module):
    
    def __init__(self):
        super(ComplexNet64, self).__init__()
        self.fc1 = ComplexLinear(256*256, 64*64)
             
    def forward(self,x):
        xr = x
        # imaginary part to zero
        xi = torch.zeros(xr.shape, dtype = xr.dtype, device = xr.device)
                
        xr = xr.view(-1, 256*256)
        xi = xi.view(-1, 256*256)

        xr,xi = self.fc1(xr,xi)

        # take the absolute value as output
        x = torch.sqrt(torch.pow(xr,2)+torch.pow(xi,2))
        # x = torch.pow(xr,2)+torch.pow(xi,2)
        return x.view(-1, 1, 64,64)

class ComplexNet256_28(nn.Module):
    
    def __init__(self):
        super(ComplexNet256_28, self).__init__()
        self.fc1 = ComplexLinear(256*256, 28*28)
             
    def forward(self,x):
        xr = x
        # imaginary part to zero
        xi = torch.zeros(xr.shape, dtype = xr.dtype, device = xr.device)
                
        xr = xr.view(-1, 256*256)
        xi = xi.view(-1, 256*256)

        xr,xi = self.fc1(xr,xi)

        # take the absolute value as output
        x = torch.sqrt(torch.pow(xr,2)+torch.pow(xi,2))
        # x = torch.pow(xr,2)+torch.pow(xi,2)
        return x.view(-1, 1, 28,28)

class ComplexNet252(nn.Module):
    
    def __init__(self):
        super(ComplexNet252, self).__init__()
        self.fc1 = ComplexLinear(252*252, 32*32)
             
    def forward(self,x):
        xr = x
        # imaginary part to zero
        xi = torch.zeros(xr.shape, dtype = xr.dtype, device = xr.device)
                
        xr = xr.view(-1, 252*252)
        xi = xi.view(-1, 252*252)

        xr,xi = self.fc1(xr,xi)

        # take the absolute value as output
        x = torch.sqrt(torch.pow(xr,2)+torch.pow(xi,2))
        # x = torch.pow(xr,2)+torch.pow(xi,2)
        return x.view(-1, 1, 32,32)

class ComplexNet(nn.Module):
    
    def __init__(self, in_dim, out_dim):
        super(ComplexNet, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc1 = ComplexLinear(self.in_dim*self.in_dim, self.out_dim[0]*self.out_dim[1])
        self.dropout = ComplexDropout()
        
    def forward(self,x):
        xr = x
        # imaginary part to zero
        xi = torch.zeros(xr.shape, dtype = xr.dtype, device = xr.device)
                
        xr = xr.view(-1, self.in_dim*self.in_dim)
        xi = xi.view(-1, self.in_dim*self.in_dim)
        xr,xi = self.fc1(xr,xi)
        xr,xi = self.dropout(xr,xi)

        # take the absolute value as output
        x = torch.sqrt(torch.pow(xr,2)+torch.pow(xi,2))
        # x = torch.pow(xr,2)+torch.pow(xi,2)
        return x.view(-1, 1, self.out_dim[0], self.out_dim[1])

# ComplexNet = ComplexNet(252, (int(128/4), int(128/4))).to('cuda')
# print(ComplexNet)

class ComplexNet1(nn.Module):
    
    def __init__(self, in_dim, out_dim):
        super(ComplexNet1, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc1 = ComplexLinear(self.in_dim*self.in_dim, self.out_dim*self.out_dim)
             
    def forward(self,x):
        xr = x
        # imaginary part to zero
        xi = torch.zeros(xr.shape, dtype = xr.dtype, device = xr.device)
                
        xr = xr.view(-1, self.in_dim*self.in_dim)
        xi = xi.view(-1, self.in_dim*self.in_dim)

        xr,xi = self.fc1(xr,xi)

        # take the absolute value as output
        x = torch.sqrt(torch.pow(xr,2)+torch.pow(xi,2))
        # x = torch.pow(xr,2)+torch.pow(xi,2)
        return x.view(-1, 1, self.out_dim, self.out_dim)