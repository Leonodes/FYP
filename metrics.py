import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)


class Gradient_Net(nn.Module):
    def __init__(self):
        super(Gradient_Net, self).__init__()
        kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(device)

        kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(device)

        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    def forward(self, x):
        grad_x = F.conv2d(x, self.weight_x)
        grad_y = F.conv2d(x, self.weight_y)
        gradient = torch.abs(grad_x) + torch.abs(grad_y)
        return gradient

class PCCloss(nn.Module):

    def __init__(self):
        super(PCCloss, self).__init__()

    def forward(self, x, y):
        x = x.view(-1, x.size()[3]*x.size()[2])
        y = y.view(-1, y.size()[3]*y.size()[2])
        x_bar = torch.mean(x, dim=1, keepdim=True)
        y_bar = torch.mean(y, dim=1, keepdim=True)
        vx = (x-x_bar)
        vy = (y-y_bar)
        c = torch.mean(vx*vy, dim=1)/(torch.sqrt(torch.mean(vx**2, dim=1)+1e-08) * torch.sqrt(torch.mean(vy ** 2,dim=1)+1e-08))
        output = torch.mean(-c) # torch.mean(1-c**2)
        return output


class com_loss(nn.Module):
    def __init__(self):
        super(com_loss, self).__init__()

    def forward(self, x, y):
        loss1 = PCCloss()(x, y)
        loss2 = nn.MSELoss()(x, y)
        loss = loss1 + loss2
        return loss#, loss1, loss2 

class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


class com_loss2(nn.Module):
    def __init__(self):
        super(com_loss2, self).__init__()

    def forward(self, x, y):
        loss1 = -mts.SSIM()(x, y)
        loss2 = L1_Charbonnier_loss()(x, y) #L1_Charbonnier_loss()(x, y)#nn.MSELoss()(x, y)
        loss =  0.98*loss1 + 0.02*loss2
        return loss#, loss1, loss2 

class com_loss3(nn.Module):
    def __init__(self):
        super(com_loss3, self).__init__()

    def forward(self, x, y):
        loss1 = PCCloss()(x, y)
        loss4 = L1_Charbonnier_loss()(x, y)
        loss =  loss1 + loss4

        return loss#, loss1, loss2 