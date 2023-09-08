from __future__ import print_function
import numpy as np
#import cupy as cp
from complex import ComplexNet
import scipy.io as sio
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as D
import time
from torch.autograd import Variable
import h5py
import torch.nn.functional as F
from folderloading_exp import SPKFolder
import metrics as mts
import matplotlib.pyplot as plt


def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight.data)
        #nn.init.constant_(m.bias.data, 0)

    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data)
        #nn.init.constant_(m.bias.data, 0)

def MAT_dataloading(bit, BATCH_SIZE, root_path):

    trainpath = root_path + str(bit) + '_train.mat'
    train_matrix = h5py.File(trainpath)
    trainset_x = torch.from_numpy(train_matrix['Speckle'][:].astype(float)).float().permute(0, 2, 1)
    trainset_y = torch.from_numpy(train_matrix['image'][:].astype(float)).float().permute(0, 2, 1)
    del train_matrix
    #trainset_x = torch.sqrt(trainset_x.unsqueeze(1))
    #trainset_y = F.avg_pool2d(trainset_y.unsqueeze(1)/255,image_down)

    testpath =  root_path + str(bit) + '_test.mat'
    test_matrix = h5py.File(testpath)
    valset_x = torch.from_numpy(test_matrix['Speckle'][:].astype(float)).float().permute(0, 2, 1)
    valset_y = torch.from_numpy(test_matrix['image'][:].astype(float)).float().permute(0, 2, 1)
    del test_matrix
    #valset_x = torch.sqrt(valset_x.unsqueeze(1))
    #valset_y = F.avg_pool2d(valset_y.unsqueeze(1)/255,image_down)

    #train_dataset = D.TensorDataset(torch.sqrt(trainset_x.unsqueeze(1)),F.avg_pool2d(trainset_y.unsqueeze(1)/255,image_down)) # F.interpolate(train_y.unsqueeze(1),scale_factor=(2,2))
    num_train = len(trainset_y)
    train_loader = D.DataLoader(
        dataset=D.TensorDataset(trainset_x.unsqueeze(1)/65535,trainset_y.unsqueeze(1)/255),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    print('size of training set:', num_train)

    #val_dataset = D.TensorDataset(torch.sqrt(valset_x.unsqueeze(1)), F.avg_pool2d(valset_y.unsqueeze(1)/255,image_down)) #F.interpolate(test_y.unsqueeze(1),scale_factor=(2,2))
    num_val = len(valset_y)
    val_loader = D.DataLoader(
        dataset=D.TensorDataset(valset_x.unsqueeze(1)/65535, valset_y.unsqueeze(1)/255),
        batch_size=1024,
        shuffle=False,
        num_workers=0,
    )
    print('size of test set:', num_val)

     
    return train_loader, val_loader, num_train, num_val

def splitloading(bit, root_path, BATCH_SIZE,split_rate=0.9):

    print('====> loading datasets')
    datapath = root_path
    full_dataset = SPKFolder(datapath)
    data_num = len(full_dataset)
    train_dataset, val_dataset = D.random_split(
        full_dataset, 
        [np.int(data_num*split_rate), data_num-np.int(data_num*(split_rate))], 
        generator=torch.Generator().manual_seed(42))
    '''
    train_dataset = D.Subset(full_dataset, range(8192))
    val_dataset = D.Subset(full_dataset, range(8192, 10000))
        '''

    num_train = len(train_dataset)
    train_loader = D.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=2, #; 0-76s;1-58; 2-59s; 4-62s; 8-66s; 16-75s
    )
    print('size of training set:', num_train)
    num_val = len(val_dataset)
    val_batch = 32
    val_loader = D.DataLoader(
        dataset=val_dataset,
        batch_size=val_batch,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )
    print('size of test set:', num_val)

    return train_loader, val_loader, num_train, num_val


def scaling(x, scale, order, up='False'):
    x = F.avg_pool2d(x, int(scale/order))
    if up == 'True':
        x = F.interpolate(x, int(scale))
    elif up == 'False':
        x = x
    return x

def test():
    
    for epoch in range(EPOCH):

        net.eval()
        loss_test = []
        test_pcc = []
        test_mse = []
        test_ssim = []
        with torch.no_grad():
            for vspeckles, vimages in val_loader:
            #for vspeckles, vimages in val_loader:
                val_x = torch.Tensor(vspeckles[:,:,:scale,:scale]).to(device)
                #val_x = scaling(val_x, scale, order, up='True')
                val_x = torch.sqrt(val_x)
                val_y = torch.Tensor(vimages).to(device)
                val_y = F.avg_pool2d(val_y,image_down)
                vpredict = net(val_x).detach()


                loss_test.append(loss_func(vpredict, val_y).item())
                loss_test_mean = np.mean(loss_test)

                test_pcc.append(mts.PCCloss()(vpredict, val_y).item())
                test_mse.append(nn.MSELoss()(vpredict, val_y).item())
                test_ssim.append(mts.SSIM()(vpredict, val_y).item())
                tpcc = np.mean(test_pcc)
                tmse = np.mean(test_mse)
                tssim = np.mean(test_ssim)

                print('Epoch {} |Test Loss: {:.4f}|PCC: {:.4f}|MSE: {:.4f}|SSIM: {:.4f}|'.format(
                    epoch+1, loss_test_mean, tpcc, tmse, tssim)
                )

                f.write('Epoch {} |Test Loss: {:.4f}|PCC: {:.4f}|MSE: {:.4f}|SSIM: {:.4f}|\n'.format(
                    epoch+1, loss_test_mean, tpcc, tmse, tssim)
                )
                f.flush()


                num = 8
                rows = 3
                sca=1
                plt.subplots(rows,num,figsize=(num*sca,rows*sca))

                for i in range(num):
                    plt.subplot(rows,num,i+1)
                    im = val_y[i,0].cpu().numpy();plt.imshow(im); plt.axis('off')
                    plt.subplot(rows,num,i+1 + num)
                    im = val_x[i,0].cpu().numpy();plt.imshow(im); plt.axis('off')
                    plt.subplot(rows,num,i+1 + num*2)
                    im = vpredict[i,0].cpu().numpy();plt.imshow(im); plt.axis('off')

                plt.tight_layout()
                plt.show()


if __name__ == '__main__':
    bit = 16
    image_down = 4
    scale = 252
    #out_dim = int(128/image_down)
    out_dim = (int(128/image_down), int(128/image_down))
    Dataname = 'FFHQ'
    version = '_vexp0326_1_w0'
    name = './TEST' + str(scale) + Dataname + 'OD' + str(out_dim) + str(bit) + version
    f = open(name + '.txt', 'w+')
    pre_path = '252FFHQOD(32, 32)16_ref4_vexp0326_1_w0.pkl'
    pretrained = True
    EPOCH = 300
    LR = 0.001 #0.001
    BATCH_SIZE = 512
    weight_decay = 0#0.001#0.001 # 0.003
    torch.cuda.set_device(0)
    dev_name = torch.cuda.get_device_name(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root_path = './dataset'

    f.write('device is '  + str(device) + '(' + dev_name + ') \n')
    f.write('datasetpath is '  + str(root_path) + '\n')
    f.write('batch size '  + str(BATCH_SIZE) + '\n')
    f.write('Number of epoch '  + str(EPOCH) + '\n')
    f.write('Initial learning rate '  + str(LR) + '\n')
    f.write('weight decay '  + str(weight_decay) + '\n')
    f.flush()
    

    print('====> loading datasets')
    _, val_loader, _, num_val = splitloading(bit, root_path=root_path, BATCH_SIZE=BATCH_SIZE)

    print('====> loading model: in_dim(' + str(scale) + ') out_dim('+ str(out_dim) +')')
    net = ComplexNet(in_dim=scale, out_dim=out_dim).to(device)
    scaler = torch.cuda.amp.GradScaler()

    if pretrained == True:
        para_recon = torch.load(pre_path)
        net.load_state_dict(para_recon, strict=False)
        f.write('pre-train path: '  + str(pre_path) + '\n')
        f.flush()
    else:
        None

    optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=weight_decay, nesterov=True)

    loss_func = mts.com_loss()

    torch.backends.cudnn.benchmark = True
    torch.cuda.synchronize()
    torch.cuda.init()
    torch.backends.cudnn.deterministic = True
    

    test()

    f.close()
