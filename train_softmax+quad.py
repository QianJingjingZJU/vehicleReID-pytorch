
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import  models, transforms

import time
import os
from torch.utils.data import Dataset
from MyResNet import resnet50,resnet50_nofc,remove_fc,remove_fcandbn

from PIL import Image
import numpy as np
from sklearn import preprocessing as pre
from IPython import embed
from tensorboardX import SummaryWriter
from make_triplet_sample import get_imgiddic,get_img_sortiddic,getquandrasample
from loss import hard_example_mining, euclidean_dist
from TripletLoss import TripletLoss
from test_CMC import get_galproset,getfeature,calacc

# use PIL Image to read image
def default_loader(path):
    try:
        img = Image.open(path)
        return img.convert('RGB')
    except:
        print("Cannot read image: {}".format(path))

def normalize(x, axis=-1):
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def calquadloss(feature,tri_loss1,tri_loss2,batchsize):
    def mynorm(delta):
        dist = torch.pow(delta,2).sum(1, keepdim=True)
        dist = dist.clamp(min=1e-12).sqrt()
        return dist
    dp = mynorm(feature[0:batchsize:1]-feature[batchsize:2*batchsize:1])
    dn1 = mynorm(feature[0:batchsize:1]-feature[2*batchsize:3*batchsize:1])
    dn = mynorm(feature[2*batchsize:3*batchsize:1]-feature[3*batchsize:4*batchsize:1])
    loss = tri_loss1(dp,dn1) + tri_loss2(dp,dn)
    return loss

class Mydatsetsoft(Dataset):
    def __init__(self, img_name, img_id,data_transforms=None, loader = default_loader):
        self.img_name = img_name
        self.img_id = img_id
        self.data_transforms = data_transforms
        self.loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        img_id = self.img_id[item]
        img = self.loader(img_name)

        if self.data_transforms is not None:
            try:
                img = self.data_transforms(img)
            except:
                print("Cannot transform image: {}".format(img))
        return img, img_id

def train_model(model,criterion1,criterion2,criterion3,optimizer, scheduler, num_epochs, use_gpu, batchnumber=50, batchsize=32):
    since = time.time()
    writer = SummaryWriter()
    d = get_imgiddic('/home/csc302/bishe/dataset/VehicleID_V1.0/train_50000.txt')
    dsort = get_img_sortiddic('/home/csc302/bishe/dataset/VehicleID_V1.0/train_50000.txt')

    for epoch in range(num_epochs):
        begin_time = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        scheduler.step()
        model.train(True)  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0
        running_softmaxloss = 0.0
        running_quadloss = 0.0

        for batch in range(batchnumber):

            imglist, idlist= getquandrasample('/home/csc302/bishe/dataset/VehicleID_V1.0/triplet_train/',
                                                batchsize=batchsize, imgiddic=d,imgsortid=dsort)
            image_datasetsoft = Mydatsetsoft(imglist, idlist,  data_transforms=data_transforms)
            data_loadsoft = torch.utils.data.DataLoader(image_datasetsoft,batch_size=batchsize*4,shuffle=False)

            for data in data_loadsoft:
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()
                # forward
                feature, outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss1 = criterion1(outputs, labels)
                running_softmaxloss += loss1.data[0]

                #feature_p, feature_n = maketrihardbatch(feature)
                feature = normalize(feature)
                loss2 = calquadloss(feature,criterion2,criterion3,batchsize)
                running_quadloss += loss2.data[0]
                loss = loss1+loss2
                #embed()
                loss.backward()
                optimizer.step()
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds==labels.data)


                #optimizer.zero_grad()
                # forward

                # loss.backward()
                # optimizer.step()
                # running_loss += loss.data[0]
                #embed()

            # print result every 10 batch
            if (batch+1)%30 == 0:
                batch_loss = running_loss/(batch+1)
                batch_softmaxloss = running_softmaxloss/(batch+1)
                batch_quadloss = running_quadloss/(batch+1)
                batch_acc = running_corrects / ((batch+1)*batchsize*4)
                print('Epoch [{}] Batch [{}] Loss: {:.4f} SoftmaxLoss: {:.4f} QuadrupletLoss:{:.4f} Accuracy:{:.4f} Time: {:.4f}s'. \
                        format(epoch, batch+1, batch_loss,batch_softmaxloss, batch_quadloss, batch_acc, time.time()-begin_time))
                begin_time = time.time()

        epoch_loss = running_loss/batchnumber
        epoch_acc = running_corrects/(batchnumber*batchsize*4)
        epoch_softmaxloss = running_softmaxloss/batchnumber
        epoch_quadloss = running_quadloss/batchnumber
        print('Loss: {:.4f} SoftmaxLoss: {:.4f} QuandrupletLoss: {:.4f} Accuracy:{:.4f}'.
              format(epoch_loss,epoch_softmaxloss,epoch_quadloss, epoch_acc))

        if not os.path.exists('output'):
            os.makedirs('output')
        torch.save(model, 'output/resnet_epoch{}.pkl'.format(epoch))
        if (epoch+1)%10 ==0:
            model_wtse = model.state_dict()
            model_nofce = resnet50_nofc(pretrained=False)
            model_nofce.load_state_dict(remove_fc(model_wtse))
            gallery, probe, gdict, pdict = get_galproset('/home/csc302/bishe/dataset/VehicleID_V1.0/train_test_split/test_list_800.txt')
            gallerydict, probedict = getfeature(imgpath='/home/csc302/bishe/dataset/VehicleID_V1.0/test_800/',
                                                model=model_nofce.cuda(), gallery=gallery, probe=probe)
            print(calacc(gallerydict=gallerydict,probedict=probedict,gdict=gdict,pdict=pdict))
        writer.add_scalar('epoch_loss', epoch_loss, epoch)
        writer.add_scalar('epoch_softmax_loss', epoch_softmaxloss, epoch)
        writer.add_scalar('epoch_triphard_loss', epoch_quadloss, epoch)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

    return model

if __name__ == '__main__':

    data_transforms = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    use_gpu = torch.cuda.is_available()

    batch_size = 32
    batchnumber = 600
    num_classes = 5055
    tri_loss1 = TripletLoss(margin=0.6)
    tri_loss2 = TripletLoss(margin=0.3)
    '''image_datasets = Mydatset(img_path='/ImagePath',
                              txt_path=('/TxtFile/' + 'x' + '.txt'),
                              data_transforms=data_transforms)
    # wrap your data and label into Tensor
    dataloders = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, shuffle=True)
    dataset_sizes =len(image_datasets)'''

    # get model and replace the original fc layer with your fc layer
    # model = resnet50_nofc_pre(pretrained=True)
    # model_dict = model.state_dict()
    model_ft = resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    # model_ft_dict = model_ft.state_dict()
    # model_ft_dict.update(model_dict)
    # model_ft.load_state_dict(model_ft_dict)


    # if use gpu
    if use_gpu:
        model_ft = model_ft.cuda()

    # define cost function
    criterion1 = nn.CrossEntropyLoss()
    #criterion2 = nn.TripletMarginLoss(margin=0.6)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)

    # Decay LR by a factor of 0.2 every 5 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.1)

    # multi-GPU
    #model_ft = torch.nn.DataParallel(model_ft, device_ids=[0,1])

    # train model
    model_ft = train_model(model=model_ft,
                           criterion1=criterion1,
                           criterion2=tri_loss1,
                           criterion3=tri_loss2,
                           optimizer=optimizer_ft,
                           scheduler=exp_lr_scheduler,
                           num_epochs=200,
                           use_gpu=use_gpu,
                           batchnumber=batchnumber,
                           batchsize=batch_size)

    # save best model
    torch.save(model_ft,"output/last_hasfc.pkl")