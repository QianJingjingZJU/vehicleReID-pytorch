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
from MyResNet import resnet50,resnet50_nofc,remove_fc

from PIL import Image
from IPython import embed
from tensorboardX import SummaryWriter
from make_triplet_sample import get_imgiddic,get_img_sortiddic,gettripletsample

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

class Mydatsettriplet(Dataset):
    def __init__(self, img_name, batchsize =32,data_transforms=None, loader = default_loader):
        self.img_name = img_name
        self.batchsize =batchsize
        self.img_a_name = self.img_name[0:self.batchsize:1]
        self.img_p_name = self.img_name[self.batchsize:2 * self.batchsize:1]
        self.img_n_name = self.img_name[2 * self.batchsize:3 * self.batchsize:1]
        self.data_transforms = data_transforms
        self.loader = loader

    def __len__(self):
        return len(self.img_a_name)

    def __getitem__(self, item):
        img_a_name = self.img_a_name[item]
        img_p_name = self.img_p_name[item]
        img_n_name = self.img_n_name[item]
        img_a = self.loader(img_a_name)
        img_p = self.loader(img_p_name)
        img_n = self.loader(img_n_name)

        if self.data_transforms is not None:
            try:
                img_a = self.data_transforms(img_a)
                img_p = self.data_transforms(img_p)
                img_n = self.data_transforms(img_n)
            except:
                print("Cannot transform image: {}".format(img_a))
                print("Cannot transform image: {}".format(img_p))
                print("Cannot transform image: {}".format(img_n))
        return img_a, img_p, img_n

def train_model(model, criterion1,criterion2,optimizer, scheduler, num_epochs, use_gpu, batchnumber=50, batchsize=32):
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
        running_softmaxloss = 0.0
        running_tripletloss = 0.0

        for batch in range(batchnumber):

            imglist, idlist= gettripletsample('/home/csc302/bishe/dataset/VehicleID_V1.0/triplet_train/',
                                                batchsize=batchsize, imgiddic=d,imgsortid=dsort)
            image_datasetsoft = Mydatsetsoft(imglist, idlist,  data_transforms=data_transforms)
            data_loadsoft = torch.utils.data.DataLoader(image_datasetsoft,batch_size=batchsize*3,shuffle=False)

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
                outputa = normalize(feature[0:batchsize:1])
                outputp = normalize(feature[batchsize:2*batchsize:1])
                outputn = normalize(feature[2*batchsize:3*batchsize:1])
                loss2 = criterion2(outputa, outputp, outputn)
                running_tripletloss += loss2.data[0]
                loss = loss1+loss2
                loss.backward()
                optimizer.step()
                running_loss += loss.data[0]


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
                batch_tripletloss = running_tripletloss/(batch+1)
                #batch_acc = running_corrects / (batch_size*count_batch)
                print('Epoch [{}] Batch [{}] Loss: {:.4f} SoftmaxLoss: {:.4f} TripletLoss:{:.4f} Time: {:.4f}s'. \
                        format(epoch, batch+1, batch_loss,batch_softmaxloss, batch_tripletloss, time.time()-begin_time))
                begin_time = time.time()

        model_wtse = model.state_dict()
        model_nofce = resnet50_nofc(pretrained=False)
        model_nofce.load_state_dict(remove_fc(model_wtse))
        epoch_loss = running_loss/batchnumber
        epoch_softmaxloss = running_softmaxloss/batchnumber
        epoch_tripletloss = running_tripletloss/batchnumber
        print('Loss: {:.4f} SoftmaxLoss: {:.4f} TripletLoss: {:.4f}'.format(epoch_loss,epoch_softmaxloss,epoch_tripletloss))

        if not os.path.exists('output'):
            os.makedirs('output')
        torch.save(model_nofce, 'output/resnet_nofc_epoch{}.pkl'.format(epoch))
        writer.add_scalar('epoch_loss', epoch_loss, epoch)
        writer.add_scalar('epoch_softmax_loss', epoch_softmaxloss, epoch)
        writer.add_scalar('epoch_triplet_loss', epoch_tripletloss, epoch)

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

    '''image_datasets = Mydatset(img_path='/ImagePath',
                              txt_path=('/TxtFile/' + 'x' + '.txt'),
                              data_transforms=data_transforms)

    # wrap your data and label into Tensor
    dataloders = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, shuffle=True)

    dataset_sizes =len(image_datasets)'''

    # get model and replace the original fc layer with your fc layer
    model_ft =resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)

    # if use gpu
    if use_gpu:
        model_ft = model_ft.cuda()

    # define cost function
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.TripletMarginLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.005, momentum=0.9)

    # Decay LR by a factor of 0.2 every 5 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.1)

    # multi-GPU
    #model_ft = torch.nn.DataParallel(model_ft, device_ids=[0,1])

    # train model
    model_ft = train_model(model=model_ft,
                           criterion1=criterion1,
                           criterion2=criterion2,
                           optimizer=optimizer_ft,
                           scheduler=exp_lr_scheduler,
                           num_epochs=200,
                           use_gpu=use_gpu,
                           batchnumber=batchnumber,
                           batchsize=batch_size)

    # save best model
    torch.save(model_ft,"output/last_hasfc.pkl")