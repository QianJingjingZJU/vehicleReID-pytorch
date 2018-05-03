import random
import os

from PIL import Image
from torchvision import  transforms
from torch.autograd import Variable
import torch

from make_triplet_sample import get_imgiddic
from MyResNet import resnet50_nofc
from IPython import embed
import numpy as np
import pickle
from sklearn import preprocessing as pre

def default_loader(path):
    try:
        img = Image.open(path)
        return img.convert('RGB')
    except:
        print("Cannot read image: {}".format(path))

def normalize(x, axis=-1):
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def get_galproset(txtpath):
    d = get_imgiddic(txtpath)
    ID = list(d.keys())
    gallery = []
    probe = []
    gdict = {}
    pdict = {}

    for i in ID:
        image = random.choice(d[i])
        gdict[image] = i
        gallery.append(image)
        for j in d[i]:
            if j != image:
                pdict[j] = i
                probe.append(j)
    print(len(gallery), len(probe))
    return gallery, probe, gdict, pdict

def getfeature(imgpath,model,gallery=[], probe=[]):
    gallerydict = {}
    probedict = {}
    count =0
    for i in gallery:
        img = default_loader(os.path.join(imgpath, str(i).__add__('.jpg')))
        img = data_transforms(img)
        img.resize_(1,3,224,224)
        gallerydict[i] = (model(Variable(img).cuda())).cpu().data.numpy()
        count +=1
        if count%100 == 0:
            print(count)
    print('gallery finished!')
    count =0
    for pr in probe:
        pimg = default_loader(os.path.join(imgpath, str(pr).__add__('.jpg')))
        pimg = data_transforms(pimg)
        pimg.resize_(1,3,224,224)
        probedict[pr] = (model(Variable(pimg).cuda())).cpu().data.numpy()
        count += 1
        if count % 100 == 0:
            print(count)
    print('probe finished!')

    return gallerydict, probedict

def calacc(gallerydict={}, probedict={},gdict=[],pdict=[],toprank=10):
    total = len(pdict)
    print(gdict)
    print(pdict)
    topright = 0

    for i in probedict:
        probeid = pdict[i]
        thegallery = gallerydict.copy()
        for j in thegallery:
            thegallery[j] = np.linalg.norm(pre.normalize(probedict[i])-pre.normalize(thegallery[j]))
        print(thegallery)
        sortgallery = sorted(thegallery.items(), key=lambda x: x[1])
        print(sortgallery)
        sortimgnum = []
        for i in sortgallery:
            sortimgnum.append(i[0])
        count = 0
        for num in sortimgnum:
            sortimgnum[count] = gdict[num]
            count = count+1
        for rank in range(toprank):
            if sortimgnum[rank] == probeid:
                topright = topright+1
                break
        print(topright)
        embed()
    accuracy = topright/total

    return accuracy

if __name__ == "__main__":
    data_transforms = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    '''model = torch.load('/home/csc302/bishe/代码/VehicleReID/output-Triplet-final/resnet_epoch180.pkl')
    model = model.cuda()
    gallery, probe, gdict, pdict = get_galproset('/home/csc302/bishe/dataset/VehicleID_V1.0/train_test_split/test_list_800.txt')
    gallerydict, probedict = getfeature(imgpath='/home/csc302/bishe/dataset/VehicleID_V1.0/test_800/',
                                        model=model, gallery=gallery, probe=probe)
    f1 = open('gdict.txt', 'wb')
    pickle.dump(gdict, f1)
    f1.close()
    f2 = open('pdict.txt', 'wb')
    pickle.dump(pdict, f2)
    f2.close()
    f3 = open('gallerydict.txt', 'wb')
    pickle.dump(gallerydict, f3)
    f3.close()
    f4 = open('probedict.txt', 'wb')
    pickle.dump(probedict, f4)
    f4.close()
    print(gdict)
    print(pdict)
    print(gallerydict)
    print(probedict)'''

    f1 = open('gdict.txt', 'rb')
    gdict = pickle.load(f1)
    f1.close()
    f2 = open('pdict.txt', 'rb')
    pdict = pickle.load(f2)
    f2.close()
    f3 = open('gallerydict.txt', 'rb')
    gallerydict = pickle.load(f3)
    f3.close()
    f4 = open('probedict.txt', 'rb')
    probedict = pickle.load(f4)
    f4.close()
    acc = calacc(gallerydict=gallerydict, probedict=probedict, gdict=gdict,pdict=pdict,toprank=50)

    print(acc)















