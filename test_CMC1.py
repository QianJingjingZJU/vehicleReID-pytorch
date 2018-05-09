import random
import os

from PIL import Image
from torchvision import  transforms
from torch.autograd import Variable
import torch

from make_triplet_sample import get_imgiddic
from MyResNet import resnet50_nofc, remove_fc
from IPython import embed
import numpy as np
import pickle
from scipy.spatial.distance import cdist
from sklearn import preprocessing as pre
from re_ranking import  re_ranking

def default_loader(path):
    try:
        img = Image.open(path)
        return img.convert('RGB')
    except:
        print("Cannot read image: {}".format(path))

def normalize(x, axis=-1):
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def dict2feature(featDict,imageDict):
    subFeatDict = {}
    labelList = []
    featList = []
    for key in imageDict.keys():
        for img in imageDict[key]:
            labelList.append(int(key))
            featList.append(featDict[img][0])
    return np.array(labelList), np.array(featList)

def getAllImamgeInfo(txtpath):
    imageInfo = get_imgiddic(txtpath)
    imageList = []
    for key in imageInfo.keys():
        imageList.extend(imageInfo[key])
    return imageInfo, imageList


def getGalleryProbe(imageInfo):
    probeDict = {}
    gallerDict = {}
    for key in imageInfo.keys():
        tempList = imageInfo[key].copy()
        ind = np.random.choice(range(len(tempList)),1)[0]
        gallerDict[key] = [tempList.pop(ind)]
        probeDict[key] = tempList.copy()
    return probeDict, gallerDict

def getfeature(imgpath,model,imageList):
    data_transforms = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    featDict = {}
    count =0
    for i in imageList:
        img = default_loader(os.path.join(imgpath, str(i).__add__('.jpg')))
        img = data_transforms(img)
        img.resize_(1,3,224,224)
        featDict[i] = (model(Variable(img).cuda())).cpu().data.numpy()
        count +=1
        if count%1000 == 0:
            print(count)
    print(count+1)
    print('feature finished!')

    return featDict

def calacc(featDict, probeDict, galleryDict, norm_flag=False, rerank=False, top_num=[1,5,10,20,30]):
    probeLabel, probeFeat = dict2feature(featDict,probeDict)
    galleryLabel, galleryFeat = dict2feature(featDict,galleryDict)
    if norm_flag:
        probeFeat = pre.normalize(probeFeat,axis=1)
        galleryFeat = pre.normalize(galleryFeat,axis=1)
    if rerank:
        dist = re_ranking(probeFeat,galleryFeat,k1=10,k2=6,lambda_value=0.3)
    else:
        dist = cdist(probeFeat, galleryFeat)
    index = []
    for i in range(len(dist)):
        a = dist[i]
        ind = np.where(galleryLabel == probeLabel[i])
        dp = a[ind]
        a.sort()
        index.append(list(a).index(dp))

    index = np.array(index)
    cmc = lambda top,index : len(np.where(index<top)[0])/len(probeLabel)
    cmc_curve = [cmc(top, index) for top in top_num]
    return cmc_curve

def cmc(model,test_num=10, norm_flag = True, rerank = False):
    test_num = 10
    cmc_result = []
    featDict = getfeature(imgpath='/home/csc302/bishe/dataset/VehicleID_V1.0/test_800/',model=model, imageList=imageList)

    # f1 = open('gdict.txt', 'wb')
    # pickle.dump(featDict, f1)
    # f1 = open('gdict.txt', 'rb')
    # featDict = pickle.load(f1)
    # f1.close()
    for i in range(test_num):

        probeDict, galleryDict = getGalleryProbe(imageInfo)
        cmc_result.append(calacc(featDict,probeDict,galleryDict,norm_flag,rerank))
        print('test time: ', i)
    cmc_mean = np.mean(np.array(cmc_result),axis=0)
    return  cmc_mean

if __name__ == "__main__":

    imageInfo, imageList = getAllImamgeInfo('/home/csc302/bishe/dataset/VehicleID_V1.0/train_test_split/test_list_800.txt')
    modelfc = torch.load('/home/csc302/bishe/代码/VehicleReID/output-triphard+softmax104/resnet_nofc_epoch104.pkl')
    wts = modelfc.state_dict()
    model = resnet50_nofc(pretrained=False)
    model.load_state_dict(remove_fc(wts))
    model = model.cuda()
    cmc_mean = cmc(model, test_num=10, norm_flag=True, rerank=False)
    print(cmc_mean)


















