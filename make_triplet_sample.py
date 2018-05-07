import os
import shutil
import random
import numpy as np

def sortid(train_id):
    train_sortid = list(set(train_id))
    train_sortid.sort()
    i=0
    for x in train_id:
        train_id[i] = train_sortid.index(x)
        i = i+1
    return train_id

def get_imgiddic(path):
    f=open(path)
    line = f.readline()
    count = 1
    d = {}
    while line:
        data = line.split()
        if d.__contains__(data[1]):
            l1 = d[data[1]]
            l1.append(data[0])
        else:
            l2 = []
            l2.append(data[0])
            d[data[1]] = l2
        line = f.readline()
        count = count+1
    f.close()
    return d

def get_img_sortiddic(path):
    d = {}
    f = open(path)
    lines = f.readlines()
    img_name = [line.strip().split(' ')[0] for line in lines]
    img_label = [int(line.strip().split(' ')[-1]) for line in lines]
    img_label = sortid(img_label)
    count =0
    for i in img_name:
        d[i] = img_label[count]
        count =count+1
    return d

def gettripletsample(imgpath,batchsize=32,imgiddic={},imgsortid={}):
    img_name_a=[]
    a_id =[]
    img_name_p=[]
    p_id =[]
    img_name_n=[]
    n_id =[]
    id = list(imgiddic.keys())

    for batch in range(batchsize):
        batchid = id.copy()
        anchorid = random.choice(batchid)
        batchid.remove(anchorid)
        ap = list(random.sample(imgiddic[anchorid],2))
        img_name_a.append(os.path.join(imgpath, str(ap[0]).__add__('.jpg')))
        a_id.append(imgsortid[ap[0]])
        img_name_p.append(os.path.join(imgpath, str(ap[1]).__add__('.jpg')))
        p_id.append(imgsortid[ap[1]])
        negid =random.choice(batchid)
        negnumber = random.choice(imgiddic[negid])
        img_name_n.append(os.path.join(imgpath, str(negnumber).__add__('.jpg')))
        n_id.append(imgsortid[negnumber])

    img = img_name_a+img_name_p+img_name_n
    id = a_id+p_id+n_id
    return img, id

def deleteless4id(path):
    imgiddic = get_imgiddic(path)
    idkey  = list(imgiddic.keys())
    for i in idkey:
        if len(imgiddic[i]) < 4:
            idkey.remove(i)
    newimgiddic = {}
    newimgsortid = {}
    for i in idkey:
        if imgiddic.__contains__(i):
            newimgiddic[i] = imgiddic[i]
    c = 0
    for i in idkey:
        idkey[c] = int(i)
        c +=1
    idkey = sortid(idkey)
    count = 0
    for vid in newimgiddic:
        for img in newimgiddic[vid]:
            newimgsortid[img] = idkey[count]
        count +=1

    return newimgiddic, newimgsortid


def gettriphardsample(imgpath,batchsize=32,imgiddic={},imgsortid={}):
    img = []
    id = []
    vehicleid = list(imgiddic.keys())
    batchid = vehicleid.copy()

    for batch in range(batchsize):
        anchorid = random.choice(batchid)
        batchid.remove(anchorid)
        if len(imgiddic[anchorid]) > 3:
            ap = random.sample(imgiddic[anchorid],4)
            for i in range(4):
                img.append(os.path.join(imgpath, str(ap[i]).__add__('.jpg')))
                id.append(imgsortid[ap[i]])
        else:
            for i in range(4):
                imgnum = random.choice(imgiddic[anchorid])
                img.append(os.path.join(imgpath, str(imgnum).__add__('.jpg')))
                id.append(imgsortid[imgnum])

    return img,id


def getquandrasample(imgpath,batchsize=32,imgiddic={}, imgsortid={}):
    img_name_a = []
    a_id = []
    img_name_p = []
    p_id = []
    img_name_n1 = []
    n1_id = []
    img_name_n2 = []
    n2_id = []
    id = list(imgiddic.keys())

    for batch in range(batchsize):
        batchid = id.copy()
        anchorid = random.choice(batchid)
        batchid.remove(anchorid)
        ap = list(random.sample(imgiddic[anchorid], 2))
        img_name_a.append(os.path.join(imgpath, str(ap[0]).__add__('.jpg')))
        a_id.append(imgsortid[ap[0]])
        img_name_p.append(os.path.join(imgpath, str(ap[1]).__add__('.jpg')))
        p_id.append(imgsortid[ap[1]])
        negid1 = random.choice(batchid)
        batchid.remove(negid1)
        negnumber1 = random.choice(imgiddic[negid1])
        img_name_n1.append(os.path.join(imgpath, str(negnumber1).__add__('.jpg')))
        n1_id.append(imgsortid[negnumber1])
        negid2 = random.choice(batchid)
        negnumber2 = random.choice(imgiddic[negid2])
        img_name_n2.append(os.path.join(imgpath, str(negnumber2).__add__('.jpg')))
        n2_id.append(imgsortid[negnumber2])


    img = img_name_a + img_name_p + img_name_n1 + img_name_n2
    id = a_id + p_id + n1_id + n2_id
    return img, id



imgiddic = get_imgiddic('/Users/CarolQian/Documents/bishe/VehicleID_V1.0/train_test_split/test_list_800.txt')
imgsortid = get_img_sortiddic('/Users/CarolQian/Documents/bishe/VehicleID_V1.0/train_test_split/test_list_800.txt')
img,imgid = getquandrasample(imgpath='/Users/CarolQian/Documents/bishe/VehicleID_V1.0/test_800/',
                             batchsize=4,imgiddic=imgiddic,imgsortid=imgsortid)
print(len(img),img)
print(len(imgid),imgid)