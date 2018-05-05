import os
import shutil
from random import choice

os.chdir('/Users/CarolQian/Documents/bishe/VehicleID_V1.0/train_test_split/')

f = open("train_50000.txt")
line = f.readline()                 # 调用文件的 readline()方法
count = 1
d = {}
while line:
    data = line.split()             # 分离两个参数
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
print(count, len(d))

pathA = '/Users/CarolQian/Documents/bishe/VehicleID_V1.0/image/'
pathB = '/Users/CarolQian/Documents/bishe/VehicleID_V1.0/train_50000/'
shutil.rmtree(pathB)
os.mkdir(pathB)
oldname = os.listdir(pathA)
for i in d:
    listim = d[i]
    for j in listim:
        if oldname.__contains__(j + ".jpg"):
            oldim = pathA + j + ".jpg"
            newim = pathB + j + ".jpg"
            shutil.copyfile(oldim, newim)
        else:
            pass
