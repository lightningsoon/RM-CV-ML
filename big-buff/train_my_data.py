'''
三种方案
第一种按他的格式把图片拼接起来
第二种，把数据处理成一样的格式
第三种，用sklearn做训练
[list(20*20ndarray)]
[list]
'''
# 选择第二种
'''
数据集格式
training
  -0
  -1
  ……
  -9
testing
  -0
  ……
  -9
'''
import os
import cv2
import numpy as np
from utils import *
digits=[]
labels = []

def join(fd):
    global Filebox
    return os.path.join(Filebox,fd)
global Filebox
Filebox="F:/code/rm/cnn_mnist/MNIST_data/mnist/training/"
if __name__ == '__main__':

    for adir in map(join,os.listdir(Filebox)):
        Filebox = adir
        for file in map(join,os.listdir(adir)[:5000]):
            # print(file)
            file = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
            digits.append(cv2.resize(file,(20,20),interpolation=cv2.INTER_LINEAR))
            labels.append(int(os.path.basename(adir)))
        Filebox = "F:/code/rm/cnn_mnist/MNIST_data/mnist/training/"
    print('preprocessing...')
    # shuffle digits
    digits,labels=np.asarray(digits),np.asarray(labels)
    rand = np.random.RandomState(321)
    shuffle = rand.permutation(len(digits))
    digits, labels = digits[shuffle], labels[shuffle]

    digits2 = list(map(deskew, digits))
    samples = preprocess_hog(digits2)

    train_n = int(0.9*len(samples))
    # cv2.imshow('test set', mosaic(25, digits[train_n:]))
    digits_train, digits_test = np.split(digits2, [train_n])
    samples_train, samples_test = np.split(samples, [train_n])
    labels_train, labels_test = np.split(labels, [train_n])
    print('training SVM...')
    model = SVM(C=2.67, gamma=5.383)
    model.train(samples_train, labels_train)
    vis = evaluate_model(model, digits_test, samples_test, labels_test)
    # cv2.imshow('SVM test', vis)
    print('saving SVM as "digits_svm2.dat"...')
    model.save('digits_svm2.dat')
    cv2.waitKey(0)