#!/usr/bin/python
def split(fO, prefix_X):
    number = 0
    count = 0
    f = open(prefix_X+'0', 'w')
    for l in fO:
        if count >= 10000:
            count = 0
            number +=1
            f.close()
            f = open(prefix_X+'%d' %number, 'w')
        f.write(l)
        count+=1
    f.close()

prefix_X = './pieces/lenet/trainX/ALFloatMatrix_'
prefix_Y = './pieces/lenet/trainY/ALFloatMatrix_'
X = open('../data/t10k/train_x_normal.txt')
Y = open('../data/t10k/train_y.txt')

split(X, prefix_X)
split(Y, prefix_Y)
