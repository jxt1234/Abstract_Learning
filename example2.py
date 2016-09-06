#!/usr/bin/python
import Renascence

producer = Renascence.init(["./libAbstract_learning.xml"])
print producer.listAllFunctions()
print producer.listAllTypes()
fileName = "/Users/jiangxiaotang/opencv_test/cv_test/train.txt"
x0 = producer.load('ALFloatMatrix', fileName)
y = producer.build('MatrixCrop(x0, x1, x2)').run(producer.merge(x0, 0.0,0.0))
y.save('.origin.txt')

function = producer.build("RandomForest(x0)").run(x0)
#function = producer.build("CreateClassify(CreateSVM(), x0)").run(x0)
#function = producer.build("C45Tree(x0)").run(x0)
function.save('.random')
producer.build('Classify(x0, x1)').run(producer.merge(function, x0)).save('.predict.txt')

