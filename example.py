#!/usr/bin/python
import Renascence

producer = Renascence.init(["./libAbstract_learning.xml"])
print producer.listAllFunctions()
print producer.listAllTypes()
x0 = producer.load('ALFloatMatrix', '../../Data/Face_Dataset/train_yuv.data')
#formula = 'CrossValidate(ADF(GodTrain), Labled(x0, x1))'
#formula = 'C45Tree(x0)'
formula = 'RamdomForest(x0)'

result = producer.build(formula).run(x0)
result.save('./.tree')


