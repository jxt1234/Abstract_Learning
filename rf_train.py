#!/usr/bin/python
import Renascence

producer = Renascence.init(["./libAbstract_learning.xml"])
print producer.listAllFunctions()
print producer.listAllTypes()
x0 = producer.load('ALFloatMatrix', '../../machine_exam/handset/train.txt')
x1 = producer.load('ALFloatMatrix', '../../machine_exam/handset/test.txt')
#formula = 'CrossValidate(ADF(GodTrain), Labled(x0, x1))'
#formula = 'C45Tree(x0)'
result = producer.build('Classify(RandomForest(x0), x1)').run(producer.merge(x0, x1))
result.save('output/handset.y_rf')



