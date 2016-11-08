#!/usr/bin/python
import Renascence
producer = Renascence.init(["./libAbstract_learning.xml"])

x0 = producer.load('ALFloatMatrix', 'bao.txt')
x1 = producer.load('ALFloatMatrix', './bao_predict.txt')
producer.build('OUTPUT(Transform(x0, Normalizer(x0)), Transform(x1, Normalizer(x0)))').run(producer.merge(x0, x1)).saveAll(['bao_normal.txt', 'bao_predict_normal.txt'])
