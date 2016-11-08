#!/usr/bin/python
import Renascence
producer = Renascence.init(["./libAbstract_learning.xml"])

x0 = producer.load('ALFloatMatrix', 'bao.txt')
producer.build('Transform(x0, Normalizer(x0))').run(x0).save('bao_normal.txt')
