#!/usr/bin/python
import Renascence
def getPieceProducer():
    producer = Renascence.init(["./libAbstract_learning.xml"])
    p_producer = Renascence.PieceFunctionProducer(producer, ['./mgpfunc.xml'], ['Map-Reduce.xml'])
    sub_p_producer = Renascence.PieceFunctionProducerParallel(p_producer, 'thread')
    return sub_p_producer

producer = getPieceProducer()
print producer

gd = producer.createInput(dataType='ALGradientMethod', path='./pieces/lenet/info/', keyDimesions=[1]);
p = producer.createFunction('ParameterInit(x0)', 'ALGradientMethod').run([gd])
train_x = producer.createInput(dataType='ALFloatMatrix',path='./pieces/lenet/trainX/', keyDimesions=[6])
train_y = producer.createInput(dataType='ALFloatMatrix',path='./pieces/lenet/trainY/', keyDimesions=[6])
merge = producer.createFunction('GDMatrixPrepare(x0,x1,x2)', 'ALFloatMatrix ALFloatMatrix ALGradientMethod').run([train_x, train_y, gd])
print merge

tempP = producer.createFunction('MatrixPlus(GDCompute(x0,x1,x2))', 'ALFloatMatrix ALGradientMethod ALFloatMatrix').run([merge, gd, p])
#tempP = producer.createFunction('GDCompute(x0,x1,x2)', 'ALFloatMatrix ALGradientMethod ALFloatMatrix').run([merge, gd, p])
print tempP


outputDetP = producer.createOutput('output/pieces/detP')
producer.copyPiece(tempP, outputDetP)

