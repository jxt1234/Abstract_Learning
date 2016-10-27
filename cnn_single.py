#!/usr/bin/python
import Renascence
def main():
    producer = Renascence.init(["./libAbstract_learning.xml"])
    gd = producer.load('ALGradientMethod', './res/cnn/lenet.json')
    trainMerge = producer.load('ALFloatMatrix', '../../machine_exam/handset/train.txt');
    trainX = producer.build('MatrixLinear(MatrixCrop(x0, x1, x2), x3, x4)').run(producer.merge(trainMerge, 1.0, -1.0, 1.0/255.0, 0.0))
    trainY = producer.build('MatrixCrop(x0, x1, x2)').run(producer.merge(trainMerge, 0.0, 0.0))
    p = producer.build('ParameterInit(x0)').run(gd)
    print trainX,trainY,gd,p
    mergeM = producer.build('GDMatrixPrepare(x0, x1, x2)').run(producer.merge(trainX, trainY, gd))

    p = producer.build('GDCompute(x0, x1, x2)').run(producer.merge(mergeM, gd, p))
    p.save('output/handset.p')

    #p = producer.load('ALFloatMatrix', 'output/pieces/parameters/ALFloatMatrix_0')
    #print p
    
    predictX = producer.load('ALFloatMatrix', '../../machine_exam/handset/test.txt')
    print predictX
    predictY = producer.build('Classify(GDPredictorLoad(x0, x1), MatrixLinear(x2,x3,x4))').run(producer.merge(gd, p, predictX, 1.0/255.0, 0.0))

    predictY.save('output/handset.y')
if __name__ == '__main__':
    main()
