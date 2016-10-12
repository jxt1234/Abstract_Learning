#include "test/GPTest.h"

#include "ALHead.h"
#include <iostream>
#include <fstream>
#include "math/ALFloatMatrix.h"
#include "learn/ALLearnFactory.h"
#include "learn/ALCNNLearner.h"
#include <math.h>
using namespace std;

static ALSp<ALFloatMatrix> _readMatrix(const char* fileName)
{
    ALSp<ALStream> input = ALStreamFactory::readFromFile(fileName);
    return ALFloatMatrix::quickLoad(input.get());
}

class ALCNNClassifyTest:public GPTest
{
    public:
        virtual void run()
        {
            ALSp<ALFloatMatrix> X_Train = _readMatrix("data/mnist/train_x");
            ALSp<ALFloatMatrix> Y_Train = _readMatrix("data/mnist/train_y");
            ALSp<ALFloatMatrix> X_Test = _readMatrix("data/mnist/test_x");
            ALSp<ALFloatMatrix> Y_Test = _readMatrix("data/mnist/test_y");
            
            ALSp<ALFloatMatrix> Y_P = ALFloatMatrix::create(Y_Test->width(), Y_Test->height());
            
            ALIMatrix4DOp::Matrix4D inputDes;
            inputDes.iDepth = 1;
            inputDes.iWidth = 28;
            inputDes.iHeight = 28;
            inputDes.iExpand = 0;
            ALSp<ALISuperviseLearner> learner = new ALCNNLearner(inputDes, 1000);
            ALSp<ALIMatrixPredictor> predictor = learner->vLearn(X_Train.get(), Y_Train.get());
            
            ALSp<ALFloatMatrix> Y_P_Detail = ALFloatMatrix::create(predictor->vGetPossiableValues()->width(), Y_Test->height());
            predictor->vPredictProbability(X_Test.get(), Y_P_Detail.get());
            {
                std::ofstream outputP("output/ALCNNLearnerTestProp.txt");
                ALFloatMatrix::print(Y_P_Detail.get(), outputP);
            }
            
            
            predictor->vPredict(X_Test.get(), Y_P.get());
            auto h = Y_Test->height();
            int correct = 0;
            for (int i=0; i<h; ++i)
            {
                auto y = Y_Test->vGetAddr(i)[0];
                auto yp = Y_P->vGetAddr(i)[0];
                if (ZERO(y-yp))
                {
                    correct++;
                }
            }
            
            ALSp<ALFloatMatrix> YYP = ALFloatMatrix::unionHorizontal(Y_P.get(), Y_Test.get());
            std::ofstream output("output/ALCNNLearnerTest.txt");
            ALFloatMatrix::print(YYP.get(), output);
            std::cout << "correct: "<<correct<<"/"<<h<<std::endl;
        }
        ALCNNClassifyTest(){}
        virtual ~ALCNNClassifyTest(){}
};

//static GPTestRegister<ALCNNClassifyTest> a("ALCNNClassifyTest");
