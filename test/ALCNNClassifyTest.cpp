#include "test/GPTest.h"

#include "ALHead.h"
#include <iostream>
#include <fstream>
#include "math/ALFloatMatrix.h"
#include "learn/ALLearnFactory.h"
#include "learn/ALCNNLearner.h"
#include <math.h>
#include <sstream>
using namespace std;

static ALSp<ALFloatMatrix> _readMatrix(const char* fileName)
{
    ALSp<ALStream> input = ALStreamFactory::readFromFile(fileName);
    return ALFloatMatrix::load(input.get());
}

static std::string readAll(const char* file)
{
    std::ostringstream output;
    std::ifstream input(file);
    output << input.rdbuf();
    return output.str();
}
class ALCNNClassifyTest:public GPTest
{
    public:
        virtual void run()
        {
            ALSp<ALFloatMatrix> X_Train = _readMatrix("data/t10k/train_x.txt");
            ALSp<ALFloatMatrix> Y_Train = _readMatrix("data/t10k/train_y.txt");
            ALSp<ALFloatMatrix> X_Test = _readMatrix("data/t10k/test_x.txt");
            ALSp<ALFloatMatrix> Y_Test = _readMatrix("data/t10k/test_y.txt");
            
            ALFloatMatrix::linearDirect(X_Train.get(), 1.0/255.0, 0.0);
            ALFloatMatrix::linearDirect(X_Test.get(), 1.0/255.0, 0.0);
            ALSp<ALFloatMatrix> Y_P = ALFloatMatrix::create(Y_Test->width(), Y_Test->height());
            
            ALIMatrix4DOp::Matrix4D inputDes;
            auto jsonString = readAll("res/cnn/test_cnn.json");
            auto jsonObject = cJSON_Parse(jsonString.c_str());
            ALSp<ALISuperviseLearner> learner = new ALCNNLearner(jsonObject);
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

static GPTestRegister<ALCNNClassifyTest> a("ALCNNClassifyTest");
