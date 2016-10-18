#include "ALHead.h"
#include <iostream>
#include <fstream>
#include "math/ALFloatMatrix.h"
#include "compose/ALRandomForestMatrix.h"
#include "learn/ALLearnFactory.h"
#include "learn/ALCNNLearner.h"
#include <math.h>
#include <sstream>
#include "cJSON/cJSON.h"
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

int test_main(int argc, char* argv[])
{
    ALSp<ALFloatMatrix> X_Train = _readMatrix("/Users/jiangxiaotang/Documents/data/t10k/train_x.txt");
    ALSp<ALFloatMatrix> Y_Train = _readMatrix("/Users/jiangxiaotang/Documents/data/t10k/train_y.txt");
    ALSp<ALFloatMatrix> X_Test = _readMatrix("/Users/jiangxiaotang/Documents/data/t10k/test_x.txt");
    ALSp<ALFloatMatrix> Y_Test = _readMatrix("/Users/jiangxiaotang/Documents/data/t10k/test_y.txt");
    
    ALFloatMatrix::linearDirect(X_Train.get(), 1.0/255.0, 0.0);
    ALFloatMatrix::linearDirect(X_Test.get(), 1.0/255.0, 0.0);
    
    ALSp<ALFloatMatrix> Y_P = ALFloatMatrix::create(Y_Test->width(), Y_Test->height());
    
    ALIMatrix4DOp::Matrix4D inputDes;
    inputDes.iDepth = 1;
    inputDes.iWidth = 28;
    inputDes.iHeight = 28;
    inputDes.iExpand = 0;
    auto jsonString = readAll("/Users/jiangxiaotang/Documents/Abstract_Learning/res/cnn/lenet.json");
    auto jsonObject = cJSON_Parse(jsonString.c_str());
    ALSp<ALISuperviseLearner> learner = new ALCNNLearner(jsonObject, 100000);
    //ALSp<ALISuperviseLearner> learner = new ALRandomForestMatrix(55);
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
    return 1;
}

int main(int argc, char* argv[])
{
    ALAUTOTIME;
    //char* _argv[] = {"", "/Users/jiangxiaotang/machine_exam/handset/train.txt"};
    char* _argv[] = {"", "/Users/jiangxiaotang/third/caffe/train.txt"};
    test_main(2, _argv);
    //test_main(argc, argv);
    return 1;
}
