#include "ALHead.h"
#include <iostream>
#include <fstream>
#include "math/ALFloatMatrix.h"
#include "learn/ALLearnFactory.h"
#include "learn/ALRNNLearner.h"
#include <math.h>
#include <sstream>
#include "cJSON/cJSON.h"
#include "core/ALARStructure.h"
#include "core/ALExpanderFactory.h"
#include "loader/ALStandardLoader.h"
#include <iostream>
#include "package/ALPackage.h"
using namespace std;
static std::string readAll(const char* file)
{
    std::ostringstream output;
    std::ifstream input(file);
    output << input.rdbuf();
    return output.str();
}

int test_main(int argc, char* argv[])
{
    ALStandardLoader s;
    ALSp<ALFloatDataChain> train_raw = s.load("./bao_normal.txt");
    ALSp<ALLabeldData> train = ALPackageLabled(train_raw.get(), 1.0);
    ALSp<ALFloatDataChain> test_raw = s.load("./bao_predict_normal.txt");
    ALSp<ALLabeldData> test = ALPackageLabled(test_raw.get(), 1.0);
    
    auto jsonString = readAll(argv[1]);
    auto jsonObject = cJSON_Parse(jsonString.c_str());
    ALSp<ALIChainLearner> learner = new ALRNNLearner(jsonObject);
    //ALSp<ALISuperviseLearner> learner = new ALRandomForestMatrix(55);
    ALSp<ALFloatPredictor> predictor = learner->vLearn(train.get());
    
    ALFLOAT sumError = 0.0f;
    {
        std::ofstream outputP("output/ALRNNLearnerTest.txt");
        for (auto p : test->get())
        {
            auto real = p.first;
            auto pre = predictor->vPredict(p.second);
            if (ZERO(pre))
            {
                continue;
            }
            auto error = real - pre;
            sumError += error*error;
            outputP << real <<"\t"<<pre<<"\n";
        }
    }
    {
        std::ofstream outputP("output/ALRNNLearnerTrain.txt");
        for (auto p : train->get())
        {
            auto real = p.first;
            auto pre = predictor->vPredict(p.second);
            if (ZERO(pre))
            {
                continue;
            }
            auto error = real - pre;
            sumError += error*error;
            outputP << real <<"\t"<<pre<<"\n";
        }
    }
    std::cout << sumError << "/"<<test->get().size()<<"\n";
    return 1;
}

int main(int argc, char* argv[])
{
    ALAUTOTIME;
    test_main(argc, argv);
    return 1;
}
