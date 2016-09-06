#include "ALHead.h"
#include "utils/ALStreamReader.h"
#include <iostream>
#include <fstream>
#include "package/ALPackage.h"
#include "core/ALExpanderFactory.h"
#include "core/ALLabeldMethodFactory.h"
#include "loader/ALStandardLoader.h"
#include "learn/ALLogicalRegress.h"
#include "learn/ALIChainLearner.h"
#include "learn/ALNetRegressor.h"
#include "math/ALFloatMatrix.h"

#include "learn/ALRegressor.h"
#include "learn/ALNetRegressor.h"
#include "loader/ALStandardLoader.h"
#include "math/ALStatistics.h"
using namespace std;

int test_main(int argc, char* argv[])
{
    ALASSERT(3<=argc);
    ALSp<ALFloatMatrix> X;
    ALSp<ALFloatMatrix> Y;
    string inputDataFile = argv[1];
    string modelFile = argv[2];
    const char* rgb[] = {"r", "g", "b"};
    for (int i=0; i<3; ++i)
    {
        string inputfile = inputDataFile + "_"+rgb[i];
        string outputfile = modelFile + "_" + rgb[i];
        {
            ALSp<ALFloatDataChain> inputs = ALStandardLoader::load(inputfile.c_str());
            ALStandardLoader::divide(inputs.get(), X, Y, 0);
        }
        ALSp<ALISuperviseLearner> learner = new ALNetRegressor(2, true);
        //ALSp<ALISuperviseLearner> learner = new ALRegressor;
        ALSp<ALIMatrixPredictor> predictor = learner->vLearn(X.get(), Y.get());
        ALSp<ALFloatMatrix> YP = ALFloatMatrix::create(1, Y->height());
        ALSp<ALFloatMatrix> YYP = ALFloatMatrix::create(3, Y->height());
        predictor->vPredict(X.get(), YP.get());
        std::ofstream modelfile(outputfile.c_str());
        predictor->vPrint(modelfile);
        auto h = Y->height();
        auto sum = 0.0;
        for (int i=0; i<h; ++i)
        {
            auto yyp = YYP->vGetAddr(i);
            auto yp = YP->vGetAddr(i);
            auto y = Y->vGetAddr(i);
            yyp[0] = yp[0];
            yyp[1] = y[0];
            yyp[2] = y[0] - yp[0];
            sum += yyp[2]*yyp[2];
        }
        
        ALSp<ALFloatMatrix> statiscMatrix = ALStatistics::statistics(YYP.get());
        ALFloatMatrix::print(statiscMatrix.get(), std::cout);
        
//        ALSp<ALWStream> f = ALStreamFactory::writeForFile("/Users/jiangxiaotang/Documents/Abstract_Learning/temp.txt");
//        ALFloatMatrix::save(YYP.get(), f.get());
        std::cout << "Sum error = " << sum/h<<"\n";
    }
    return 1;
}

int main(int argc, char* argv[])
{
    test_main(argc, argv);
    return 1;
}
