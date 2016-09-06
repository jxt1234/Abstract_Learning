#include "ALHead.h"
#include <iostream>
#include <fstream>
#include "loader/ALStandardLoader.h"
#include "math/ALFloatMatrix.h"
#include "learn/ALGMMClassify.h"
#include "learn/ALDecisionTree.h"
#include "learn/ALPCABasic.h"
#include "loader/ALStandardLoader.h"
#include "math/ALStatistics.h"
#include "compose/ALRandomForestMatrix.h"
#include "learn/ALMatrixNormalizer.h"
#include <math.h>
using namespace std;

int test_main(int argc, char* argv[])
{
    ALASSERT(argc>=2);
    std::string dataName = argv[1];
    ALSp<ALFloatMatrix> X;
    ALSp<ALFloatMatrix> Y;
    {
        ALSp<ALFloatDataChain> inputs = ALStandardLoader::load(dataName.c_str());
        ALStandardLoader::divide(inputs.get(), X, Y, 0);
    }
    bool normalize = false;
    ALSp<ALIMatrixTransformer> normalizer;
    if (normalize)
    {
        FUNC_PRINT(X->width());
        normalizer = new ALMatrixNormalizer(X.get());
        X = normalizer->vTransform(X.get());
        ofstream output("temp.txt");
        FUNC_PRINT(X->width());
        ALFloatMatrix::print(X.get(), output);
    }
    
    ALSp<ALISuperviseLearner> learner = new ALRandomForestMatrix(25, true);
    //ALSp<ALISuperviseLearner> learner = new ALGMMClassify;
    //ALSp<ALISuperviseLearner> learner = new ALDecisionTree;
    ALSp<ALIMatrixPredictor> predictor = learner->vLearn(X.get(), Y.get());
    ALSp<ALFloatMatrix> YP = ALFloatMatrix::create(1, Y->height());
    ALSp<ALFloatMatrix> YYP = ALFloatMatrix::create(3, Y->height());
    predictor->vPredict(X.get(), YP.get());
    std::ofstream modelfile(dataName+".modle_with_offset");
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
        if (ZERO(yyp[2]))
        {
            sum++;
        }
    }
    
    ALSp<ALFloatMatrix> statiscMatrix = ALStatistics::statistics(YYP.get());
    ALSp<ALWStream> fstatistics = ALStreamFactory::writeForFile((dataName+"_result_offset.txt").c_str());
    ALFloatMatrix::save(statiscMatrix.get(), fstatistics.get());
    
    ALSp<ALWStream> f = ALStreamFactory::writeForFile((dataName+"_result_offset_detail.txt").c_str());
    ALFloatMatrix::save(YYP.get(), f.get());
    std::cout << dataName << ": Correct = " << sum/h<<"\n";
    
    if (argc>2)
    {
        std::string predictDataName = argv[2];
        ALSp<ALStream> inputs = ALStreamFactory::readFromFile(predictDataName.c_str());
        ALSp<ALFloatMatrix> XP = ALFloatMatrix::load(inputs.get());
        inputs = NULL;
        if (normalize)
        {
            FUNC_PRINT(XP->width());
            XP = normalizer->vTransform(XP.get());
            FUNC_PRINT(XP->width());
        }
        ALSp<ALFloatMatrix> YPP = ALFloatMatrix::create(1, XP->height());
        predictor->vPredict(XP.get(), YPP.get());
        ALSp<ALWStream> f = ALStreamFactory::writeForFile((predictDataName+"_predict.txt").c_str());
        ALFloatMatrix::save(YPP.get(), f.get());
    }
    
    return 1;
}

int main(int argc, char* argv[])
{
    ALAUTOTIME;
    char* _argv[] = {"", "/Users/jiangxiaotang/machine_exam/handset/train.txt"};
    test_main(2, _argv);
    //test_main(argc, argv);
    return 1;
}
