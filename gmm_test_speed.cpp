#include "ALHead.h"
#include "learn/ALGMM.h"
#include "learn/ALLearnFactory.h"
#include "core/ALExpanderFactory.h"
#include "core/ALILabeldMethod.h"
#include "core/ALLabeldMethodFactory.h"
#include "learn/ALDecisionTree.h"
#include "learn/ALGMMClassify.h"
#include "learn/ALKMeans.h"
#include "loader/ALStandardLoader.h"
#include "learn/ALLogicalRegress.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <math.h>
#include "learn/ALRegressor.h"
using namespace std;
int main()
{
    ALClock __t("KMEANS", 1);
    ALSp<ALFloatMatrix> X;
    {
        ALSp<ALStream> input = ALStreamFactory::readFromFile("/Users/jiangxiaotang/Data/Face_Dataset/train.data");
        X = ALFloatMatrix::load(input.get());
    }
    //ALSp<ALIUnSuperLearner> learner = new ALGMM(3);
    ALSp<ALIUnSuperLearner> learner = new ALKMeans(3, 1000);
    ALSp<ALIMatrixPredictor> detected = learner->vLearn(X.get());
    detected->vPrint(cout);
    //ALSp<ALFloatMatrix> YP = ALFloatMatrix::create(1, X->height());
    //detected->vPredict(X.get(), YP.get());
    return 0;
}
