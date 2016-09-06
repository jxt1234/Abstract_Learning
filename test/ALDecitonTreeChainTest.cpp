#include "test/GPTest.h"
#include "learn/ALDecisionTree.h"
#include "core/ALExpanderFactory.h"
#include "loader/ALStandardLoader.h"
#include "learn/ALIChainLearner.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <math.h>
#include "UpthreadLabeld.h"
#include "core/ALLabeldMethodFactory.h"
using namespace std;

static int test_main()
{
    ALStandardLoader s;
    ALSp<ALFloatDataChain> c = s.load("bao.txt");
    ALARStructure ar;
    ar.l = 3;
    ar.w = 3;
    ar.d = 1;
    ar.c = 0;
    ALSp<ALIExpander> xe = ALExpanderFactory::createAR(ar);
    ALSp<ALILabeldMethod> lm = new TrendLabeld(1, 0);
    ALSp<ALLabeldData> ldata = ALLabeldMethodFactory::delayLabel(c->get(), lm.get(), 1);
    ALSp<ALIChainLearner> tree = ALIChainLearner::createFromBasic(new ALDecisionTree(), xe);
    ALSp<ALFloatPredictor> p = tree->vLearn(ldata.get());
    ofstream out("output/ALDecitionTreeChainTest.txt");
    p->vPrint(out);
    auto error = ALIChainLearner::computeError(ldata.get(), p.get());
    out << error.sum << "/"<<error.num<<endl;

    return 1;
}
class ALDecitonTreeChainTest:public GPTest
{
    public:
        virtual void run()
        {
            test_main();
        }
        ALDecitonTreeChainTest(){}
        virtual ~ALDecitonTreeChainTest(){}
};

static GPTestRegister<ALDecitonTreeChainTest> a("ALDecitonTreeChainTest");
