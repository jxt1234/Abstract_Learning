#include "test/GPTest.h"
#include <iostream>
#include <fstream>
#include "core/ALExpanderFactory.h"
#include "core/ALLabeldMethodFactory.h"
#include "loader/ALStandardLoader.h"
#include "learn/ALLogicalRegress.h"
#include "learn/ALIChainLearner.h"
#include "math/ALFloatMatrix.h"
#include "UpthreadLabeld.h"
using namespace std;
static int test_main()
{
    ALStandardLoader s;
    ALSp<ALFloatDataChain> c = s.load("a1a.tra");
    ALSp<ALISuperviseLearner> l = new ALLogicalRegress(10000, 0.1);
    ALSp<ALFloatMatrix> X;
    ALSp<ALFloatMatrix> Y;
    ALStandardLoader::divide(c.get(), X, Y, 0);
    for (int i=0; i<Y->height(); ++i)
    {
        auto y = *(Y->vGetAddr(i));
        if (ZERO(y+1))
        {
            *(Y->vGetAddr(i)) = 0;
        }
    }
    ALSp<ALFloatMatrix> YP = ALFloatMatrix::create(Y->width(), Y->height());

    ALSp<ALIMatrixPredictor> p = l->vLearn(X.get(), Y.get());
    p->vPredict(X.get(), YP.get());

    int correct = 0;
    int sum = Y->height();
    ofstream out("output/ALLogicalRegressTest.txt");
    for (int i=0; i<Y->height(); ++i)
    {
        ALFLOAT y_real = *(Y->vGetAddr(i));
        auto y_predict = *(YP->vGetAddr(i));
        if (y_predict > 0.5) y_predict = 1;
        else y_predict = 0;
        out << "Predict: "<<y_predict << ", Real is "<<y_real<<endl;
        if (ZERO(y_predict-y_real))
        {
            correct++;
        }
    }
    cout << correct << " / "<<sum << endl;
    p->vPrint(cout);
    return 1;
}


class ALLogicalRegressTest:public GPTest
{
    public:
        virtual void run()
        {
            test_main();
        }
        ALLogicalRegressTest(){}
        virtual ~ALLogicalRegressTest(){}
};

static GPTestRegister<ALLogicalRegressTest> a("ALLogicalRegressTest");
