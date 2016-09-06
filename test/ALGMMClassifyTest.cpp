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
#include "learn/ALGMMClassify.h"

using namespace std;

static void test_main()
{
    ALStandardLoader s;
    ALSp<ALFloatDataChain> c = s.load("a1a.tra");
    ALSp<ALISuperviseLearner> l = new ALGMMClassify(5);
    ALSp<ALFloatMatrix> X;
    ALSp<ALFloatMatrix> Y;
    ALStandardLoader::divide(c.get(), X, Y, 0);
    ALSp<ALFloatMatrix> YP = ALFloatMatrix::create(Y->width(), Y->height());
    
    ALSp<ALIMatrixPredictor> p = l->vLearn(X.get(), Y.get());
    p->vPredict(X.get(), YP.get());
    
    int correct = 0;
    auto sum = Y->height();
    {
        ofstream out("output/ALGMMClassifyTest.model");
        p->vPrint(out);
    }
    ofstream out("output/ALGMMClassifyTest.txt");
    for (int i=0; i<Y->height(); ++i)
    {
        ALFLOAT y_real = *(Y->vGetAddr(i));
        auto y_predict = *(YP->vGetAddr(i));
        out << "Predict: "<<y_predict << ", Real is "<<y_real<<endl;
        if (ZERO(y_predict-y_real))
        {
            correct++;
        }
    }
    cout << correct << " / "<<sum << endl;
    //p->vPrint(cout);
}

class ALGMMClassifyTest:public GPTest
{
    public:
        virtual void run()
        {
            test_main();
        }
        ALGMMClassifyTest(){}
        virtual ~ALGMMClassifyTest(){}
};

static GPTestRegister<ALGMMClassifyTest> a("ALGMMClassifyTest");
