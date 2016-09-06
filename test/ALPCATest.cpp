#include "test/GPTest.h"
#include <iostream>
#include <fstream>
#include "loader/ALStandardLoader.h"
#include "math/ALFloatMatrix.h"
#include "math/ALStatistics.h"
#include "learn/ALPCABasic.h"
using namespace std;

static int test_main()
{
    ALStandardLoader s;
    ALSp<ALFloatDataChain> c = s.load("bao.txt");
    ALSp<ALFloatMatrix> X =  ALFloatMatrix::create(c->width(), c->size());
    for (int i=0; i<c->size(); ++i)
    {
        auto x = X->vGetAddr(i);
        auto p = (c->get())[i];
        p->copy(x);
    }
    ALSp<ALFloatMatrix> XT = ALFloatMatrix::transpose(X.get());
    ALSp<ALPCABasic> pca = new ALPCABasic(XT.get(), 0.8);
    ALSp<ALFloatMatrix> Y = pca->vTransform(X.get());
    ofstream output("output/ALPCATest.txt");
    ALFloatMatrix::print(Y.get(), output);
    return 1;
}
class ALPCATest:public GPTest
{
    public:
        virtual void run()
        {
            test_main();
        }
        ALPCATest(){}
        virtual ~ALPCATest(){}
};

static GPTestRegister<ALPCATest> a("ALPCATest");
