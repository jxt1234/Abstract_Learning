#include "test/GPTest.h"
#include <iostream>
#include <fstream>
#include "loader/ALStandardLoader.h"
#include "math/ALFloatMatrix.h"
#include "learn/ALGMM.h"
using namespace std;
static int test_main()
{
    ofstream out("output/ALGMMTest.txt");
    ALStandardLoader s;
    ALSp<ALFloatDataChain> c = s.load("bao.txt");
    ALSp<ALFloatMatrix> X =  ALFloatMatrix::create(c->width(), c->size());
    c->expand(X->vGetAddr(), X->width()*sizeof(ALFLOAT));
    ALGMM gmmlearner(3);
    ALSp<ALIMatrixPredictor> predictor = gmmlearner.vLearn(X.get());
    ALSp<ALFloatMatrix> result = ALFloatMatrix::create(1, X->height());
    predictor->vPredict(X.get(), result.get());
    ALFloatMatrix::print(result.get(), out);
    return 1;
}
class ALGMMTest:public GPTest
{
    public:
        virtual void run()
        {
            test_main();
        }
        ALGMMTest(){}
        virtual ~ALGMMTest(){}
};

static GPTestRegister<ALGMMTest> a("ALGMMTest");
