#include "test/GPTest.h"
#include <iostream>
#include <fstream>
#include "loader/ALStandardLoader.h"
#include "math/ALFloatMatrix.h"
#include "learn/ALKMeans.h"
using namespace std;

static int test_main()
{
    ofstream out("output/ALKMeansTest.txt");
    ALStandardLoader s;
    ALSp<ALFloatDataChain> c = s.load("bao.txt");
    ALSp<ALFloatMatrix> X =  ALFloatMatrix::create(c->width(), c->size());
    c->expand(X->vGetAddr(), X->width()*sizeof(ALFLOAT));
    ALSp<ALFloatMatrix> centers = ALKMeans::learn(X.get(), 5);
    out << "Class Center"<<endl;
    ALFloatMatrix::print(centers.get(), out);
    ALSp<ALFloatMatrix> result = ALFloatMatrix::create(1, X->height());
    ALKMeans::predict(X.get(), centers.get(), result.get());
    out << "Class Result"<<endl;
    ALFloatMatrix::print(result.get(), out);
    return 1;
}
class ALKMeansTest:public GPTest
{
    public:
        virtual void run()
        {
            test_main();
        }
        ALKMeansTest(){}
        virtual ~ALKMeansTest(){}
};

static GPTestRegister<ALKMeansTest> a("ALKMeansTest");
