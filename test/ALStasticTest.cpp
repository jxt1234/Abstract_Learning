#include "test/GPTest.h"
#include <iostream>
#include <fstream>
#include "loader/ALStandardLoader.h"
#include "math/ALFloatMatrix.h"
#include "math/ALStatistics.h"
using namespace std;

static int test_main()
{
    ofstream out("output/ALStatisticsTest.txt");
    ALStandardLoader s;
    ALSp<ALFloatDataChain> c = s.load("bao.txt");
    ALSp<ALFloatMatrix> X =  ALFloatMatrix::create(c->width(), c->size());
    c->expand(X->vGetAddr(), X->width()*sizeof(ALFLOAT));
    ALSp<ALFloatMatrix> stats = ALStatistics::statistics(X.get());
    ALFloatMatrix::print(stats.get(), out);
    out << "\n";

    ALSp<ALFloatMatrix> NX = ALStatistics::normalize(X.get());
    ALFloatMatrix::print(NX.get(), out);
    X = ALFloatMatrix::transpose(X.get());
    ALSp<ALFloatMatrix> C = ALStatistics::covariance(X.get());
    ALFloatMatrix::print(C.get(), out);
    out << endl;

    ALSp<ALFloatMatrix> H1 = ALStatistics::characteristic_root(C.get());
    ALFloatMatrix::print(H1.get(), out);

    ALSp<ALFloatMatrix> Root,Vector;
    ALStatistics::characteristic_compute(C.get(), Root, Vector);
    out << endl;
    ALFloatMatrix::print(Root.get(), out);
    out << endl;
    ALFloatMatrix::print(Vector.get(), out);

    return 1;
}
class ALStasticTest:public GPTest
{
    public:
        virtual void run()
        {
            test_main();
        }
        ALStasticTest(){}
        virtual ~ALStasticTest(){}
};

static GPTestRegister<ALStasticTest> a("ALStasticTest");
