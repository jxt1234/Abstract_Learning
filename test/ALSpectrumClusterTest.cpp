#include "test/GPTest.h"
#include "learn/ALSpectralClustering.h"
#include <iostream>
#include <fstream>
#include "loader/ALStandardLoader.h"
#include "math/ALFloatMatrix.h"
using namespace std;
static int test_main()
{
    ALAUTOSTORAGE(v, ALFLOAT, 100);
    ALStandardLoader s;
    ALSp<ALFloatDataChain> c = s.load("bao2.txt");
    ALSp<ALFloatMatrix> X =  ALFloatMatrix::create(c->width(), c->size());
    c->expand(X->vGetAddr(), X->width()*sizeof(ALFLOAT));
    ALSp<ALIExpander> xe;
    ALSpectralClustering sc(xe, 50, 4, 4);
    ofstream out("output/ALSpectrumClusterTest.txt");
    out << "Class Result"<<endl;
    ALSp<ALFloatMatrix> result = sc.classify(X.get());
    result = ALFloatMatrix::transpose(result.get());
    ALFloatMatrix::print(result.get(), out);
    return 1;
}
class ALSpectrumClusterTest:public GPTest
{
    public:
        virtual void run()
        {
            test_main();
        }
        ALSpectrumClusterTest(){}
        virtual ~ALSpectrumClusterTest(){}
};

static GPTestRegister<ALSpectrumClusterTest> a("ALSpectrumClusterTest");
