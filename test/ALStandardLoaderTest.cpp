#include "test/GPTest.h"
#include "loader/ALStandardLoader.h"
#include <fstream>
using namespace std;

class ALStandardLoaderTest:public GPTest
{
    public:
        virtual void run()
        {
            ALSp<ALStream> input = ALStreamFactory::readFromFile("bao.txt");
            ALSp<ALFloatDataChain> c = ALStandardLoader::load(input.get());
            ALSp<ALFloatMatrix> X;
            ALSp<ALFloatMatrix> Y;
            ALStandardLoader::divide(c.get(), X, Y, 0);
            ofstream outX("output/ALStandardLoaderTest_X.txt");
            ofstream outY("output/ALStandardLoaderTest_Y.txt");
            ALFloatMatrix::print(X.get(), outX);
            ALFloatMatrix::print(Y.get(), outY);
        }
        ALStandardLoaderTest(){}
        virtual ~ALStandardLoaderTest(){}
};

static GPTestRegister<ALStandardLoaderTest> a("ALStandardLoaderTest");
