#include "test/GPTest.h"
#include "ALHead.h"
#include "math/ALFloatMatrix.h"
#include <fstream>
class ALLargeMatrixTest:public GPTest
{
    public:
        virtual void run()
        {
            int w = 7,h=4;
            {
                ALSp<ALWStream> largef = ALStreamFactory::writeForFile("output/ALLargeMatrixTest.out");
                std::ofstream out("output/ALLargeMatrixTest_origin.txt");
                for (int tt=0;tt<10; ++tt)
                {
                    ALSp<ALFloatMatrix> X = ALFloatMatrix::create(w,h);
                    ALFLOAT* x = X->vGetAddr();
                    for (int i=0; i<w*h; ++i)
                    {
                        *(x+i) = (rand()%1000)/103.1;
                    }
                    ALFloatMatrix::print(X.get(), out);
                    ALFloatMatrix::quickSave(X.get(), largef.get());
                }
            }
            {
                ALSp<ALStream> largef = ALStreamFactory::readFromFile("output/ALLargeMatrixTest.out");
                ALSp<ALFloatMatrix> largeM = ALFloatMatrix::quickLoadLarge(largef.get());
                std::ofstream out("output/ALLargeMatrixTest_result.txt");
                ALFloatMatrix::print(largeM.get(), out);
            }
        }
        ALLargeMatrixTest(){}
        virtual ~ALLargeMatrixTest(){}
};

static GPTestRegister<ALLargeMatrixTest> a("ALLargeMatrixTest");
