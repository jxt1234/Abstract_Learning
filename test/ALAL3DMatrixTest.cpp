#include "test/GPTest.h"
#include "math/AL3DMatrix.h"
class ALAL3DMatrixTest:public GPTest
{
    public:
        virtual void run()
        {
            int w = 100;
            int h = 20;
            ALSp<ALFloatMatrix> matrix = ALFloatMatrix::create(w, h);
            for(int i=0; i<h; ++i)
            {
                auto f = matrix->vGetAddr(i);
                for (int j=0; j<w; ++j)
                {
                    f[j] = ALRandom::rate();
                }
            }
            
            
            ALSp<AL3DMatrix> matrix3D = AL3DMatrix::create(matrix, 5, 20);
            for (int i=0; i<h; ++i)
            {
                auto f = matrix->vGetAddr(i);
                int cur = 0;
                for (int j=0; j<20; ++j)
                {
                    auto target = matrix3D->getAddr(j, i);
                    for (int k=0; k<5; ++k)
                    {
                        ALASSERT(f[cur] == target[k]);
                        cur++;
                    }
                }
            }
        }
        ALAL3DMatrixTest(){}
        virtual ~ALAL3DMatrixTest(){}
};

static GPTestRegister<ALAL3DMatrixTest> a("ALAL3DMatrixTest");
