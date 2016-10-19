#include "test/GPTest.h"
#include "math/ALFloatMatrix.h"
#include "ALHead.h"
#include <fstream>
#include <stdlib.h>
#include <iostream>
using namespace std;
class ALMatrixTest:public GPTest
{
    public:
        virtual void run()
        {
            int w = 7, h=4;
            ALSp<ALFloatMatrix> X = ALFloatMatrix::create(w,h);
            ALSp<ALFloatMatrix> XM = ALFloatMatrix::create(w, h);
            ALFLOAT* x = X->vGetAddr();
            auto rX = X->width();
            ALFLOAT* xm = XM->vGetAddr();
            auto rXM = XM->width();
            for (int i=0; i<h; ++i)
            {
                for (int j=0; j<w; ++j)
                {
                    *(x+rX*i+j) = rand()%1000/103.1;
                    *(xm+rXM*i+j) = rand()%1000/103.34;
                }
            }
            
            ALSp<ALFloatMatrix> Y = ALFloatMatrix::transpose(X.get());
            ofstream out("output/ALMatrixTest.txt");
            out << "X:\n";
            ALFloatMatrix::print(X.get(), out);
            out << "Virtual Matrix\n";
            {
                int l = w/4;
                int r = 3*w/4;
                int t = h/4;
                int b = 3*h/4;
                ALSp<ALFloatMatrix> crop = ALFloatMatrix::createCropVirtualMatrix(X.get(), l, t, r, b);
                ALFloatMatrix::print(crop.get(), out);
            }
            out << "XM:\n";
            ALFloatMatrix::print(XM.get(), out);
            XM = ALFloatMatrix::linear(X.get(), 0.5, XM.get(), -0.4);
            out << "X*0.5-XM*0.4:\n";
            ALFloatMatrix::print(XM.get(), out);
            XM = ALFloatMatrix::createIdentity(5);
            out << "I5\n";
            ALFloatMatrix::print(XM.get(), out);
            out << "Y:\n";
            ALFloatMatrix::print(Y.get(), out);
            out << "XTX:\n";
            ALSp<ALFloatMatrix> XTX = ALFloatMatrix::sts(X.get(), false);
            ALFloatMatrix::print(XTX.get(), out);
            out << "XXT:\n";
            ALSp<ALFloatMatrix> XXT = ALFloatMatrix::sts(X.get(), true);
            ALFloatMatrix::print(XXT.get(), out);

            out << "Enlarge(XTX):\n";
            ALSp<ALFloatMatrix> E_XTX = ALFloatMatrix::enlarge(10, XTX.get());
            ALFloatMatrix::print(E_XTX.get(), out);
            ALSp<ALFloatMatrix> XY = ALFloatMatrix::product(X.get(), Y.get());
            out << "XY:\n";
            ALFloatMatrix::print(XY.get(), out);
            ALSp<ALFloatMatrix> Z = ALFloatMatrix::inverse(XY.get());
            out << "Z:\n";
            ALFloatMatrix::print(Z.get(),out);
            X = ALFloatMatrix::product(XY.get(), Z.get());
            out << "XY * Z:\n";
            ALFloatMatrix::print(X.get(), out);

            X = ALFloatMatrix::create(10, 1);
            x = X->vGetAddr();
            for (int i=0; i<10; ++i)
            {
                x[i] = i*10+1;
            }
            out << "Single V\n";
            ALFloatMatrix::print(X.get(), out);
            X = ALFloatMatrix::createDiag(X.get());
            out << "Diag V\n";
            ALFloatMatrix::print(X.get(), out);
            out << "Zero \n";
            X = ALFloatMatrix::create(3, 4);
            ALFloatMatrix::zero(X.get());
            ALFloatMatrix::print(X.get(), out);
            ALSp<ALFloatMatrix> copyX = ALFloatMatrix::create(Z->width(), Z->height());
            ALFloatMatrix::copy(copyX.get(), Z.get());
            out << "Copy \n";
            ALFloatMatrix::print(copyX.get(), out);
            out << "Save And Load\n";
            {
                ALSp<ALWStream> f = ALStreamFactory::writeForFile("output/ALMatrixTest_save.txt");
                ALFloatMatrix::save(copyX.get(), f.get());
            }
            {
                ALSp<ALStream> f = ALStreamFactory::readFromFile("output/ALMatrixTest_save.txt");
                ALSp<ALFloatMatrix> qSM = ALFloatMatrix::load(f.get());
                ALFloatMatrix::print(qSM.get(), out);
            }
            out << "QuickSave And Load\n";
            {
                ALSp<ALWStream> f = ALStreamFactory::writeForFile("output/temp.txt");
                ALFloatMatrix::quickSave(copyX.get(), f.get());
            }
            {
                ALSp<ALStream> f = ALStreamFactory::readFromFile("output/temp.txt");
                ALSp<ALFloatMatrix> qSM = ALFloatMatrix::quickLoad(f.get());
                ALFloatMatrix::print(qSM.get(), out);
            }
            out << "Check the same\n";
            Y = ALFloatMatrix::create(X->width(), X->height());
            ALFloatMatrix::copy(Y.get(), X.get());
            bool theSame = ALFloatMatrix::theSame(X.get(), Y.get());
            ALASSERT(theSame);
            auto y = Y->vGetAddr(X->height()/2);
            y[0] = y[0]-1.0f;
            theSame = ALFloatMatrix::theSame(X.get(), Y.get(), 0.1);
            ALASSERT(!theSame);
        }
        ALMatrixTest(){}
        virtual ~ALMatrixTest(){}
};

static GPTestRegister<ALMatrixTest> a("ALMatrixTest");
