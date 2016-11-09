#include "test/GPTest.h"
#include "math/ALFloatMatrix.h"
#include "ALHead.h"
#include <fstream>
#include <stdlib.h>
#include <iostream>
#include <math.h>
using namespace std;
class ALMatrixTest:public GPTest
{
    public:
        virtual void run()
        {
            int w = 7, h=4;
            ALSp<ALFloatMatrix> X = ALFloatMatrix::create(w,h);
            ALSp<ALFloatMatrix> XM = ALFloatMatrix::create(w, h);
            ALSp<ALFloatMatrix> VEC = ALFloatMatrix::create(w, 1);
            ALSp<ALFloatMatrix> XFunc = ALFloatMatrix::create(w, h);
            ALSp<ALFloatMatrix> XFunc_dot_X = ALFloatMatrix::create(w, h);
            for (int i=0; i<h; ++i)
            {
                ALFLOAT* x = X->vGetAddr(i);
                ALFLOAT* xm = XM->vGetAddr(i);
                for (int j=0; j<w; ++j)
                {
                    *(x+j) = rand()%1000/103.1;
                    *(xm+j) = rand()%1000/103.34;
                }
            }
            auto VEC_v = VEC->vGetAddr();
            for (int i=0; i<w; ++i)
            {
                VEC_v[i] = rand()%1000/300.24;
            }
            ALSp<ALFloatMatrix> X_2ADDV_3 = ALFloatMatrix::create(w, h);
            ALFloatMatrix::linearVector(X_2ADDV_3.get(), X.get(), 2.0, VEC.get(), 3.0);
            ALSp<ALFloatMatrix> Y = ALFloatMatrix::transpose(X.get());
            
            auto function = [](ALFLOAT* dst, ALFLOAT* src, size_t w){
                for (size_t i=0; i<w; ++i)
                {
                    dst[i] = ::sin(src[i]);
                }
            };
            ALFloatMatrix::runLineFunction(XFunc.get(), X.get(), function);
            ALFloatMatrix::productDot(XFunc_dot_X.get(), X.get(), XFunc.get());
            
            
            ofstream out("output/ALMatrixTest.txt");
            out << "X:\n";
            ALFloatMatrix::print(X.get(), out);
            out << "VEC:\n";
            ALFloatMatrix::print(VEC.get(), out);
            out << "X_2ADDV_3:\n";
            ALFloatMatrix::print(X_2ADDV_3.get(), out);
            out << "sin(X):\n";
            ALFloatMatrix::print(XFunc.get(), out);
            out << "Xsin(X):\n";
            ALFloatMatrix::print(XFunc_dot_X.get(), out);
            
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
            ALFloatMatrix::linear(XM.get(), X.get(), 0.5, XM.get(), -0.4);
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
            auto x = X->vGetAddr();
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
