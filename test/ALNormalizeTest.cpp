#include "test/GPTest.h"
#include "core/ALNormalizer.h"
#include "core/ALBasicExpander.h"
#include "loader/ALStandardLoader.h"
#include <fstream>

using namespace std;
class ALNormalizeTest:public GPTest
{
    public:
        virtual void run()
        {
            ALStandardLoader s;
            ALSp<ALFloatDataChain> c = s.load("bao.txt");
            ALARStructure ar;
            ar.l = 1;
            ar.w = 3;
            ar.c = 1;
            ar.d = 1;
            ALSp<ALARExpander> b = new ALARExpander(ar);
            ALSp<ALNormalizer> n = new ALNormalizer(c->get(), b.get());
            int l = n->vLength();
            ALAutoStorage<ALFLOAT> _dst(l);
            ALFLOAT* d = _dst.get();
            ofstream out("output/ALNormalizeTest.txt");
            for (auto datapoint : c->get())
            {
                bool res = n->vExpand(datapoint, d);
                out << res <<": ";
                for (int i=0; i<l; ++i)
                {
                    out << d[i] <<" ";
                }
                out <<"\n";
            }
        }
        ALNormalizeTest(){}
        virtual ~ALNormalizeTest(){}
};

static GPTestRegister<ALNormalizeTest> a("ALNormalizeTest");
