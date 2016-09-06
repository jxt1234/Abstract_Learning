#include "test/GPTest.h"
#include "package/ALPackage.h"
#include "learn/ALRegressor.h"
#include "loader/ALStandardLoader.h"
#include "core/ALExpanderFactory.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <math.h>
using namespace std;

class ALRegressAR2Test:public GPTest
{
    public:
        virtual void run()
        {
            ALStandardLoader s;
            ALSp<ALFloatDataChain> c = s.load("bao.txt");
            ALSp<ALLabeldData> ldata = ALPackageLabled(c.get(), 1.0);
            ALARStructure ar;
            ofstream total("output/ALARRegressAR2Test.txt");
            /*l:1~2, w:1~3, c:0~1, d:1~3*/
            for (int i=0; i<36; ++i)
            {
                ar.l = i/18;
                ar.w = (i-18*ar.l)/6;
                ar.c = (i-18*ar.l-6*ar.w)/3;
                ar.d = i%3;

                ar.l=ar.l+1;
                ar.w=ar.w+1;

                total << "l="<<ar.l<<"  w="<<ar.w<<"  d="<<ar.d<<"  c="<<ar.c<<endl;
                ALSp<ALIChainLearner> reg = ALIChainLearner::createFromBasic(new ALRegressor, ALExpanderFactory::createAR(ar));
                ALSp<ALFloatPredictor> p = reg->vLearn(ldata.get());
                p->vPrint(total);
                ALIChainLearner::Error res = ALIChainLearner::computeError(ldata.get(), p.get());
                ALFLOAT error = res.sum;
                int num = res.num;
                total << "error="<< error<<"  num="<<num << endl;
                ostringstream name_os;
                name_os << "output/ALRegressTest_l"<<ar.l<<"_w"<<ar.w<<"_c"<<ar.c<<"_d"<<ar.d<<".txt";
                ofstream out(name_os.str().c_str());
                for (auto point : ldata->get())
                {
                    out << point.first << " : " << p->vPredict(point.second) <<endl;
                }
                out.close();
            }
            total.close();
        }
        ALRegressAR2Test(){}
        virtual ~ALRegressAR2Test(){}
};

static GPTestRegister<ALRegressAR2Test> a("ALRegressAR2Test");
