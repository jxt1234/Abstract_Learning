#include "test/GPTest.h"
#include "loader/ALStandardLoader.h"
#include <fstream>
#include <iostream>

using namespace std;

class ALLoadDataTest:public GPTest
{
    public:
        virtual void run()
        {
            ALStandardLoader s;
            ALSp<ALFloatDataChain> c = s.load("bao.txt");
            ofstream f("output/ALLoadDataTest.txt");
            for (auto p : c->get())
            {
                auto w = p->num();
                for (int j=0; j<w; ++j)
                {
                    f << p->value(0) <<" ";
                }
                f << "\n";
            }
            f.close();
        }
        ALLoadDataTest(){}
        virtual ~ALLoadDataTest(){}
};

static GPTestRegister<ALLoadDataTest> a("ALLoadDataTest");
