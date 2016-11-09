#include "test/GPTest.h"
#include "data/ALVaryArray.h"
#include "utils/ALStream.h"
#include <fstream>


class ALVaryArrayTest:public GPTest
{
    public:
        virtual void run()
        {
            ALSp<ALStream> read =  ALStreamFactory::readFromFile("./data/lstm.numbers");
            ALSp<ALVaryArray> array = ALVaryArray::create(read.get());
            auto n = array->size();
            std::ofstream output("./output/ALVaryArrayTest.txt");
            for (size_t i=0; i<n; ++i)
            {
                auto a = array->getArray(i);
                output << a.length << "\t";
                for (size_t j=0; j<a.length; ++j)
                {
                    output << a.c[j]<<"\t";
                }
                output << "\n";
            }
        }
        ALVaryArrayTest(){}
        virtual ~ALVaryArrayTest(){}
};

static GPTestRegister<ALVaryArrayTest> a("ALVaryArrayTest");
