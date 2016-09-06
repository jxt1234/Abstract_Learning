#include "test/GPTest.h"
#include "utils/ALStreamReader.h"
#include <fstream>
using namespace std;

class ALStreamReaderTest:public GPTest
{
    public:
        virtual void run()
        {
            ALSp<ALStream> input = ALStreamFactory::readFromFile("bao.txt");
            ALSp<ALStreamReader> reader = new ALStreamReader(input.get());
            char buffer[4096];
            ofstream out("output/ALStreamReaderTest.txt");
            while (!reader->end())
            {
                reader->readline(buffer, 4095);
                out << buffer;
            }
        }
        ALStreamReaderTest(){}
        virtual ~ALStreamReaderTest(){}
};

static GPTestRegister<ALStreamReaderTest> a("ALStreamReaderTest");
