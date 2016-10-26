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
            ALDynamicBuffer dyBuffer(4096);
            ofstream out("output/ALStreamReaderTest.txt");
            while (!reader->end())
            {
                reader->readline(&dyBuffer);
                out << dyBuffer.content();
            }
        }
        ALStreamReaderTest(){}
        virtual ~ALStreamReaderTest(){}
};

static GPTestRegister<ALStreamReaderTest> a("ALStreamReaderTest");
