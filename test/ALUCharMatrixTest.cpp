#include "test/GPTest.h"
#include "math/ALUCharMatrix.h"
#include "utils/ALStream.h"
#include <fstream>
class ALUCharMatrixTest:public GPTest
{
    public:
        virtual void run()
        {
            ALSp<ALStream> input = ALStreamFactory::readFromFile("a1a.tra");
            ALSp<ALFloatMatrix> origin = ALFloatMatrix::load(input.get());
            ALSp<ALUCharMatrix> ucharMatrix = ALUCharMatrix::create(origin.get());
            std::ofstream output("output/ALUCharMatrix_test.txt");
            ucharMatrix->print(output);
        }
        ALUCharMatrixTest(){}
        virtual ~ALUCharMatrixTest(){}
};

static GPTestRegister<ALUCharMatrixTest> a("ALUCharMatrixTest");
