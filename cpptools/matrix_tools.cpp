#include "ALHead.h"
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include "math/ALFloatMatrix.h"
using namespace std;

static void turnSelf(const char* src, const char* dst)
{ 
    ALSp<ALStream> input = ALStreamFactory::readFromFile(src);
    ALSp<ALFloatMatrix> matrix = ALFloatMatrix::load(input.get());

    ALSp<ALWStream> output = ALStreamFactory::writeForFile(dst);
    ALFloatMatrix::quickSave(matrix.get(), output.get());
}

static void turnBasic(const char* src, const char* dst)
{
    ALSp<ALStream> input = ALStreamFactory::readFromFile(src);
    ALSp<ALFloatMatrix> matrix = ALFloatMatrix::quickLoad(input.get());
    std::ofstream output(dst);
    ALFloatMatrix::print(matrix.get(), output);
}

int main(int argc, char* argv[])
{
    if (argc < 4)
    {
        printf("Usage: ./matrix_tools.out Export src.txt dst \n or \n ./matrix_tools.out Import src dst.txt\n");
        return 0;
    }
    string option = argv[1];
    if (option == "Export")
    {
        printf("Turn unreadable %s to readable %s\n", argv[2], argv[3]);
        turnBasic(argv[2], argv[3]);
        return 0;
    }
    if (option == "Import")
    {
        printf("Turn readable %s to unreadable %s\n", argv[2], argv[3]);
        turnSelf(argv[2], argv[3]);
        return 0;
    }
    return 1;
}
