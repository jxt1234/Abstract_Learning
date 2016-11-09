#include "ALHead.h"
#include <iostream>
#include <fstream>
#include "utils/ALStream.h"
#include "math/ALFloatMatrix.h"
#include "learn/ALLearnFactory.h"
#include "learn/ALVaryArrayLearner.h"
#include <math.h>
#include <sstream>
#include "cJSON/cJSON.h"
using namespace std;
static std::string readAll(const char* file)
{
    std::ostringstream output;
    std::ifstream input(file);
    output << input.rdbuf();
    return output.str();
}

int test_main(int argc, char* argv[])
{
    ALSp<ALStream> inputStream = ALStreamFactory::readFromFile("../data/lstm.numbers_pre");
    ALSp<ALVaryArray> array = ALVaryArray::create(inputStream.get());
    auto jsonString = readAll(argv[1]);
    auto jsonObject = cJSON_Parse(jsonString.c_str());
    ALSp<ALVaryArrayLearner> learner = new ALVaryArrayLearner(jsonObject);

    learner->train(array.get());

    return 1;
}

int main(int argc, char* argv[])
{
    ALAUTOTIME;
    test_main(argc, argv);
    return 1;
}
