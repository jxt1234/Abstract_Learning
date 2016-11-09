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
    ALSp<ALStream> inputStream = ALStreamFactory::readFromFile("../data/imdb2/train_x.txt");
    ALSp<ALVaryArray> array = ALVaryArray::create(inputStream.get());
    ALSp<ALStream> propStream = ALStreamFactory::readFromFile("../data/imdb2/train_y.txt");
    auto jsonString = readAll("./res/cnn/lstm_imdb.json");
    auto jsonObject = cJSON_Parse(jsonString.c_str());
    ALSp<ALVaryArrayLearner> learner = new ALVaryArrayLearner(jsonObject);
    ALSp<ALFloatMatrix> Y = ALFloatMatrix::load(propStream.get());

    learner->train(array.get(), Y.get());

    return 1;
}

int main(int argc, char* argv[])
{
    ALAUTOTIME;
    test_main(argc, argv);
    return 1;
}
