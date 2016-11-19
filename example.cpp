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

//#define PREFIX "/Users/jiangxiaotang/Documents/Abstract_Learning/"
#define PREFIX

static std::string readAll(const char* file)
{
    std::ostringstream output;
    std::ifstream input(file);
    output << input.rdbuf();
    return output.str();
}

int test_main(int argc, char* argv[])
{
    ALSp<ALStream> inputStream = ALStreamFactory::readFromFile(PREFIX"../data/imdb2/train_x.txt");
    ALSp<ALVaryArray> array = ALVaryArray::create(inputStream.get());
    ALSp<ALStream> propStream = ALStreamFactory::readFromFile(PREFIX"../data/imdb2/train_y.txt");
    auto jsonString = readAll(PREFIX"./res/cnn/lstm_imdb.json");
    auto jsonObject = cJSON_Parse(jsonString.c_str());
    ALSp<ALVaryArrayLearner> learner = new ALVaryArrayLearner(jsonObject);
    ALSp<ALFloatMatrix> Y = ALFloatMatrix::load(propStream.get());
    
    learner->train(array.get(), Y.get());
    
    inputStream = ALStreamFactory::readFromFile(PREFIX"../data/imdb2/test_x.txt");
    ALSp<ALVaryArray> predictArray = ALVaryArray::create(inputStream.get());
    ALSp<ALFloatMatrix> YP = ALFloatMatrix::create(2, predictArray->size());
    inputStream = ALStreamFactory::readFromFile(PREFIX"../data/imdb2/test_y.txt");
    ALSp<ALFloatMatrix> YP_Real = ALFloatMatrix::load(inputStream.get());
    
    learner->predict(predictArray.get(), YP.get());
    {
        std::ofstream output(PREFIX"./output/test_lstm_imdb.txt");
        ALFloatMatrix::print(YP.get(), output);
    }
    auto h = YP->height();
    size_t correct = 0;
    for (size_t i=0; i<h; ++i)
    {
        auto yp = YP->vGetAddr(i);
        auto y = YP_Real->vGetAddr(i);
        size_t yi = *y;
        if (yp[yi]>yp[1-yi])
        {
            correct++;
        }
    }
    
    std::cout<< correct << "/"<<h<<"\n";

    return 1;
}

int main(int argc, char* argv[])
{
    ALAUTOTIME;
    test_main(argc, argv);
    return 1;
}
