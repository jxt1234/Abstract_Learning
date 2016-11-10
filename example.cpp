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
    ALSp<ALStream> inputStream = ALStreamFactory::readFromFile("/Users/jiangxiaotang/Documents/Abstract_Learning/../data/imdb2/train_x.txt");
    ALSp<ALVaryArray> array = ALVaryArray::create(inputStream.get());
    ALSp<ALStream> propStream = ALStreamFactory::readFromFile("/Users/jiangxiaotang/Documents/Abstract_Learning/../data/imdb2/train_y.txt");
    auto jsonString = readAll("/Users/jiangxiaotang/Documents/Abstract_Learning/./res/cnn/softmax_imdb.json");
    auto jsonObject = cJSON_Parse(jsonString.c_str());
    ALSp<ALVaryArrayLearner> learner = new ALVaryArrayLearner(jsonObject);
    ALSp<ALFloatMatrix> Y = ALFloatMatrix::load(propStream.get());
    ALSp<ALFloatMatrix> YP = ALFloatMatrix::create(2, Y->height());
    
    learner->train(array.get(), Y.get());
    learner->predict(array.get(), YP.get());
    {
        std::ofstream output("/Users/jiangxiaotang/Documents/Abstract_Learning/output/test_lstm_imdb.txt");
        ALFloatMatrix::print(YP.get(), output);
    }
    auto h = Y->height();
    size_t correct = 0;
    for (size_t i=0; i<h; ++i)
    {
        auto yp = YP->vGetAddr(i);
        auto y = Y->vGetAddr(i);
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
