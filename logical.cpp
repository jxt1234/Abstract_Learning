#include "ALHead.h"
#include "learn/ALLearnFactory.h"
#include "core/ALExpanderFactory.h"
#include "core/ALILabeldMethod.h"
#include "core/ALLabeldMethodFactory.h"
#include "learn/ALDecisionTree.h"
#include "learn/ALGMMClassify.h"
#include "compose/ALRandomForestMatrix.h"
#include "loader/ALStandardLoader.h"
#include "learn/ALLogicalRegress.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <math.h>
#include "learn/ALRegressor.h"
using namespace std;
int main()
{
    ALSp<ALFloatMatrix> X;
    ALSp<ALFloatMatrix> Y;
#if 1
    ALSp<ALFloatMatrix> origin;
    {
        ALSp<ALStream> input = ALStreamFactory::readFromFile("../../Data/Face_Dataset/train.data");
        origin = ALFloatMatrix::load(input.get());
        X = ALFloatMatrix::createCropVirtualMatrix(origin.get(), 1, 0, origin->width()-1, origin->height()-1);
        Y = ALFloatMatrix::createCropVirtualMatrix(origin.get(), 0, 0, 0, origin->height()-1);
    }
    {
        ALSp<ALWStream> x = ALStreamFactory::writeForFile("temp_x.out");
        ALSp<ALWStream> y = ALStreamFactory::writeForFile("temp_y.out");
        ALFloatMatrix::quickSave(X.get(), x.get());
        ALFloatMatrix::quickSave(Y.get(), y.get());
    }
#endif
    {
        ALSp<ALStream> x = ALStreamFactory::readFromFile("temp_x.out");
        ALSp<ALStream> y = ALStreamFactory::readFromFile("temp_y.out");
        X = ALFloatMatrix::quickLoad(x.get());
        Y = ALFloatMatrix::quickLoad(y.get());
    }
    {
        ALSp<ALFloatMatrix> Mean = ALFloatMatrix::create(X->width(), 1);
        ALSp<ALFloatMatrix> Sqrt = ALFloatMatrix::create(X->width(), 1);
        ALSp<ALFloatMatrix> Min = ALFloatMatrix::create(X->width(), 1);
        ALSp<ALFloatMatrix> Max = ALFloatMatrix::create(X->width(), 1);
        auto _mean = Mean->vGetAddr(0);
        auto _sqrt = Sqrt->vGetAddr(0);
        auto _min = Min->vGetAddr(0);
        auto _max = Max->vGetAddr(0);
        size_t sum = 0;
        for (int i=0;i<X->width(); ++i)
        {
            _mean[i] = 0;
            _sqrt[i] = 0;
            _min[i] = 2.0;
            _max[i] = -2.0;
        }
        for (int i=0; i<X->height(); ++i)
        {
            auto x = X->vGetAddr(i);
            auto y = Y->vGetAddr(i);
            if (ZERO(y[0]))
            {
                continue;
            }
            sum++;
            for (int j=0; j<X->width(); ++j)
            {
                _mean[j] += x[j];
                if (x[j]<_min[j])
                {
                    _min[j] = x[j];
                }
                if (x[j]>_max[j])
                {
                    _max[j] = x[j];
                }
            }
        }
        for (int i=0;i<X->width(); ++i)
        {
            _mean[i] = _mean[i]/(ALFLOAT)(sum);
        }
        for (int i=0; i<X->height(); ++i)
        {
            auto x = X->vGetAddr(i);
            auto y = Y->vGetAddr(i);
            if (ZERO(y[0]))
            {
                continue;
            }
            for (int j=0; j<X->width(); ++j)
            {
                _sqrt[j] += (x[j]- _mean[j])*(x[j]-_mean[j]);
            }
        }
        for (int i=0;i<X->width(); ++i)
        {
            _sqrt[i] = _sqrt[i]/(ALFLOAT)(sum);
        }
        cout << "Sum: "<<sum <<endl;
        cout << "Mean:\n";
        ALFloatMatrix::print(Mean.get(), cout);
        cout << "\nSqr:\n";
        ALFloatMatrix::print(Sqrt.get(), cout);
        cout << "\nMin:\n";
        ALFloatMatrix::print(Min.get(), cout);
        cout << "\nMax:\n";
        ALFloatMatrix::print(Max.get(), cout);
        cout <<'\n';
    }
    //ALSp<ALISuperviseLearner> learner = new ALLogicalRegress(10000, 1.0);
    ALSp<ALISuperviseLearner> learner = new ALDecisionTree(5);
    //ALSp<ALISuperviseLearner> learner = new ALRandomForestMatrix(25);
    //ALSp<ALISuperviseLearner> learner = new ALGMMClassify(3);
    //ALSp<ALISuperviseLearner> learner = new ALRegressor;
    ALSp<ALIMatrixPredictor> detected = learner->vLearn(X.get(),Y.get());
    //ALSp<ALFloatMatrix> YP = ALFloatMatrix::create(Y->width(), Y->height());
    //detected->vPredict(X.get(), YP.get());
    ALSp<ALFloatMatrix> YP = ALFloatMatrix::create(2, Y->height());
    detected->vPredictProbability(X.get(), YP.get());
    if (0)
    {
        ofstream output_temp("temp.txt");
    ALFloatMatrix::print(YP.get(), output_temp);
    }
    size_t po = 0;
    size_t pp = 0;
    size_t fo = 0;
    size_t fp = 0;
    for (size_t i=0; i<YP->height(); ++i)
    {
        auto y = *(Y->vGetAddr(i));
        auto yp = YP->vGetAddr(i)[0];
        if (yp > 0.02 && y > 0.5)
        {
            pp++;
        }
        if (yp <=0.02 && y <=0.5)
        {
            fp++;
        }
        if (y > 0.5)
        {
            po++;
        }
        else
        {
            fo++;
        }
    }
    printf("PP/PO: %ld/%ld, %f, FP/FO: %ld/%ld, %f\n", pp, po, (double)pp/(double)po, fp, fo, (double)fp/(double)fo);
    ofstream of("model_logical.out");
    detected->vPrint(of);

    return 0;
}
