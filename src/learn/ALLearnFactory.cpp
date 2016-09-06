#include "learn/ALISuperviseLearner.h"
#include "learn/ALLearnFactory.h"
#include <iostream>

using namespace std;
ALISuperviseLearner* ALLearnFactory::createLearner()
{
    /*TODO*/
    return NULL;
}
ALFLOAT ALLearnFactory::crossValidate(ALIChainLearner* l, const ALLabeldData* c, int div)
{
    ALASSERT(NULL!=l);
    ALASSERT(NULL!=c);
    auto points = c->get();
    int totalNum = 0;
    ALFLOAT totalError = 0;
    for (int i=0; i<div; ++i)
    {
        /*Divide*/
        ALSp<ALLabeldData> train = new ALLabeldData;
        ALSp<ALLabeldData> predict = new ALLabeldData;
        auto iter = points.begin();
        int pos = 0;
        auto predictSta = (i)*(points.size() /div);
        auto predictFin = (i+1)*(points.size() /div);
        for (; iter!=points.end(); iter++, pos++)
        {
            if (pos < predictSta || pos >= predictFin)
            {
                train->insert(iter->first, iter->second);
            }
            else
            {
                predict->insert(iter->first, iter->second);
            }
        }
        /*Train and compare*/
        ALSp<ALFloatPredictor> p = l->vLearn(train.get());
        for (auto datapoint : predict->get())
        {
            ALFLOAT value = datapoint.first;
            auto current = datapoint.second;
            ALFLOAT err = value - p->vPredict(current);
            totalError += (err*err);
            totalNum++;
        }
    }
    return totalNum/1.0/totalError;
}

ALFLOAT ALLearnFactory::crossValidateForClassify(ALIChainLearner* l, const ALLabeldData* c, int div)
{
    ALASSERT(NULL!=l);
    ALASSERT(NULL!=c);
    auto points = c->get();
    int totalNum = 0;
    ALFLOAT match = 0;
    for (int i=0; i<div; ++i)
    {
        /*Divide*/
        ALSp<ALLabeldData> train = new ALLabeldData;
        ALSp<ALLabeldData> predict = new ALLabeldData;
        auto iter = points.begin();
        int pos = 0;
        auto predictSta = (i)*(points.size() /div);
        auto predictFin = (i+1)*(points.size() /div);
        for (; iter!=points.end(); iter++, pos++)
        {
            if (pos < predictSta || pos >= predictFin)
            {
                train->insert(iter->first, iter->second);
            }
            else
            {
                predict->insert(iter->first, iter->second);
            }
        }
        /*Train and compare*/
        ALSp<ALFloatPredictor> p = l->vLearn(train.get());
        for (auto datapoint : predict->get())
        {
            ALFLOAT value = datapoint.first;
            auto current = datapoint.second;
            ALFLOAT err = value - p->vPredict(current);
            if (ZERO(err))
            {
                match++;
            }
            totalNum++;
        }
    }
    return match/1.0/totalNum;
}

ALFLOAT ALLearnFactory::crossValidateForClassify(ALISuperviseLearner* l, const ALFloatMatrix* X, const ALFloatMatrix* Y, int div)
{
    ALASSERT(NULL!=l);
    ALASSERT(NULL!=X);
    ALASSERT(NULL!=Y);
    ALASSERT(Y->width() == 1);//TODO
    ALASSERT(X->height() == Y->height());
    auto h = X->height();
    auto xw = X->width();
    auto yw = Y->width();
    auto ph = h/div;
    auto th = h-ph;
    ALSp<ALFloatMatrix> TX = ALFloatMatrix::create(xw, th);
    ALSp<ALFloatMatrix> TY = ALFloatMatrix::create(yw, th);
    ALSp<ALFloatMatrix> PX = ALFloatMatrix::create(xw, ph);
    ALSp<ALFloatMatrix> PY = ALFloatMatrix::create(yw, ph);
    ALSp<ALFloatMatrix> PYY = ALFloatMatrix::create(yw, ph);
    size_t sum = 0;
    size_t match = 0;
    for (int i=0; i<div; ++i)
    {
        size_t ysta = i*h/div;
        size_t yfin = ysta+ph;
        for (size_t i=0; i<ph; ++i)
        {
            auto _y = PY->vGetAddr(i);
            auto y = Y->vGetAddr( i+ysta);
            ::memcpy(_y, y, yw*sizeof(ALFLOAT));
            auto _x = PX->vGetAddr(i);
            auto x = X->vGetAddr( i+ysta);
            ::memcpy(_x, x, xw*sizeof(ALFLOAT));
        }
        for (size_t i=0; i<ysta; ++i)
        {
            auto _y = TY->vGetAddr(i);
            auto y = Y->vGetAddr( i);
            ::memcpy(_y, y, yw*sizeof(ALFLOAT));
            auto _x = TX->vGetAddr(i);
            auto x = X->vGetAddr( i);
            ::memcpy(_x, x, xw*sizeof(ALFLOAT));
        }
        for (size_t i=yfin; i<h; ++i)
        {
            auto _y = TY->vGetAddr(i+ysta-yfin);
            auto y = Y->vGetAddr( i);
            ::memcpy(_y, y, yw*sizeof(ALFLOAT));
            auto _x = TX->vGetAddr(i+ysta-yfin);
            auto x = X->vGetAddr( i);
            ::memcpy(_x, x, xw*sizeof(ALFLOAT));
        }
        ALSp<ALIMatrixPredictor> p = l->vLearn(TX.get(), TY.get());
        p->vPredict(PX.get(), PYY.get());
        for (size_t i=0; i<ph; ++i)
        {
            auto yp = PYY->vGetAddr(i);
            auto y = PY->vGetAddr(i);
            if (ZERO(yp[0]-y[0]))
            {
                match++;
            }
        }
        sum+=ph;
    }
    return match/1.0/sum;
}

ALFLOAT ALLearnFactory::crossValidate(ALISuperviseLearner* l, const ALFloatMatrix* X, const ALFloatMatrix* Y, int div)
{
    ALASSERT(NULL!=l);
    ALASSERT(NULL!=X);
    ALASSERT(NULL!=Y);
    ALASSERT(Y->width() == 1);//TODO
    ALASSERT(X->height() == Y->height());
    auto h = X->height();
    auto xw = X->width();
    auto yw = Y->width();
    auto ph = h/div;
    auto th = h-ph;
    ALSp<ALFloatMatrix> TX = ALFloatMatrix::create(xw, th);
    ALSp<ALFloatMatrix> TY = ALFloatMatrix::create(yw, th);
    ALSp<ALFloatMatrix> PX = ALFloatMatrix::create(xw, ph);
    ALSp<ALFloatMatrix> PY = ALFloatMatrix::create(yw, ph);
    ALSp<ALFloatMatrix> PYY = ALFloatMatrix::create(yw, ph);
    size_t sum = 0;
    ALFLOAT error = 0;
    for (int i=0; i<div; ++i)
    {
        size_t ysta = i*h/div;
        size_t yfin = ysta+ph;
        for (size_t i=0; i<ph; ++i)
        {
            auto _y = PY->vGetAddr(i);
            auto y = Y->vGetAddr( i+ysta);
            ::memcpy(_y, y, yw*sizeof(ALFLOAT));
            auto _x = PX->vGetAddr(i);
            auto x = X->vGetAddr( i+ysta);
            ::memcpy(_x, x, xw*sizeof(ALFLOAT));
        }
        for (size_t i=0; i<ysta; ++i)
        {
            auto _y = TY->vGetAddr(i);
            auto y = Y->vGetAddr( i);
            ::memcpy(_y, y, yw*sizeof(ALFLOAT));
            auto _x = TX->vGetAddr(i);
            auto x = X->vGetAddr( i);
            ::memcpy(_x, x, xw*sizeof(ALFLOAT));
        }
        for (size_t i=yfin; i<h; ++i)
        {
            auto _y = TY->vGetAddr(i+ysta-yfin);
            auto y = Y->vGetAddr( i);
            ::memcpy(_y, y, yw*sizeof(ALFLOAT));
            auto _x = TX->vGetAddr(i+ysta-yfin);
            auto x = X->vGetAddr( i);
            ::memcpy(_x, x, xw*sizeof(ALFLOAT));
        }
        ALSp<ALIMatrixPredictor> p = l->vLearn(TX.get(), TY.get());
        p->vPredict(PX.get(), PYY.get());
        for (size_t i=0; i<ph; ++i)
        {
            auto yp = PYY->vGetAddr(i);
            auto y = PY->vGetAddr(i);
            error += (yp[0] - y[0])*(yp[0]-y[0]);
        }
        sum+=ph;
    }
    return sum/1.0/error;
}

