#include "learn/ALCGPRegressor.h"
#include "learn/ALRegressor.h"
#include "learn/ALDivider.h"
#include "learn/ALStepRegressor.h"

class _DirectExpander:public ALIExpander
{
public:
    _DirectExpander(int b):mBack(b){ALASSERT(b>=0);}
    virtual ~_DirectExpander(){}
    virtual int vLength() const override{return 1;}
    virtual bool vExpand(const ALFloatData* d, ALFLOAT* dst) const override
    {
        for (int i=0; i<mBack; ++i)
        {
            d = d->front();
            if (NULL == d)
            {
                dst[0] = 0.0;
                return false;
            }
        }
        dst[0] = d->value(0);
        return true;
    }
    virtual void vPrint(std::ostream& os) const override
    {
    }
    int mBack;
};

class _DiffExpander:public ALIExpander
{
public:
    _DiffExpander(int b):mBack(b){ALASSERT(b>=0);}
    virtual ~_DiffExpander(){}

    virtual int vLength() const override
    {
        return 1;
    }
    virtual bool vExpand(const ALFloatData* d, ALFLOAT* dst) const override
    {
        for (int i=0; i<mBack; ++i)
        {
            d = d->front();
            if (NULL == d)
            {
                dst[0] = 0.0;
                return false;
            }
        }
        if (NULL == d->front())
        {
            dst[0] = 0.0;
            return false;
        }
        dst[0] = d->value(0) - d->front()->value(0);
        return true;
    }
    virtual void vPrint(std::ostream& os) const override
    {
    }
    int mBack;
};


static ALIExpander* _createExpander(int type, int back)
{
    switch(type)
    {
        case 0:
            return new _DirectExpander(back);
        case 1:
            return new _DiffExpander(back);
    }
    ALASSERT(0);
    return NULL;
}

ALCGPRegressor::TrainBox::TrainBox()
{
    mType = MODEL;
    mModelType = 0;
    
    mDivideType = 0;
    mDivideRate = 0;
    mDivideBackLength = 0;
    mDown = NULL;
    mRight = NULL;
}
ALCGPRegressor::TrainBox::~TrainBox()
{
}


//FIXME
void ALCGPRegressor::TrainBox::map(double type, double rate)
{
    ALASSERT(0<=type&&1>=type);
    ALASSERT(0<=rate&&1>=rate);
    mType = type > 0.5 ? DIVIDE : MODEL;
    mDivideRate = rate;
    if (type > 0.5) type -=0.5;
    if (type > 0.25)
    {
        mDivideType = DIFF;
        type-=0.25;
    }
    else
    {
        mDivideType = DIRECT;
    }
    mDivideBackLength = type*40;
}

ALFloatPredictor* ALCGPRegressor::TrainBox::learn(const ALLabeldData* data) const
{
    ALFloatPredictor* result = NULL;
    switch (mType)
    {
        case MODEL:
        {
            ALSp<ALIChainLearner> learner = new ALStepRegressor(mDivideBackLength);
            result = learner->vLearn(data);
            break;
        }
        case DIVIDE:
        {
            ALSp<ALIExpander> xe = _createExpander(mDivideType, mDivideBackLength);
            ALSp<ALFloatPredictor> sep = ALSepPreditor::train(xe, data, mDivideRate);
            ALSp<ALLabeldData> left = new ALLabeldData;
            ALSp<ALLabeldData> right = new ALLabeldData;
            for (auto p : data->get())
            {
                auto v = p.first;
                auto d = p.second;
                ALFLOAT temp;
                if (xe->vExpand(d, &temp))
                {
                    continue;
                }
                ALFLOAT judge = sep->vPredict(d);
                if (judge > 0)
                {
                    right->insert(v, d);
                }
                else
                {
                    left->insert(v, d);
                }
            }
            /*Use left and right learner*/
            ALASSERT(mDown!=NULL);
            ALASSERT(mRight!=NULL);
            ALSp<ALFloatPredictor> l = mDown->learn(left.get());
            ALSp<ALFloatPredictor> r = mRight->learn(right.get());
            result = new ALComposePredictor(sep, l, r);
            break;
        }
        default:
            ALASSERT(0);
            break;
    }
    return result;
}
ALCGPRegressor::ALCGPRegressor(int w, int h)
{
    ALASSERT(0<w && 0<h);
    mW = w;
    mH = h;
    mBoxes = new TrainBox[w*h];
    mDefaultBox = new TrainBox;
    ALASSERT(NULL!=mBoxes);
    ALASSERT(NULL!=mDefaultBox);
    for (int i=0; i<h; ++i)
    {
        TrainBox* line = mBoxes + i*w;
        for (int j=0; j<w; ++j)
        {
            TrainBox* target = line+j;
            TrainBox* down = i<h-1 ? target+w : mDefaultBox;
            TrainBox* right = j<w-1 ? target+1 : mDefaultBox;
            target->setOutput(right, down);
        }
    }
}
ALCGPRegressor::~ALCGPRegressor()
{
    ALASSERT(NULL!=mBoxes);
    ALASSERT(NULL!=mDefaultBox);
    delete [] mBoxes;
    delete mDefaultBox;
}

int ALCGPRegressor::map(double* p, int n)
{
    if (NULL == p)
    {
        return 2*mW*mH;
    }
    ALASSERT(n>=2*mW*mH);
    for (int i=0; i<mH; ++i)
    {
        TrainBox* line = mBoxes+i*mW;
        for (int j=0; j<mW; ++j)
        {
            double* target_p = p+(i*mW+j)*2;
            TrainBox* target = line+j;
            target->map(target_p[0], target_p[1]);
        }
    }
    return 2*mW*mH;
}

ALFloatPredictor* ALCGPRegressor::vLearn(const ALLabeldData* data) const
{
    return mBoxes->learn(data);
}

