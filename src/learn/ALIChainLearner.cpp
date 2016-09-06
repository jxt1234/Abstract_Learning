#include "learn/ALIChainLearner.h"
#include "math/ALFloatMatrix.h"

class matrixpredictorWarp:public ALFloatPredictor
{
public:
    matrixpredictorWarp(ALSp<ALIMatrixPredictor> p, ALSp<ALIExpander> xe):mP(p), mXe(xe)
    {
        mN = xe->vLength();
        ALASSERT(mN>0);
    }
    virtual ALFLOAT vPredict(const ALFloatData* data) const override
    {
        ALSp<ALFloatMatrix> X = ALFloatMatrix::create(mN, 1);
        bool res = mXe->vExpand(data, X->vGetAddr());
        if (!res)
        {
            return 0;
        }
        ALSp<ALFloatMatrix> Y = ALFloatMatrix::create(1, 1);
        mP->vPredict(X.get(), Y.get());
        return *(Y->vGetAddr());
    }
    virtual void vPrint(std::ostream& out) const override
    {
        out << "<WarpPredictor>"<<std::endl;
        mP->vPrint(out);
        out << "</WarpPredictor>"<<std::endl;
    }
    ALSp<ALIMatrixPredictor> mP;
    ALSp<ALIExpander> mXe;
    int mN;
};
class matrixlearnerWarp:public ALIChainLearner
{
public:
    matrixlearnerWarp(ALSp<ALISuperviseLearner> b, ALSp<ALIExpander> x)
    {
        mXe = x;
        mBasic = b;
    }
    virtual ~matrixlearnerWarp()
    {
    }
    virtual ALFloatPredictor* vLearn(const ALLabeldData* data) const override
    {
        ALSp<ALFloatMatrix> X;
        ALSp<ALFloatMatrix> YT;
        ALIExpander::expandXY(mXe.get(), data, X, YT);
        if (NULL == X.get() || NULL == YT.get())
        {
            return new ALDummyFloatPredictor;
        }
        ALSp<ALFloatMatrix> Y = ALFloatMatrix::transpose(YT.get());
        ALSp<ALIMatrixPredictor> m = mBasic->vLearn(X.get(), Y.get());
        if (NULL == m.get())
        {
            return new ALDummyFloatPredictor;
        }
        return new matrixpredictorWarp(m, mXe);
    }
    ALSp<ALISuperviseLearner> mBasic;
    ALSp<ALIExpander> mXe;
};
ALIChainLearner* ALIChainLearner::createFromBasic(ALSp<ALISuperviseLearner> basic, ALSp<ALIExpander> Xe)
{
    return new matrixlearnerWarp(basic, Xe);
}

ALIChainLearner::Error ALIChainLearner::computeError(const ALLabeldData* data, const ALFloatPredictor* p)
{
    Error error;
    int num = 0;
    ALFLOAT totalerror = 0.0;
    for (auto d : data->get())
    {
        auto real = d.first;
        auto predict = p->vPredict(d.second);
        num++;
        totalerror+= (real-predict)*(real-predict);
    }
    error.num = num;
    error.sum = totalerror;
    return error;
}
