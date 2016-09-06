#ifndef INCLUDE_LEARN_ALDIVIDER_H
#define INCLUDE_LEARN_ALDIVIDER_H
#include "ALIChainLearner.h"
/*Double divider learner, divide the chain into two chain for other learner to build ALFloatPredictor*/
class ALDivider:public ALIChainLearner
{
    public:
        ALDivider(ALFLOAT div, int step, ALIChainLearner* left, ALIChainLearner* right);
        virtual ~ALDivider();
        virtual ALFloatPredictor* vLearn(const ALLabeldData* data) const;
    private:
        ALFLOAT mPer;
        ALSp<ALIExpander> mXe;
        ALIChainLearner* mL;
        ALIChainLearner* mR;
};
class ALComposePredictor:public ALFloatPredictor
{
    public:
        ALComposePredictor(ALSp<ALFloatPredictor> sep, ALSp<ALFloatPredictor> left, ALSp<ALFloatPredictor> right);
        virtual ~ALComposePredictor();
        virtual ALFLOAT vPredict(const ALFloatData* data) const;
        virtual void vPrint(std::ostream& out) const;
    private:
        ALSp<ALFloatPredictor> s;
        ALSp<ALFloatPredictor> l;
        ALSp<ALFloatPredictor> r;
};

class ALSepPreditor:public ALFloatPredictor
{
public:
    ALSepPreditor(ALSp<ALIExpander> Xe, ALFLOAT p):mXe(Xe), mP(p)
    {
        ALASSERT(1 == mXe->vLength());
    }
    virtual ~ALSepPreditor()
    {
    }
    virtual ALFLOAT vPredict(const ALFloatData* d) const override
    {
        ALFLOAT x;
        mXe->vExpand(d, &x);
        return x - mP;
    }
    static ALSepPreditor* train(ALSp<ALIExpander> xe, const ALFloatDataChain* data, ALFLOAT per);
    static ALSepPreditor* train(ALSp<ALIExpander> xe, const ALLabeldData* data, ALFLOAT per);
private:
    ALSp<ALIExpander> mXe;
    ALFLOAT mP;
};


#endif
