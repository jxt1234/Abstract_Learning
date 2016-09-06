#ifndef INCLUDE_LEARN_ALCGPREGRESSOR_H
#define INCLUDE_LEARN_ALCGPREGRESSOR_H
#include "ALIChainLearner.h"
class ALCGPRegressor:public ALIChainLearner
{
public:
    ALCGPRegressor(int w, int h);
    virtual ~ALCGPRegressor();
    
    int map(double* p, int n);

    virtual ALFloatPredictor* vLearn(const ALLabeldData* data) const override;
private:
    class TrainBox:public ALRefCount
    {
    public:
        TrainBox();
        virtual ~TrainBox();
        void map(double type, double rate);
        void setOutput(TrainBox* right, TrainBox* down)
        {
            mRight = right;
            mDown = down;
        }        
        ALFloatPredictor* learn(const ALLabeldData* data) const;
    private:
        enum
        {
//            BACK,
            DIVIDE,
            MODEL,
            TYPENUM
        };
        
        //BACK/DIVIDE/MODEL
        int mType;
        
        //For DIVIDE
        enum
        {
            DIRECT = 0,
            DIFF = 1,
            DIVIDETYPE = 2
        };
        int mDivideType;
        int mDivideBackLength;
        ALFLOAT mDivideRate;
        
        //For MODEL
        int mModelType;
        
        TrainBox* mRight;
        TrainBox* mDown;
    };
    
    TrainBox* mBoxes;
    int mW;
    int mH;
    TrainBox* mDefaultBox;
};
#endif
