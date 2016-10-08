#ifndef LEARN_CNN_CNNPREDICTOR_H
#define LEARN_CNN_CNNPREDICTOR_H
#include "learn/ALISuperviseLearner.h"
#include "LayerWrap.h"
namespace ALCNN {
    class CNNPredictor : public ALIMatrixPredictor
    {
    public:
        virtual void vPredict(const ALFloatMatrix* X, ALFloatMatrix* Y/*Output*/) const override;
        virtual void vPredictProbability(const ALFloatMatrix* X, ALFloatMatrix* Y/*Output*/) const override;
        virtual const ALFloatMatrix* vGetPossiableValues() const override;
        
        CNNPredictor(ALSp<LayerWrap> net, ALSp<ALFloatMatrix> prob);
        virtual ~ CNNPredictor();

    private:
        mutable ALSp<LayerWrap> mNet;
        ALSp<ALFloatMatrix> mProbability;

    };
}
#endif
