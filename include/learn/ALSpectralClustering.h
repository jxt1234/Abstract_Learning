#ifndef INCLUDE_LEARN_ALSPECTRALCLUSTERING_H
#define INCLUDE_LEARN_ALSPECTRALCLUSTERING_H
#include "ALIUnSuperLearner.h"
#include "core/ALIExpander.h"
#include "math/ALFloatMatrix.h"
class ALSpectralClustering:public ALIUnSuperLearner
{
    public:
        ALSpectralClustering(ALSp<ALIExpander> X, ALFLOAT sigma, size_t k, size_t class_number);
        virtual ~ALSpectralClustering();
        virtual ALIMatrixPredictor* vLearn(const ALFloatMatrix* X) const override;
        ALFloatMatrix* classify(ALFloatMatrix* X) const;
    private:
        ALFloatMatrix* relativeGauss(ALFloatMatrix* Data) const;
        ALSp<ALIExpander> mX;
        ALFLOAT mSigma;
        size_t mK;
        size_t mNum;
};
#endif
