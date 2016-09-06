#ifndef INCLUDE_COMPOSE_ALCOMPOSECLASSIFIER_H
#define INCLUDE_COMPOSE_ALCOMPOSECLASSIFIER_H
#include <map>
#include "ALClassifierSet.h"
class ALComposeClassifier:public ALIMatrixPredictor
{
public:
    typedef std::pair<ALFLOAT, std::vector<int>> PREDICTORMETA;
    
    ALComposeClassifier(const ALClassifierSet* set);
    virtual ~ALComposeClassifier();

    /*Assert X->height() == Y->height()*/
    virtual void vPredict(const ALFloatMatrix* X, ALFloatMatrix* Y/*Output*/) const;

    /*For Classify*/
    virtual void vPredictProbability(const ALFloatMatrix* X, ALFloatMatrix* Y/*Output*/) const;
    virtual const ALFloatMatrix* vGetPossiableValues() const;
    virtual void vPrint(std::ostream& output) const;

private:
    std::vector<ALSp<ALIMatrixPredictor>> mPredictors;
    std::map<const ALIMatrixPredictor*, PREDICTORMETA> mMetas;
    
    ALSp<ALFloatMatrix> mPossibles;
};

#endif
