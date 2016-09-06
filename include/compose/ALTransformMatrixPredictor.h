#ifndef INCLUDE_COMPOSE_ALTRANSFORMMATRIXPREDICTOR_H
#define INCLUDE_COMPOSE_ALTRANSFORMMATRIXPREDICTOR_H
#include "ALHead.h"
#include "learn/ALISuperviseLearner.h"

class ALTransformMatrixPredictor:public ALIMatrixPredictor
{
public:
    ALTransformMatrixPredictor();
    virtual ~ALTransformMatrixPredictor();
};



#endif
