#ifndef LEARN_ALDISCRETEC45TREE_H
#define LEARN_ALDISCRETEC45TREE_H
#include <ostream>
#include "ALISuperviseLearner.h"
#include "math/ALUCharMatrix.h"
class ALDiscreteC45Tree : public ALISuperviseLearner
{
public:
    ALDiscreteC45Tree();
    virtual ~ALDiscreteC45Tree();
    
    /*Assert X->height() == Y->height()*/
    virtual ALIMatrixPredictor* vLearn(const ALFloatMatrix* X, const ALFloatMatrix* Y) const override;
private:
    
};
#endif
