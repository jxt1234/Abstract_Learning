#ifndef INCLUDE_COMPOSE_ALCLASSIFIERSET_H
#define INCLUDE_COMPOSE_ALCLASSIFIERSET_H
#include <vector>
#include "ALHead.h"
#include "learn/ALISuperviseLearner.h"
class ALClassifierSet:public ALRefCount
{
public:
    typedef std::vector<std::pair<ALSp<ALIMatrixPredictor>, long> > SET;
    const SET& get() const {return mSet;}
    
    ALClassifierSet();
    virtual ~ALClassifierSet();
    void clear();
    void push(ALSp<ALIMatrixPredictor> pre, long count);
    
    static ALClassifierSet* merge(const ALClassifierSet* s1, const ALClassifierSet* s2);
    
private:
    SET mSet;
};

#endif
