#include "compose/ALClassifierSet.h"
ALClassifierSet::ALClassifierSet()
{
}
ALClassifierSet::~ALClassifierSet()
{
}
void ALClassifierSet::clear()
{
    mSet.clear();
}

void ALClassifierSet::push(ALSp<ALIMatrixPredictor> pre, long count)
{
    mSet.push_back(std::make_pair(pre, count));
}

ALClassifierSet* ALClassifierSet::merge(const ALClassifierSet* s1, const ALClassifierSet* s2)
{
    ALClassifierSet* set = new ALClassifierSet;
    for (auto s : s1->get())
    {
        set->push(s.first, s.second);
    }
    for (auto s : s2->get())
    {
        set->push(s.first, s.second);
    }
    return set;
}
