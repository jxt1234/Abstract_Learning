#include "core/ALBasicExpander.h"
#include "utils/ALUtils.h"
#include "learn/ALDivider.h"

#include <iostream>
using namespace std;
ALDivider::ALDivider(ALFLOAT div, int step, ALIChainLearner* left, ALIChainLearner* right):mPer(div), mL(left), mR(right)
{
    mL->addRef();
    mR->addRef();
    ALARStructure ar;
    ar.d = step;
    ar.c = 0;
    mXe = new ALARExpander(ar);
}

ALDivider::~ALDivider()
{
    mL->decRef();
    mR->decRef();
}

ALFloatPredictor* ALDivider::vLearn(const ALLabeldData* data) const
{
    ALASSERT(NULL!=data);
    if (data->size() <=0)
    {
        return new ALDummyFloatPredictor;
    }
    ALSp<ALFloatPredictor> div = ALSepPreditor::train(mXe, data, mPer);
    /*Use the divider to seperate chain*/
    ALSp<ALLabeldData> left = new ALLabeldData;
    ALSp<ALLabeldData> right = new ALLabeldData;
    for (auto p : data->get())
    {
        auto v = p.first;
        auto d = p.second;
        ALFLOAT judge = div->vPredict(d);
        if (judge > 0)
        {
            right->insert(v, d);
        }
        else
        {
            left->insert(v, d);
        }
    }
    /*Use left and right learner*/
    ALSp<ALFloatPredictor> l = mL->vLearn(left.get());
    ALSp<ALFloatPredictor> r = mR->vLearn(right.get());
    ALComposePredictor* res = new ALComposePredictor(div, l, r);
    return res;
}

/*divide = (max-min)*mPer + min*/
ALSepPreditor* ALSepPreditor::train(ALSp<ALIExpander> xe, const ALLabeldData* data, ALFLOAT per)
{
    ALASSERT(NULL!=xe.get());
    ALASSERT(NULL!=data);
    ALASSERT(1 == xe->vLength());
    bool first = true;
    ALFLOAT min=0, max=0;
    ALFLOAT dstData = 0.0f;
    for (auto p : data->get())
    {
        if (!xe->vExpand(p.second, &dstData))
        {
            continue;
        }
        ALFLOAT v = dstData;
        if (first)
        {
            min = v;
            max = v;
            first = false;
        }
        if (v > max) max = v;
        else if (v < min) min = v;
    }
    ALFLOAT divide = (min + per*(max - min));
    return new ALSepPreditor(xe, divide);
}


ALSepPreditor* ALSepPreditor::train(ALSp<ALIExpander> xe, const ALFloatDataChain* chain, ALFLOAT per)
{
    ALASSERT(NULL!=xe.get());
    ALASSERT(NULL!=chain);
    ALASSERT(1 == xe->vLength());
    bool first = true;
    ALFLOAT min=0, max=0;
    ALFLOAT dstData = 0.0f;
    for (auto p : chain->get())
    {
        if (!xe->vExpand(p, &dstData))
        {
            continue;
        }
        ALFLOAT v = dstData;
        if (first)
        {
            min = v;
            max = v;
            first = false;
        }
        if (v > max) max = v;
        else if (v < min) min = v;
    }
    ALFLOAT divide = (min + per*(max - min));
    return new ALSepPreditor(xe, divide);
}

ALComposePredictor::ALComposePredictor(ALSp<ALFloatPredictor> sep, ALSp<ALFloatPredictor> left, ALSp<ALFloatPredictor> right)
{
    s = sep;
    l = left;
    r = right;
}

ALComposePredictor::~ALComposePredictor()
{
}

ALFLOAT ALComposePredictor::vPredict(const ALFloatData* data) const
{
    ALFLOAT v = s->vPredict(data);
    ALFLOAT res = 0;
    if (v > 0)
    {
        res = r->vPredict(data);
    }
    else
    {
        res = l->vPredict(data);
    }
    return res;
}
void ALComposePredictor::vPrint(std::ostream& out) const
{
    out << "<Compose>\n";
    out << "<seperator>\n";
    s->vPrint(out);
    out << "</seperator>\n";
    out << "<left>\n";
    l->vPrint(out);
    out << "</left>\n";
    out << "<right>\n";
    r->vPrint(out);
    out << "</right>\n";
    out << "</Compose>\n";
}


