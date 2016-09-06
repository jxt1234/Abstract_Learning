#include "core/ALBasicExpander.h"

ALARExpander::ALARExpander(const ALARStructure& ar):mAR(ar)
{
    mL = mAR.l*mAR.w+mAR.c;
}

ALARExpander::~ALARExpander()
{
}

bool ALARExpander::vExpand(const ALFloatData* dat, ALFLOAT* dst) const
{
    ALASSERT(NULL!=dat);
    ALASSERT(NULL!=dst);
    if (mAR.w > dat->num())
    {
        return false;
    }
    /*Delayed*/
    for (int i=0; i<mAR.d; ++i)
    {
        dat = dat->front();
        if (NULL == dat)
        {
            break;
        }
    }
    if (NULL == dat)
    {
        return false;
    }
    /*expand data*/
    ALFLOAT* xCur = dst;
    for (int i=0; i<mAR.l; ++i)
    {
        if (NULL == dat)
        {
            return false;
        }
        for (int j=0; j<mAR.w; ++j)
        {
            *(xCur++) = dat->value(j);
        }
        dat = dat->front();
    }
    for (int c=0; c<mAR.c; ++c)
    {
        *(xCur++) = 1.0;
    }
    return true;
}

void ALARExpander::vPrint(std::ostream& out) const
{
    out <<std::endl;
    for (int i=0; i<mAR.l; ++i)
    {
        for (int j=0; j<mAR.w; ++j)
        {
            out << "x(t-"<<(mAR.d + i) << ")["<<j<<"]"<<",";
        }
    }
    for (int c=0; c<mAR.c; ++c)
    {
        out << "1,";
    }
    out <<std::endl;
}
