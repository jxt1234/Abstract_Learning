#include "learn/ALMatrixSelector.h"
ALMatrixSelector::ALMatrixSelector(const std::vector<int> positions)
{
    ALASSERT(positions.size()>0);
    mPos = new int[positions.size()];
    mN = positions.size();
    for (int i=0; i<positions.size(); ++i)
    {
        mPos[i] = positions[i];
    }
    mMaxPos = mPos[mN-1];
}

ALMatrixSelector::~ALMatrixSelector()
{
    delete [] mPos;
}
ALFloatMatrix* ALMatrixSelector::vTransform(const ALFloatMatrix* origin) const
{
    ALASSERT(NULL!=origin);
    ALASSERT(origin->width()>mMaxPos);
    auto h = origin->height();
    ALFloatMatrix* result = ALFloatMatrix::create(mN, h);
    for (size_t i=0; i<h; ++i)
    {
        auto _r = result->vGetAddr(i);
        auto _o = origin->vGetAddr(i);
        for (int j=0; j<mN; ++j)
        {
            _r[j] = _o[mPos[j]];
        }
    }
    return result;
}


void ALMatrixSelector::vPrint(std::ostream& output) const
{
    output << "<Select>";
    for (size_t i=0; i<mN; ++i)
    {
        output << mPos[i] << " ";
    }
    output << "</Select>\n";
}
