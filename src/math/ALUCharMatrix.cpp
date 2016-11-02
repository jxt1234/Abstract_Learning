#include "math/ALUCharMatrix.h"

unsigned char* ALUCharMatrix::getLine(size_t s) const
{
    ALASSERT(s<mHeight);
    ALASSERT(s>=0);
    return mContents + s*mWidth;
}
unsigned char ALUCharMatrix::getMaxNumber(size_t s) const
{
    ALASSERT(s<mWidth);
    ALASSERT(s>=0);
    return mMaxNumbers[s];
}
ALFLOAT* ALUCharMatrix::getTypes(size_t w) const
{
    ALASSERT(w<mWidth);
    ALASSERT(w>=0);
    return mRealValues->vGetAddr(w);
}

ALUCharMatrix* ALUCharMatrix::create(const ALFloatMatrix* origin)
{
    ALASSERT(NULL!=origin);
    ALSp<ALFloatMatrix> originT = ALFloatMatrix::transpose(origin);
    auto w = origin->width();
    auto h = origin->height();
    ALUCharMatrix* charResult = new ALUCharMatrix(origin->width(), origin->height());
    
    /*Construct possible Values*/
    unsigned char* maxNumbers = charResult->mMaxNumbers;
    const int MAXNUMBER = 256;
    ALSp<ALFloatMatrix> typeValues = ALFloatMatrix::create(MAXNUMBER, w);
    unsigned int cur =0;
    unsigned int maxTypesNumber = 0;
    for (int i=0; i<w; ++i)
    {
        auto possibles = typeValues->vGetAddr(i);
        cur = 0;
        auto xt = originT->vGetAddr(i);
        for (int j=0; j<h; ++j)
        {
            auto target = charResult->mContents+w*j+i;
            bool find = false;
            auto x = xt[j];
            for (int k=0; k<cur; ++k)
            {
                if (ZERO(x-possibles[k]))
                {
                    *target = k;
                    find = true;
                    break;
                }
            }
            if (!find)
            {
                possibles[cur] = x;
                *target = cur;
                cur++;
                ALASSERT(cur<MAXNUMBER);
            }
        }
        maxNumbers[i] = cur;
        if (maxTypesNumber < cur)
        {
            maxTypesNumber = cur;
        }
    }
    charResult->mRealValues = ALFloatMatrix::create(maxTypesNumber, w);
    ALFloatMatrix::zero(charResult->mRealValues.get());
    for (int i=0; i<w; ++i)
    {
        auto dst = charResult->mRealValues->vGetAddr(i);
        auto src = typeValues->vGetAddr(i);
        ::memcpy(dst, src, sizeof(ALFLOAT)*maxTypesNumber);
    }
    return charResult;
}

ALUCharMatrix::ALUCharMatrix(size_t w, size_t h)
{
    ALASSERT(w>0);
    ALASSERT(h>0);
    mWidth = w;
    mHeight = h;
    mContents = new unsigned char[w*h];
    ::memset(mContents, 0, sizeof(unsigned char)*w*h);
    mMaxNumbers = new unsigned char[w];
    ::memset(mMaxNumbers, 0, sizeof(unsigned char)*w);
}

ALUCharMatrix::~ALUCharMatrix()
{
    delete [] mContents;
    delete [] mMaxNumbers;
}
void ALUCharMatrix::print(std::ostream& output) const
{
    for (int i=0; i<mWidth; ++i)
    {
        int u = mMaxNumbers[i];
        output << u << " ";
    }
    output << "\n";
    ALFloatMatrix::print(mRealValues.get(), output);
    for (int i=0; i<mHeight; ++i)
    {
        for (int j=0; j<mWidth; ++j)
        {
            int u = mContents[j+i*mWidth];
            output << u << " ";
        }
        output << "\n";
    }
}
