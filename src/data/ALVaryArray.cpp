#include "data/ALVaryArray.h"
#include "utils/ALStreamReader.h"
#include <string.h>
static ALINT* loadNumbers(char* buffer, size_t& n)
{
    char* pos = buffer;
    n = strtol(pos, &pos, 10);
    auto dst = new ALINT[n];
    for (size_t i=0; i<n; ++i)
    {
        dst[i] = strtol(pos, &pos, 10);
    }
    return dst;
}
void ALVaryArray::addArray(ALINT* v, size_t length)
{
    Array a;
    a.c = v;
    a.length = length;
    mArray.push_back(a);
    mContent.push_back(v);
}


const ALVaryArray::Array& ALVaryArray::getArray(size_t index) const
{
    ALASSERT(index<mArray.size());
    return mArray[index];
}

ALVaryArray* ALVaryArray::create(ALStream* input)
{
    ALASSERT(NULL!=input);
    ALVaryArray* result = new ALVaryArray;
    ALSp<ALStreamReader> reader = new ALStreamReader(input);
    ALASSERT(!reader->end());
    ALDynamicBuffer dyBuffer(4096);//TODO
    size_t n;
    while (!reader->end())
    {
        size_t len = reader->readline(&dyBuffer);
        auto line = loadNumbers(dyBuffer.content(), n);
        result->addArray(line, n);
    }
    return result;
}

ALVaryArray::ALVaryArray()
{
}

ALVaryArray::~ALVaryArray()
{
    for (auto c : mContent)
    {
        delete [] c;
    }
}
