#include "utils/ALDynamicBuffer.h"
#include "utils/ALDebug.h"
#include <stdlib.h>
#include <string.h>
ALDynamicBuffer::ALDynamicBuffer(size_t originSize)
{
    ALASSERT(originSize>0);
    mSize = 0;
    mContent = (char*)::malloc(originSize*sizeof(char));
    ALASSERT(NULL!=mContent);
    mCapacity = originSize;
}

ALDynamicBuffer::~ALDynamicBuffer()
{
    ::free(mContent);
}
void ALDynamicBuffer::load(const char* src, size_t len)
{
    ALASSERT(NULL!=src);
    ALASSERT(len>0);
    if (mCapacity < len + mSize)
    {
        int enlargeNumber = (len+mSize+mCapacity-1)/mCapacity;
        char* newContent = (char*)malloc(mCapacity*enlargeNumber*sizeof(char));
        mCapacity = mCapacity*enlargeNumber;
        ::memcpy(newContent, mContent, mSize*sizeof(char));
        ::free(mContent);
        mContent = newContent;
    }
    ALASSERT(mCapacity >= len + mSize);
    ::memcpy(mContent+mSize, src, len*sizeof(char));
    mSize+=len;
}
void ALDynamicBuffer::reset()
{
    mSize = 0;
}
