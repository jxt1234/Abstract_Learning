#ifndef INCLUDE_UTILS_ALSTREAMREADER_H
#define INCLUDE_UTILS_ALSTREAMREADER_H
#include "ALHead.h"
#include "ALStream.h"
#include "ALDynamicBuffer.h"
#include <string>
class ALStreamReader:public ALRefCount
{
public:
    ALStreamReader(ALStream* stream, size_t buffersize = 4096);
    virtual ~ALStreamReader();
    size_t readline(ALDynamicBuffer* buffer);
    bool end() const;
private:
    void _refreshCache();
    ALSp<ALStream> mStream;
    char* mBuffer;
    const size_t mSize;
    size_t mOffset;
    size_t mRemain;
};

#endif
