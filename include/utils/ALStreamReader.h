#ifndef INCLUDE_UTILS_ALSTREAMREADER_H
#define INCLUDE_UTILS_ALSTREAMREADER_H
#include "ALHead.h"
#include "ALStream.h"
#include <string>
class ALStreamReader:public ALRefCount
{
public:
    ALStreamReader(ALStream* stream, size_t buffersize = 4096);
    virtual ~ALStreamReader();
    /*dst must has the lenght larger that maxSize+1, 1 for \0*/
    size_t readline(char* dst, size_t maxSize);
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
