#include "utils/ALStreamReader.h"

ALStreamReader::ALStreamReader(ALStream* stream, size_t buffersize):mSize(buffersize)
{
    ALASSERT(NULL!=stream);
    ALASSERT(buffersize>=512);
    mBuffer = new char[buffersize];
    ALASSERT(NULL!=mBuffer);
    stream->addRef();
    mStream = stream;
    _refreshCache();
}


ALStreamReader::~ALStreamReader()
{
    delete [] mBuffer;
}

void ALStreamReader::_refreshCache()
{
    if (mStream->vIsEnd())
    {
        mRemain = 0;
        mOffset = 0;
        return;
    }
    mRemain = mStream->vRead(mBuffer, mSize*sizeof(char));
    mOffset = 0;
}
size_t ALStreamReader::readline(char* dst, size_t maxSize)
{
    ALASSERT(NULL!=dst);
    ALASSERT(0<maxSize);
    size_t len = 0;
    bool complete = false;
    bool nobuffer = false;
    while (!end() && !complete && !nobuffer)
    {
        auto buffer = mBuffer+mOffset;
        auto dst_current = dst + len;
        size_t n = mRemain;
        if (maxSize < len+mRemain)
        {
            n = maxSize-len;
            nobuffer = true;
        }
        for (size_t i=0; i<n; ++i)
        {
            if ('\n' == buffer[i])
            {
                auto s = i+1;
                /*Copy \n as well*/
                ::memcpy(dst_current, buffer, s*sizeof(char));
                complete = true;
                len += s;
                mOffset += s;
                mRemain -= s;
                break;
            }
        }
        if (!complete)
        {
            ::memcpy(dst_current, buffer, n*sizeof(char));
            len+=n;
            mRemain = 0;
        }
        if (mRemain <= 0)
        {
            _refreshCache();
        }
    }
    dst[len] = '\0';
    //FUNC_PRINT_ALL(dst, s);
    return len;
}
bool ALStreamReader::end() const
{
    return mRemain <= 0;
}
