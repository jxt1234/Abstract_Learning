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
size_t ALStreamReader::readline(ALDynamicBuffer* dBuffer)
{
    ALASSERT(NULL!=dBuffer);
    size_t len = 0;
    bool complete = false;
    bool nobuffer = false;
    dBuffer->reset();
    while (!end() && !complete && !nobuffer)
    {
        auto buffer = mBuffer+mOffset;
        size_t n = mRemain;
        for (size_t i=0; i<n; ++i)
        {
            if ('\n' == buffer[i])
            {
                /*Copy \n as well*/
                auto s = i+1;
                dBuffer->load(buffer, s);
                complete = true;
                len += s;
                mOffset += s;
                mRemain -= s;
                break;
            }
        }
        if (!complete)
        {
            dBuffer->load(buffer, n);
            len+=n;
            mRemain = 0;
        }
        if (mRemain <= 0)
        {
            _refreshCache();
        }
    }
    char last = '\0';
    dBuffer->load(&last, 1);
    //dst[len] = '\0';
    //FUNC_PRINT_ALL(dst, s);
    return len;
}
bool ALStreamReader::end() const
{
    return mRemain <= 0;
}
