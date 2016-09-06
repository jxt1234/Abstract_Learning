//
//  ALStream.cpp
//  abs
//
//  Created by jiangxiaotang on 15/7/21.
//  Copyright (c) 2015年 jiangxiaotang. All rights reserved.
//

#include "utils/ALStream.h"
#include "ALFileStream.h"
#include "lowlevelAPI/GPStream.h"

ALStream*  ALStreamFactory::readFromFile(const char* file)
{
    return new ALFileStream(file);
}
ALWStream* ALStreamFactory::writeForFile(const char* file)
{
    return new ALWFileStream(file);
}

class ALStreamWrap:public ALStream
{
public:
    ALStreamWrap(GPStream* stream)
    {
        ALASSERT(NULL!=stream);
        mMetaData = stream;
    }
    virtual ~ALStreamWrap()
    {
    }
    virtual size_t vRead(void* buffer, size_t size)
    {
        return mMetaData->vRead(buffer, size);
    }
    /*Return true if the stream has moved to end*/
    virtual bool vIsEnd() const
    {
        return mMetaData->vIsEnd();
    }
private:
    GPStream* mMetaData;
};
class ALWStreamWrap:public ALWStream
{
public:
    ALWStreamWrap(GPWStream* meta)
    {
        ALASSERT(NULL!=meta);
        mMetaData = meta;
    }
    virtual ~ALWStreamWrap()
    {
    }
    virtual size_t vWrite(const void* buffer, size_t size)
    {
        return mMetaData->vWrite(buffer, size);
    }
    virtual bool vFlush()
    {
        return mMetaData->vFlush();
    }
private:
    GPWStream* mMetaData;
};



ALStream* ALStreamFactory::wrap(GPStream* m)
{
    return new ALStreamWrap(m);
}

ALWStream* ALStreamFactory::wrapW(GPWStream* m)
{
    return new ALWStreamWrap(m);
}

