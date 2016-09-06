#ifndef INCLUDE_UTILS_ALSTREAM_H
#define INCLUDE_UTILS_ALSTREAM_H
//
//  ALStream.h
//  abs
//
//  Created by jiangxiaotang on 15/7/21.
//  Copyright (c) 2015年 jiangxiaotang. All rights reserved.
//
#include "ALRefCount.h"

class ALStream : public ALRefCount
{
public:
    /*If the buffer is NULL, it means skip size bytes, return the fact bytes it read*/
    virtual size_t vRead(void* buffer, size_t size) = 0;
    
    /*Return true if the stream has moved to end*/
    virtual bool vIsEnd() const = 0;
    
    template <typename T>
    T read()
    {
        T buffer;
        this->vRead(&buffer, sizeof(T));
        return buffer;
    }
protected:
    ALStream() = default;
    virtual ~ALStream() = default;
private:
    ALStream(const ALStream& stream) = default;
    ALStream& operator=(const ALStream& stream) = default;
};
class ALWStream : public ALRefCount
{
public:
    virtual size_t vWrite(const void* buffer, size_t size) = 0;
    virtual bool vFlush() = 0;
    
    template<typename T>
    bool write(T v)
    {
        return this->vWrite(&v, sizeof(T));
    }
protected:
    ALWStream() = default;
    virtual ~ALWStream() = default;
};

class GPStream;
class GPWStream;
class ALStreamFactory
{
public:
    static ALStream* readFromMem(const void* buffer);
    static ALWStream* writeForMem(void* buffer, int size);
    static ALStream* readFromFile(const char* file);
    static ALWStream* writeForFile(const char* file);
    
    static ALStream* wrap(GPStream* m);
    static ALWStream* wrapW(GPWStream* m);

};
#endif
