#include "loader/ALStandardLoader.h"
#include <assert.h>
#include <string>
#include <string.h>
#include <sstream>
#include <iostream>
#include <stdio.h>
#include "utils/ALStreamReader.h"

using namespace std;

ALFloatDataChain* ALStandardLoader::load(const char* file)
{
    ALSp<ALStream> input = ALStreamFactory::readFromFile(file);
    ALFloatDataChain* res = load(input.get());
    ALASSERT(NULL!=res);
    return res;
}

void ALStandardLoader::divide(const ALFloatDataChain* c, ALSp<ALFloatMatrix>& X, ALSp<ALFloatMatrix>& Y, int ypos)
{
    ALASSERT(NULL!=c);
    ALASSERT(c->size()>0);
    ALASSERT(c->width()>0);
    ALASSERT(c->width()>ypos);
    auto h = c->size();
    auto w = c->width();
    X = ALFloatMatrix::create(w-1, h);
    Y = ALFloatMatrix::create(1, h);
    size_t cur = 0;
    for (auto d : c->get())
    {
        auto x = X->vGetAddr(cur);
        auto y = Y->vGetAddr(cur);
        cur++;
        *y = d->value(ypos);
        for (int i=0; i<ypos; ++i)
        {
            x[i] = d->value(i);
        }
        for (int i=ypos+1; i<w; ++i)
        {
            x[i-1] = d->value(i);
        }
    }
}

void ALStandardLoader::load(const char* file, ALSp<ALFloatMatrix>& X, ALSp<ALFloatMatrix>& Y, int ypos)
{
    ALASSERT(NULL!=file);
    ALSp<ALStream> input = ALStreamFactory::readFromFile(file);
    ALSp<ALFloatMatrix> basic = ALFloatMatrix::load(input.get());
    auto w = basic->width();
    auto h = basic->height();
    X = ALFloatMatrix::create(w, h);
    Y = ALFloatMatrix::create(1, h);
    for (size_t i=0; i<h; ++i)
    {
        auto x = X->vGetAddr(i);
        auto y = Y->vGetAddr(i);
        auto b = basic->vGetAddr(i);
        ::memcpy(x, b, ypos*sizeof(ALFLOAT));
        y[0] = b[ypos];
        ::memcpy(x+ypos, b+ypos+1, (w-ypos-1)*sizeof(ALFLOAT));
    }
}

ALFloatDataChain* ALStandardLoader::load(ALStream* input)
{
    ALASSERT(NULL!=input);
    if(input->vIsEnd())
    {
        ALASSERT(0);//FIXME
        return NULL;
    }
    const int buffersize = 4096;
    ALAUTOSTORAGE(buffer, char, buffersize);
    ALSp<ALStreamReader> reader = new ALStreamReader(input);
    auto len = reader->readline(buffer, buffersize-1);
    auto num = measureNumbers(buffer, len);
    ALASSERT(num>0);
    ALFloatData* back = new ALFloatData(num);
    loadNumbers(back->get(), num, buffer, len);
    ALFloatDataChain* res = new ALFloatDataChain(num);
    res->add(back);
    back->decRef();
    while (!reader->end())
    {
        ALFloatData* next = new ALFloatData(num);
        res->add(next);
        next->decRef();
        back->addNext(next);
        len = reader->readline(buffer, buffersize-1);
        loadNumbers(next->get(), num, buffer, len);
        back = next;
    }
    return res;
}



size_t ALStandardLoader::measureNumbers(char* buffer, size_t len)
{
    size_t n = 0;
    float f;
    std::istringstream input(buffer);
    while (input >> f)
    {
        n++;
    }
    return n;
}


char* ALStandardLoader::loadNumbers(ALFLOAT* dst, size_t n, char* buffer, size_t len)
{
    char* pos = buffer;
    for (size_t i=0; i<n; ++i)
    {
        dst[i] = strtod(pos, &pos);
    }
    return pos;
}
