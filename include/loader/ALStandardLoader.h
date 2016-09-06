#ifndef LOADER_ALSTANDARDLOADER_H
#define LOADER_ALSTANDARDLOADER_H

/*This class assume the file is neat as src/test/bao.txt and the memory is enough to load all data in file*/
#include "core/ALFloatDataChain.h"
#include "math/ALFloatMatrix.h"
#include "utils/ALStream.h"
#include <fstream>

class ALStandardLoader
{
public:
    static ALFloatDataChain* load(const char* file);
    static ALFloatDataChain* load(ALStream* input);
    static void load(const char* file, ALSp<ALFloatMatrix>& X, ALSp<ALFloatMatrix>& Y, int y);
    static void divide(const ALFloatDataChain* c, ALSp<ALFloatMatrix>& X, ALSp<ALFloatMatrix>& Y, int y);
    static int getLineNumber(const char* file);
    static size_t measureNumbers(char* buffer, size_t len);
    static char* loadNumbers(ALFLOAT* dst, size_t n, char* buffer, size_t len);
};


#endif
