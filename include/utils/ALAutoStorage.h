#ifndef INCLUDE_UTILS_ALAUTOSTORAGE_H
#define INCLUDE_UTILS_ALAUTOSTORAGE_H
/*Auto clean, used for temp use malloc*/
//TODO Use memory pool
#include "ALDebug.h"
template<typename T>
class ALAutoStorage
{
    public:
        ALAutoStorage(size_t size)
        {
            ALASSERT(size>0);
            m = new T[size];
            ALASSERT(NULL!=m);
        }
        ~ALAutoStorage()
        {
            delete [] m;
        }
        T* get() const {return m;}
    private:
        T* m;
};
#define ALAUTOSTORAGE(x, type, size)\
    ALAutoStorage<type> __##x(size);\
    type * x = __##x.get();
    
#endif
