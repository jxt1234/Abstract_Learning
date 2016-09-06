#ifndef INCLUDE_UTILS_ALSP_H
#define INCLUDE_UTILS_ALSP_H
#include <stdlib.h>
#define SAFE_UNREF(x)\
    if (NULL!=(x)) {(x)->decRef();}
#define SAFE_REF(x)\
    if (NULL!=(x)) (x)->addRef();

#define SAFE_ASSIGN(dst, src) \
    {\
        if (src!=NULL)\
        {\
            src->addRef();\
        }\
        if (dst!=NULL)\
        {\
            dst->decRef();\
        }\
        dst = src;\
    }
template <typename T>
class ALSp {
    public:
        ALSp() : mT(NULL) {}
        ALSp(T* obj) : mT(obj) {}
        ALSp(const ALSp& o) : mT(o.mT) { SAFE_REF(mT); }
        ~ALSp() { SAFE_UNREF(mT); }

        ALSp& operator=(const ALSp& rp) {
            SAFE_ASSIGN(mT, rp.mT);
            return *this;
        }
        ALSp& operator=(T* obj) {
            SAFE_UNREF(mT);
            mT = obj;
            return *this;
        }

        T* get() const { return mT; }
        T& operator*() const { return *mT; }
        T* operator->() const { return mT; }

    private:
        T* mT;
};
#endif
