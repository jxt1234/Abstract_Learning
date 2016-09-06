#ifndef INCLUDE_UTILS_ALREFCOUNT_H
#define INCLUDE_UTILS_ALREFCOUNT_H
#include "utils/ALDebug.h"
class ALRefCount
{
    public:
        void addRef() const
        {
            mNum++;
        }
        void decRef() const
        {
            --mNum;
            assert(mNum>=0);
            if (0 >= mNum)
            {
                delete this;
            }
        }
    protected:
        ALRefCount():mNum(1){}
        ALRefCount(const ALRefCount& f):mNum(f.mNum){}
        void operator=(const ALRefCount& f)
        {
            if (this != &f)
            {
                mNum = f.mNum;
            }
        }
        virtual ~ALRefCount(){}
    private:
        inline int count() const{return mNum;}
        mutable int mNum;
};

#define ALSAVEUNREF(x)\
    if (NULL!=(x)) (x)->decRef();

template<class T>
class ALAutoClean
{
    public:
        ALAutoClean(T* t):mT(t){}
        ~ALAutoClean(){mT->decRef();}
        T* operator->() const{return mT;}
    private:
        T* mT;
};


class ALAutoUnRef
{
    public:
        ALAutoUnRef(ALRefCount* t):mT(t){}
        ~ALAutoUnRef(){mT->decRef();}
    private:
        ALRefCount* mT;
};
#define ALAUTOUNREF(x) ALAutoUnRef __##x(x);


#endif
