#ifndef CORE_ALFLOATDATA_H
#define CORE_ALFLOATDATA_H
#include <vector>
#include <iostream>
#include "ALHead.h"
#include "utils/ALRefCount.h"

class ALFloatData:public ALRefCount
{
public:
    ALFloatData(size_t num);
    virtual ~ALFloatData();
    void load(std::istream& input);
    inline size_t num() const{return mNum;}
    ALFLOAT value(int n) const;
    void copy(void* dst) const;
    inline ALFLOAT* get() const {return mData;}
    ALFloatData* front() const {return mFront;}
    void addNext(ALFloatData* next);
    bool canBack(int l) const;
private:
    ALFLOAT* mData;
    size_t mNum;
    ALFloatData* mFront;
};

class ALFloatPredictor:public ALRefCount
{
public:
    ALFloatPredictor(){}
    virtual ~ALFloatPredictor(){}
    
    virtual ALFLOAT vPredict(const ALFloatData* data) const= 0;
    virtual void vPrint(std::ostream& out) const {}
};

class ALDummyFloatPredictor:public ALFloatPredictor
{
public:
    ALDummyFloatPredictor(){}
    virtual ~ALDummyFloatPredictor(){}
    
    virtual ALFLOAT vPredict(const ALFloatData* data) const{ return 0;}
};

class ALLabeldData:public ALRefCount
{
public:
    ALLabeldData(){}
    virtual ~ALLabeldData(){}
    inline void insert(ALFLOAT v, const ALFloatData* d)
    {
        mData.push_back(std::make_pair(v, d));
        d->addRef();
    }
    inline void clear()
    {
        for (auto p : mData)
        {
            (p.second)->decRef();
        }
        mData.clear();
    }
    inline size_t size() const {return mData.size();}
    inline const std::vector<std::pair<ALFLOAT, const ALFloatData*> >& get() const {return mData;}
    void collect(std::vector<const ALFloatData*>& result) const;
private:
    std::vector<std::pair<ALFLOAT, const ALFloatData*> > mData;
};

#endif
