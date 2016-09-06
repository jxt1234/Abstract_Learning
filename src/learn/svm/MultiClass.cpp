#include "MultiClass.h"
#include <sstream>
using namespace std;
typedef map<string, string>::const_iterator ITER;

int MultiClass::_totolNumber() const
{
    int sum = 0;
    for (int i=0; i<mNrClass; ++i)
    {
        sum+=mSVNums[i];
    }
    return sum;
}
MultiClass::MultiClass(const std::map<std::string, std::string>& heads)
{
    ITER iter = heads.find("nr_class");
    ALASSERT(iter!=heads.end());
    {
        istringstream is(iter->second);
        is >> mNrClass;
    }
    ALASSERT(mNrClass > 1);
    mSVNums = new int[mNrClass];
    mSVStarts = new int[mNrClass];
    mLabels = new int[mNrClass];
    mRbos = new ALFLOAT[mNrClass*(mNrClass-1)/2];
    iter = heads.find("nr_sv");
    ALASSERT(iter!=heads.end());
    {
        istringstream is(iter->second);
        for (int i=0; i<mNrClass; ++i)
        {
            is >> mSVNums[i];
        }
        mSVStarts[0] = 0;
        for (int i=1; i<mNrClass; ++i)
        {
            mSVStarts[i] = mSVStarts[i-1] + mSVNums[i-1];
        }
    }
    iter = heads.find("label");
    ALASSERT(iter!=heads.end());
    {
        istringstream is(iter->second);
        for (int i=0; i<mNrClass; ++i)
        {
            is >> mLabels[i];
        }
    }
    iter = heads.find("rho");
    ALASSERT(iter!=heads.end());
    {
        istringstream is(iter->second);
        for (int i=0; i<_planes(); ++i)
        {
            is >> mRbos[i];
        }
    }
}
MultiClass::~MultiClass()
{
    delete [] mSVNums;
    delete [] mLabels;
    delete [] mSVStarts;
    delete [] mRbos;
}
void MultiClass::vHandle(ALFloatMatrix* Y, ALFloatMatrix* kvalue, ALFloatMatrix* coe) const
{
    ALASSERT(NULL!=Y && NULL!=coe && NULL!=kvalue);
    ALASSERT(1==Y->width());
    for (int i=0; i<Y->height(); ++i)
    {
        int plane = 0;
        ALFLOAT* kv = (ALFLOAT*)(kvalue->vGetAddr(i));
        vector<int> vote(mNrClass, 0);
        for (int j=0; j<mNrClass; ++j)
        {
            for (int k=j+1; k<mNrClass; ++k)
            {
                ALFLOAT sum = 0.0f;
                int si = mSVStarts[j];
                int sj = mSVStarts[k];
                int ci = mSVNums[j];
                int cj = mSVNums[k];
                ALFLOAT* coef1 = (ALFLOAT*)(coe->vGetAddr(k-1));
                ALFLOAT* coef2 = (ALFLOAT*)(coe->vGetAddr(j));
                for(int _k=0;_k<ci;_k++)
                    sum += coef1[si+_k] * kv[si+_k];
                for(int _k=0;_k<cj;_k++)
                    sum += coef2[sj+_k] * kv[sj+_k];
                sum -= mRbos[plane++];
                if (sum > 0)
                {
                    vote[j]++;
                }
                else
                {
                    vote[k]++;
                }
            }
        }
        int maxVoteNum = 0;
        for (int _i=1; _i<vote.size(); ++_i)
        {
            if (vote[_i] > vote[maxVoteNum])
            {
                maxVoteNum = _i;
            }
        }
        ALFLOAT* y = (ALFLOAT*)(Y->vGetAddr(i));
        *y = mLabels[maxVoteNum];
    }
}

std::map<std::string, std::string> MultiClass::vPrint() const
{
    std::map<std::string, std::string> res;
    ostringstream os;
    os << mNrClass;
    res.insert(make_pair("nr_class", os.str()));
    os.str("");
    for (int i=0; i<mNrClass; ++i)
    {
        os << mLabels[i]<< " ";
    }
    res.insert(make_pair("label", os.str()));
    
    os.str("");
    for (int i=0; i<mNrClass; ++i)
    {
        os << mSVNums[i] << " ";
    }
    res.insert(make_pair("nr_sv", os.str()));
    os.str("");
    for (int i=0; i<mNrClass*(mNrClass-1)/2; ++i)
    {
        os << mRbos[i] << " ";
    }
    res.insert(make_pair("rho", os.str()));
    os.str("");
    os << _totolNumber();
    res.insert(make_pair("total_sv", os.str()));
    return res;
}
