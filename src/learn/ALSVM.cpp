#include <string>
#include <sstream>
#include <fstream>
#include "learn/ALSVM.h"
#include "learn/ALSVMKernelFactory.h"
#include "learn/ALSVMHandleFactory.h"
#include "svm/SMO.h"
#include <iostream>
#include "math/ALFloatMatrix.h"

using namespace std;
using namespace ALSMO;
typedef map<string, string>::const_iterator ITER;

ALSVM::ALSVM()
{
}
ALSVM::ALSVM(std::istream& input)
{
    string line;
    /*Read Head*/
    map<string, string> heads;
    for (;getline(input, line);)
    {
        istringstream is(line);
        string name;
        is >> name;
        if (name == "SV")
        {
            break;
        }
        ostringstream remain;
        remain << is.rdbuf();
        heads.insert(make_pair(name, remain.str()));
    }
    /*Create kernel and report, ALSVM has nothing to do*/
    mKernel = ALSVMKernelFactory::create(heads);
    mReportor = ALSVMHandleFactory::create(heads);
    vector<string> alllines;
    while (getline(input, line))
    {
        alllines.push_back(line);
    }
    _loadSVS(heads, alllines);
}

ALSVM::~ALSVM()
{
}
ALSp<ALSVM> ALSVM::train(const ALFloatMatrix* YT, const ALFloatMatrix* X, const std::map<std::string, std::string>& heads)
{
    ALSp<Kernel> k = ALSVMKernelFactory::create(heads);
    ALASSERT(NULL!=k.get());//FIXME
    ALFLOAT bounder=512;
    int iternumber=2;
    {
        auto iter = heads.find("Bound");
        if (iter!=heads.end())
        {
            std::istringstream is(iter->second);
            is >> bounder;
        }
        iter = heads.find("iternumber");
        if (iter!=heads.end())
        {
            std::istringstream is(iter->second);
            is >> iternumber;
        }
    }
    return train(YT, X, k, bounder, iternumber);
}
ALSp<ALSVM> ALSVM::train(const ALFloatMatrix* YT, const ALFloatMatrix* X, ALSp<Kernel> k, ALFLOAT bounder, size_t iternumber)
{
    ALAUTOTIME;
    ALASSERT(NULL!=X);
    ALASSERT(NULL!=YT);
    ALASSERT(X->height()==YT->width());
    auto l = X->height();
    ALSp<ALSVM> result = new ALSVM;
    result->mKernel = k;
    ALSp<ALFloatMatrix> KX = ALFloatMatrix::create(l, l);
    k->vComputeSST(KX.get(), X);
    ALSp<ALFloatMatrix> coe = ALFloatMatrix::create(l, 1);
    ALSp<ALFloatMatrix> CT = ALFloatMatrix::create(l, 1);
    ALFLOAT* c = CT->vGetAddr();
    for (int i=0; i<l; ++i)
    {
        c[i] = bounder;
    }
    ALFLOAT b;
    SMO solver(l);
    solver.sovle(coe.get(), b, CT.get(), KX.get(), YT, iternumber);
    b = -b;
    vector<int> valid;
    valid.reserve(l);
    ALFLOAT* alpha = coe->vGetAddr();
    ALFLOAT* y = YT->vGetAddr();
    int nsv1=0, nsv2=0;
    for (int i=0; i<l; ++i)
    {
        ALFLOAT _a = alpha[i];
        if (_a > 0.001 || _a < -0.001)//FIXME
        {
            valid.push_back(i);
            if (y[i]>0)
            {
                nsv1++;
            }
            else
            {
                nsv2++;
            }
        }
    }
    if (valid.empty())
    {
        return NULL;
    }
    ALSp<ALFloatMatrix> sv = ALFloatMatrix::create(X->width(), valid.size());
    ALSp<ALFloatMatrix> newcoe = ALFloatMatrix::create(valid.size(), 1);
    for (int i=0; i<valid.size(); ++i)
    {
        ALFLOAT* dst = sv->vGetAddr(i);
        ALFLOAT* src = X->vGetAddr(valid[i]);
        ::memcpy(dst, src, sizeof(ALFLOAT)*X->width());

        newcoe->vGetAddr(0)[i] = (coe->vGetAddr(0)[valid[i]] * y[valid[i]]);
    }

    result->mCoe = newcoe;
    result->mSV = sv;

    map<string, string> reportorheads;
    reportorheads.insert(make_pair("svm_type", "c_svc"));
    reportorheads.insert(make_pair("nr_class", "2"));
    reportorheads.insert(make_pair("label", "1 -1"));
    ostringstream rho;rho<<b;
    reportorheads.insert(make_pair("rho", rho.str()));
    ostringstream nr_sv;
    nr_sv << nsv1 << " "<<nsv2;
    reportorheads.insert(make_pair("nr_sv", nr_sv.str()));

    result->mReportor = ALSVMHandleFactory::create(reportorheads);
    return result;
}


void ALSVM::predict(ALFloatMatrix* Y, const ALFloatMatrix* X) const
{
    ALASSERT(NULL!=Y);
    ALASSERT(NULL!=X);
    ALASSERT(Y->height() == X->height());
    ALASSERT(NULL!=mReportor.get());
    ALASSERT(NULL!=mKernel.get());
    ALSp<ALFloatMatrix> kvalue = ALFloatMatrix::create(mSV->height(), X->height());
    mKernel->vCompute(kvalue.get(), X, mSV.get());
    mReportor->vHandle(Y, kvalue.get(), mCoe.get());
}
template <typename T>
static T getValue(const map<string, string>& heads, const char* name)
{
    T res;
    ITER iter = heads.find(name);
    ALASSERT(iter!=heads.end());
    {
        istringstream is(iter->second);
        is >> res;
    }
    return res;
}

static int findMaxPos(const string& line)
{
    auto index = line.find(":");
    auto posindex = index;
    posindex = 0;
    while (index != string::npos)
    {
        posindex = index;
        index = line.find(":", posindex+1);
    }
    auto fin = posindex-1;
    int result = 0;
    int step = 1;
    for (;line[fin]!=' ';fin--)
    {
        result += (line[fin]-'0')*step;
        step*=10;
    }
    return result;
}

static void writeVector(const string& line, ALFLOAT* v)
{
    vector<string> words;
    string w;
    istringstream _line(line);
    while (_line >> w)
    {
        auto index = w.find(":");
        if (index!=string::npos)
        {
            w[index] = ' ';
            words.push_back(w);
        }
    }
    for (int i=0; i<words.size(); ++i)
    {
        int index;
        double value;
        istringstream is(words[i]);
        is >> index >> value;
        v[index-1] = value;//The libsvm's data format is begining from 1
    }
}

pair<int, int> ALSVM::measure(const char* file)
{
    ifstream inp(file);
    ALASSERT(inp.good());
    string line;
    while(getline(inp, line))
    {
        if (line.find(":")!=string::npos)
        {
            break;
        }
    }
    int maxlength = findMaxPos(line);
    int h = 1;
    while(getline(inp, line))
    {
        int l = findMaxPos(line);
        if (maxlength < l)
        {
            maxlength = l;
        }
        h++;
    }
    return make_pair(maxlength, h);
}

void ALSVM::loadTrainData(ALSp<ALFloatMatrix> &X, ALSp<ALFloatMatrix> &YT, const char* file)
{
    ALASSERT(NULL!=file);
    pair<int, int> size = measure(file);
    int w = size.first;
    int h = size.second;
    X = ALFloatMatrix::create(w, h);
    YT = ALFloatMatrix::create(h, 1);
    ifstream inp(file);
    int i = 0;
    ALFLOAT* y = YT->vGetAddr();
    string line;
    while(getline(inp, line))
    {
        ALFLOAT* l = X->vGetAddr(i);
        ::memset(l, 0, sizeof(ALFLOAT)*w);
        writeVector(line, l);
        istringstream is(line);
        is >> y[i];
        ++i;
    }
}

ALSp<ALFloatMatrix> ALSVM::loadData(const char* file)
{
    ifstream inp(file);
    if(!inp.good()) return NULL;
    string line;
    getline(inp, line);
    int w = findMaxPos(line);
    int n = 1;
    while(getline(inp, line))
    {
        int maxw = findMaxPos(line);
        n++;
        if (maxw > w)
        {
            w = maxw;
        }
    }
    inp.close();

    ALSp<ALFloatMatrix> result = ALFloatMatrix::create(w, n);
    ifstream datainp(file);
    n = 0;
    while (getline(datainp, line))
    {
        ALFLOAT* l = result->vGetAddr(n++);
        ::memset(l, 0, sizeof(ALFLOAT)*w);
        writeVector(line, l);
    }
    return result;
}

void ALSVM::_loadSVS(const std::map<std::string, std::string>& heads, const std::vector<std::string>& alllines)
{
    int nr_class = getValue<int>(heads, "nr_class");
    int numberSV = getValue<int>(heads, "total_sv");
    ALASSERT(alllines.size()==numberSV);
    mCoe = ALFloatMatrix::create(nr_class-1, numberSV);
    int w = 0;
    for (int i=0; i<alllines.size(); ++i)
    {
        int _w = findMaxPos(alllines[i]);
        if (_w > w)
        {
            w = _w;
        }
    }
    /*Count the total numbers of a Vector*/
    mSV = ALFloatMatrix::create(w, numberSV);
    /*Get all support vector*/
    {
        ALAUTOTIME;
        for (int k=0; k<alllines.size(); ++k)
        {
            string oneline = alllines[k];
            for (int i=0; i<oneline.size();++i)
            {
                if (oneline[i] == ':')
                {
                    oneline[i] = ' ';
                }
            }
            istringstream is(oneline);
            ALFLOAT* c = mCoe->vGetAddr(k);
            for (int i=0; i<nr_class-1; ++i)
            {
                is >> c[i];
            }
            ALFLOAT* v = mSV->vGetAddr(k);
            ::memset(v, 0, sizeof(ALFLOAT)*w);
            int pos;ALFLOAT value;
            while (is >> pos >> value)
            {
                v[pos-1] = value;
            }
        }
    }

    /*Transpose mCoe*/
    {
        ALAUTOTIME;
        ALSp<ALFloatMatrix> newcoe = ALFloatMatrix::transpose(mCoe.get());
        mCoe = newcoe;
    }
}

void ALSVM::save(std::ostream& output)
{
    /*TODO Support other types*/
    output << "svm_type c_svc"<<endl;
    auto klist = mKernel->vPrint();
    for (auto p : klist)
    {
        output << p.first << " "<<p.second<<endl;
    }
    auto rlist = mReportor->vPrint();
    for (auto p : rlist)
    {
        output << p.first << " "<<p.second<<endl;
    }
    output << "SV"<<endl;
    ALASSERT(mCoe->width() == mSV->height());
    auto l = mCoe->width();
    for (int i=0; i<l; ++i)
    {
        for (int j=0; j<mCoe->height(); ++j)
        {
            auto c = mCoe->vGetAddr(j)[i];
            output << c << " ";
        }
        auto v = mSV->vGetAddr(i);
        for (int j=0; j<mSV->width();++j)
        {
            output<<j+1<<":"<<v[j]<<" ";
        }
        output << endl;
    }
}
