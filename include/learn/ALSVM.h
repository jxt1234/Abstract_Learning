#ifndef INCLUDE_LEARN_ALSVM_H
#define INCLUDE_LEARN_ALSVM_H
/*This class refer to a model of svm, with support the same model of libsvm*/
#include <string>
#include <istream>
#include <map>
#include <vector>
#include "ALHead.h"
#include "math/ALFloatMatrix.h"
class ALSVM:public ALRefCount
{
public:
    /*Load from file / string*/
    ALSVM(std::istream& model);
    ALSVM();
    void save(std::ostream& output);
    /*Train By Data*/
    static ALSp<ALSVM> train(const ALFloatMatrix* YT, const ALFloatMatrix* X, const std::map<std::string, std::string>& heads);
    /*In this case, Y is Output, X is input*/
    void predict(ALFloatMatrix* Y, const ALFloatMatrix* X) const;
    static ALSp<ALFloatMatrix> loadData(const char* file);//For Test convient
    static std::pair<int, int> measure(const char* file);
    static void loadTrainData(ALSp<ALFloatMatrix> &X, ALSp<ALFloatMatrix> &YT, const char* file);//Fort Test
    virtual ~ALSVM();
    class Kernel:public ALRefCount
    {
    public:
        /*The Shape like Y = X1*X2T*/
        virtual void vCompute(ALFloatMatrix* Y, const ALFloatMatrix* X1, const ALFloatMatrix* X2) const = 0;
        /*The Shape as Y = X1*X1T*/
        virtual void vComputeSST(ALFloatMatrix* Y, const ALFloatMatrix* X) const = 0;
        virtual std::map<std::string, std::string> vPrint() const = 0;
        Kernel(){}
        virtual ~Kernel(){}
    };
    class Reportor:public ALRefCount
    {
    public:
        Reportor(){}
        virtual ~Reportor(){}
        virtual void vHandle(ALFloatMatrix* Y, ALFloatMatrix* kvalue, ALFloatMatrix* coe) const = 0;
        virtual std::map<std::string, std::string> vPrint() const = 0;
    };
    static ALSp<ALSVM> train(const ALFloatMatrix* YT, const ALFloatMatrix* X, ALSp<Kernel> k, ALFLOAT bounder=512.0, size_t iternumber=2);
private:
    void _loadSVS(const std::map<std::string, std::string>& heads, const std::vector<std::string>& alllines);
    ALSp<Kernel> mKernel;
    ALSp<Reportor> mReportor;
    ALSp<ALFloatMatrix> mCoe;
    ALSp<ALFloatMatrix> mSV;
};
#endif
