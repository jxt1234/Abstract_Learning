#ifndef SRC_LEARN_SVM_MULTICLASS_H
#define SRC_LEARN_SVM_MULTICLASS_H
#include "learn/ALSVM.h"
class MultiClass:public ALSVM::Reportor
{
public:
    MultiClass(const std::map<std::string, std::string>& heads);
    virtual ~MultiClass();
    virtual void vHandle(ALFloatMatrix* Y, ALFloatMatrix* kvalue, ALFloatMatrix* coe) const override;
    virtual std::map<std::string, std::string> vPrint() const override;
private:
    int _planes() const{return mNrClass*(mNrClass-1)/2;}
    int _totolNumber() const;
    
    int* mSVNums;
    int* mLabels;
    ALFLOAT* mRbos;
    int mNrClass;
    /*Cache*/
    int* mSVStarts;
};
#endif
