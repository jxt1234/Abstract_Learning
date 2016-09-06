#ifndef SRC_LEARN_SVM_ONECLASS_H
#define SRC_LEARN_SVM_ONECLASS_H
#include "learn/ALSVM.h"
class OneClass:public ALSVM::Reportor
{
public:
    OneClass(){}
    virtual ~OneClass(){}
    virtual void vHandle(ALFloatMatrix* Y, ALFloatMatrix* kvalue, ALFloatMatrix* coe) const;
    virtual std::map<std::string, std::string> vPrint() const;
private:
};
#endif
