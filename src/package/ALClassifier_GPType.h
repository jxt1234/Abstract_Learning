#ifndef SRC_PACKAGE_ALCLASSIFIER_GPTYPE_H
#define SRC_PACKAGE_ALCLASSIFIER_GPTYPE_H
#include <sstream>
class ALClassifier_GPType:public IStatusType
{
public:
    ALClassifier_GPType():IStatusType("ALClassifier"){}
    virtual void* vLoad(GPStream* input) const
    {
        return NULL;
    }
    virtual void vSave(void* contents, GPWStream* output) const
    {
        ALClassifier* c = (ALClassifier*)contents;
        std::ostringstream os;
        c->vPrint(os);
        output->vWrite(os.str().c_str(), os.str().size());
    }
    virtual void vFree(void* contents) const
    {
        ALClassifier* c = (ALClassifier*)contents;
        c->decRef();
    }
    virtual int vMap(void** content, double* value) const
    {
        int mapnumber=0;
        if (NULL == value || NULL == content)
        {
            return mapnumber;
        }
        if (NULL == *content)
        {
        }
        return mapnumber;
    }
};
#endif
