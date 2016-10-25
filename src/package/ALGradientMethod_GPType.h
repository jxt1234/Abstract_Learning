#include "cJSON/cJSON.h"
#include "learn/ALCNNLearner.h"
class ALGradientMethod_GPType:public IStatusType
{
public:
ALGradientMethod_GPType():IStatusType("ALGradientMethod"){}
virtual void* vLoad(GPStream* input) const
{
    //FIXME
    size_t maxSize = 32768;
    char block[maxSize];
    auto realSize = input->vRead(block, maxSize);
    ALASSERT(realSize < maxSize);
    block[realSize] = '\0';
    auto jsonObject = cJSON_Parse(block);
    ALSp<ALCNNLearner> learner = new ALCNNLearner(jsonObject);
    auto gd = learner->getGDMethod();
    gd->other = jsonObject;
    return gd;
}
virtual void vSave(void* contents, GPWStream* output) const
{
}
virtual void vFree(void* contents) const
{
    ALGradientMethod* c = (ALGradientMethod*)contents;
    cJSON_Delete((cJSON*)(c->other));
    delete c;
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
virtual bool vCheckCompleted(void* content) const {return NULL!=content;}
virtual void* vMerge(void* dst, void* src) const {return NULL;}
};
