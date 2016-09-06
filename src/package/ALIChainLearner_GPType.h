#ifndef SRC_PACKAGE_ALICHAINLEARNER_GPTYPE_H
#define SRC_PACKAGE_ALICHAINLEARNER_GPTYPE_H
class ALIChainLearner_GPType:public IStatusType
{
public:
ALIChainLearner_GPType():IStatusType("ALIChainLearner"){}
virtual void* vLoad(GPStream* input) const
{
return NULL;
}
virtual void vSave(void* contents, GPWStream* output) const
{
}
virtual void vFree(void* contents) const
{
ALIChainLearner* c = (ALIChainLearner*)contents;
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
