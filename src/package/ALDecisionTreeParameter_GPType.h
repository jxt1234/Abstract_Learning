#ifndef SRC_PACKAGE_ALDECISIONTREEPARAMETER_GPTYPE_H
#define SRC_PACKAGE_ALDECISIONTREEPARAMETER_GPTYPE_H
class ALDecisionTreeParameter_GPType:public IStatusType
{
public:
ALDecisionTreeParameter_GPType():IStatusType("ALDecisionTreeParameter"){}
virtual void* vLoad(GPStream* input) const
{
return NULL;
}
virtual void vSave(void* contents, GPWStream* output) const
{
}
virtual void vFree(void* contents) const
{
ALDecisionTreeParameter* c = (ALDecisionTreeParameter*)contents;
c->decRef();
}
virtual int vMap(void** content, double* value) const
{
int mapnumber=1;
if (NULL == value || NULL == content)
{
return mapnumber;
}
if (NULL == *content)
{
    *content = (void*)(new ALDecisionTreeParameter);
}
ALDecisionTreeParameter* p = (ALDecisionTreeParameter*)(*content);
//p->minGroup = 1 + value[0]*50;
//p->targetPrune = 0.1 + 0.1*value[1];
p->maxDepth = 3 + 10*value[0];
return mapnumber;
}
};
#endif
