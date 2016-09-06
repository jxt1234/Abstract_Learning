#ifndef SRC_PACKAGE_ALCLASSIFIERCREATOR_GPTYPE_H
#define SRC_PACKAGE_ALCLASSIFIERCREATOR_GPTYPE_H
class ALClassifierCreator_GPType:public IStatusType
{
public:
ALClassifierCreator_GPType():IStatusType("ALClassifierCreator"){}
virtual void* vLoad(GPStream* input) const
{
return NULL;
}
virtual void vSave(void* contents, GPWStream* output) const
{
}
virtual void vFree(void* contents) const
{
ALClassifierCreator* c = (ALClassifierCreator*)contents;
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
virtual bool vCheckCompleted(void* content) const {return NULL!=content;}
virtual void* vMerge(void* dst, void* src) const {return NULL;}
};
#endif
