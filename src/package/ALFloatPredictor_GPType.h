#ifndef SRC_PACKAGE_ALFLOATPREDICTOR_GPTYPE_H
#define SRC_PACKAGE_ALFLOATPREDICTOR_GPTYPE_H
class ALFloatPredictor_GPType:public IStatusType
{
public:
ALFloatPredictor_GPType():IStatusType("ALFloatPredictor"){}
virtual void* vLoad(GPStream* input) const
{
return NULL;
}
virtual void vSave(void* contents, GPWStream* output) const
{
}
virtual void vFree(void* contents) const
{
ALFloatPredictor* c = (ALFloatPredictor*)contents;
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
