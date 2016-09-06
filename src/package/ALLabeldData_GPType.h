#ifndef SRC_PACKAGE_ALLABELDDATA_GPTYPE_H
#define SRC_PACKAGE_ALLABELDDATA_GPTYPE_H
class ALLabeldData_GPType:public IStatusType
{
public:
ALLabeldData_GPType():IStatusType("ALLabeldData"){}
virtual void* vLoad(GPStream* input) const
{
return NULL;
}
virtual void vSave(void* contents, GPWStream* output) const
{
}
virtual void vFree(void* contents) const
{
ALLabeldData* c = (ALLabeldData*)contents;
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
