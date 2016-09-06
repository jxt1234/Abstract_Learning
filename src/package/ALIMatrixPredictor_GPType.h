#ifndef SRC_PACKAGE_ALIMATRIXPREDICTOR_GPTYPE_H
#define SRC_PACKAGE_ALIMATRIXPREDICTOR_GPTYPE_H
class ALIMatrixPredictor_GPType:public IStatusType
{
public:
ALIMatrixPredictor_GPType():IStatusType("ALIMatrixPredictor"){}
virtual void* vLoad(GPStream* input) const
{
return NULL;
}
virtual void vSave(void* contents, GPWStream* output) const
{
}
virtual void vFree(void* contents) const
{
ALIMatrixPredictor* c = (ALIMatrixPredictor*)contents;
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
