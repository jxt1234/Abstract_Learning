#ifndef SRC_PACKAGE_ALIUNSUPERLEARNER_GPTYPE_H
#define SRC_PACKAGE_ALIUNSUPERLEARNER_GPTYPE_H
class ALIUnSuperLearner_GPType:public IStatusType
{
public:
ALIUnSuperLearner_GPType():IStatusType("ALIUnSuperLearner"){}
virtual void* vLoad(GPStream* input) const
{
return NULL;
}
virtual void vSave(void* contents, GPWStream* output) const
{
}
virtual void vFree(void* contents) const
{
ALIUnSuperLearner* c = (ALIUnSuperLearner*)contents;
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
