#ifndef SRC_PACKAGE_ALISUPERVISELEARNER_GPTYPE_H
#define SRC_PACKAGE_ALISUPERVISELEARNER_GPTYPE_H
class ALISuperviseLearner_GPType:public IStatusType
{
public:
ALISuperviseLearner_GPType():IStatusType("ALISuperviseLearner"){}
virtual void* vLoad(GPStream* input) const
{
return NULL;
}
virtual void vSave(void* contents, GPWStream* output) const
{
}
virtual void vFree(void* contents) const
{
ALISuperviseLearner* c = (ALISuperviseLearner*)contents;
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
