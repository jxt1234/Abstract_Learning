class ALClassifierSet_GPType:public IStatusType
{
public:
ALClassifierSet_GPType():IStatusType("ALClassifierSet"){}
virtual void* vLoad(GPStream* input) const
{
return NULL;
}
virtual void vSave(void* contents, GPWStream* output) const
{
}
virtual void vFree(void* contents) const
{
ALClassifierSet* c = (ALClassifierSet*)contents;
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
