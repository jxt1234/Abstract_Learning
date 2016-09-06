#ifndef SRC_PACKAGE_ALFLOATMATRIX_GPTYPE_H
#define SRC_PACKAGE_ALFLOATMATRIX_GPTYPE_H
class ALFloatMatrix_GPType:public IStatusType
{
public:
ALFloatMatrix_GPType():IStatusType("ALFloatMatrix"){}
virtual void* vLoad(GPStream* input) const
{
    ALSp<ALStream> wrap = ALStreamFactory::wrap(input);
    return (void*)ALFloatMatrix::load(wrap.get());
}
virtual void vSave(void* contents, GPWStream* output) const
{
    ALSp<ALWStream> s = ALStreamFactory::wrapW(output);
    ALFloatMatrix::save((ALFloatMatrix*)contents, s.get());
}
virtual void vFree(void* contents) const
{
ALFloatMatrix* c = (ALFloatMatrix*)contents;
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
