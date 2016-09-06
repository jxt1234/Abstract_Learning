#ifndef SRC_PACKAGE_ALSVMPARAMETER_GPTYPE_H
#define SRC_PACKAGE_ALSVMPARAMETER_GPTYPE_H
class ALSVMParameter_GPType:public IStatusType
{
    public:
        ALSVMParameter_GPType():IStatusType("ALSVMParameter"){}
        virtual void* vLoad(GPStream* input) const
        {
            return NULL;
        }
        virtual void vSave(void* contents, GPWStream* output) const
        {
        }
        virtual void vFree(void* contents) const
        {
            ALSVMParameter* c = (ALSVMParameter*)contents;
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
                *content = (void*)(new ALSVMParameter);
            }
            ALSVMParameter* p = (ALSVMParameter*)(*content);
            p->Gamma = value[0]*0.1;
            return mapnumber;
        }
};
#endif
