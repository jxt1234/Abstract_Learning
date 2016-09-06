#ifndef SRC_PACKAGE_ALCGPPARAMETER_GPTYPE_H
#define SRC_PACKAGE_ALCGPPARAMETER_GPTYPE_H
class ALCGPParameter_GPType:public IStatusType
{
    public:
        ALCGPParameter_GPType():IStatusType("ALCGPParameter"){}
        virtual void* vLoad(GPStream* input) const
        {
            return NULL;
        }
        virtual void vSave(void* contents, GPWStream* output) const
        {
        }
        virtual void vFree(void* contents) const
        {
            ALCGPParameter* c = (ALCGPParameter*)contents;
            c->decRef();
        }
        virtual int vMap(void** content, double* value) const
        {
            const int w = 10;
            const int h = 10;
            int mapnumber=w*h*2;
            if (NULL == value || NULL == content)
            {
                return mapnumber;
            }
            ALCGPParameter* res = (ALCGPParameter*)(*content);
            if (NULL == *content)
            {
                res = new ALCGPParameter(w, h);
                *content = (void*)res;
            }
            ::memcpy(res->pValues, value, mapnumber*sizeof(double));
            return mapnumber;
        }
        virtual bool vCheckCompleted(void* content) const {return NULL!=content;}
        virtual void* vMerge(void* dst, void* src) const {return NULL;}
};
#endif
