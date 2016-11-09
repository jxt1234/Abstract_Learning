#ifndef PACKAGE_ALIMATRIXTRANSFORMER_GPTYPE_H
#define PACKAGE_ALIMATRIXTRANSFORMER_GPTYPE_H
#include <sstream>
class ALIMatrixTransformer_GPType:public IStatusType
{
    public:
        ALIMatrixTransformer_GPType():IStatusType("ALIMatrixTransformer"){}
        virtual void* vLoad(GPStream* input) const
        {
            //FIXME
            return NULL;
        }
        virtual void vSave(void* contents, GPWStream* output) const
        {
            //FIXME
            ALIMatrixTransformer* c = (ALIMatrixTransformer*)contents;
            std::ostringstream output_s;
            c->vPrint(output_s);
            auto str_S = output_s.str();
            output->vWrite(str_S.c_str(), str_S.size());
        }
        virtual void vFree(void* contents) const
        {
            ALIMatrixTransformer* c = (ALIMatrixTransformer*)contents;
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
