#ifndef SRC_PACKAGE_ALFLOATDATACHAIN_GPTYPE_H
#define SRC_PACKAGE_ALFLOATDATACHAIN_GPTYPE_H
class ALFloatDataChain_GPType:public IStatusType
{
    public:
        ALFloatDataChain_GPType():IStatusType("ALFloatDataChain"){}
        virtual void* vLoad(GPStream* input) const
        {
            ALASSERT(NULL!=input);
            ALSp<ALStream> wrap = ALStreamFactory::wrap(input);
            return (void*)(ALStandardLoader::load(wrap.get()));
        }
        virtual void vSave(void* contents, GPWStream* output) const
        {
        }
        virtual void vFree(void* contents) const
        {
            ALFloatDataChain* c = (ALFloatDataChain*)contents;
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
