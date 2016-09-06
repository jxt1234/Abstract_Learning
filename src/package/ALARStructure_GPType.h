#ifndef SRC_PACKAGE_ALARSTRUCTURE_GPTYPE_H
#define SRC_PACKAGE_ALARSTRUCTURE_GPTYPE_H
class ALARStructure_GPType:public IStatusType
{
    public:
        ALARStructure_GPType():IStatusType("ALARStructure"){}
        virtual void* vLoad(GPStream* input) const
        {
            return NULL;
        }
        virtual void vSave(void* contents, GPWStream* output) const
        {
        }
        virtual void vFree(void* contents) const
        {
            ALARStructure* c = (ALARStructure*)contents;
            c->decRef();
        }
        virtual int vMap(void** content, double* value) const
        {
            int mapnumber=3;
            if (NULL == value || NULL == content)
            {
                return mapnumber;
            }
            if (NULL == *content)
            {
                *content = new ALARStructure;
            }
            ALARStructure* p = (ALARStructure*)(*content);
            p->l = 1 + 10*value[0];
            p->w = 1 + 10*value[1];
            p->d = 0;
            p->c = value[2] > 0.5 ? 1:0;
            return mapnumber;
        }
};
#endif
