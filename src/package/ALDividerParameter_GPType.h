#ifndef SRC_PACKAGE_ALDIVIDERPARAMETER_GPTYPE_H
#define SRC_PACKAGE_ALDIVIDERPARAMETER_GPTYPE_H
class ALDividerParameter_GPType:public IStatusType
{
    public:
        ALDividerParameter_GPType():IStatusType("ALDividerParameter"){}
        virtual void* vLoad(GPStream* input) const
        {
            return NULL;
        }
        virtual void vSave(void* contents, GPWStream* output) const
        {
        }
        virtual void vFree(void* contents) const
        {
            ALDividerParameter* c = (ALDividerParameter*)contents;
            c->decRef();
        }
        virtual int vMap(void** content, double* value) const
        {
            int mapnumber=2;
            if (NULL == value || NULL == content)
            {
                return mapnumber;
            }
            if (NULL == *content)
            {
                *content = (void*)(new ALDividerParameter);
            }
            ALDividerParameter* p = (ALDividerParameter*)(*content);
            p->per = 0.05 + 0.90*value[0];
            p->step = 10*value[1];
            return mapnumber;
        }
};
#endif
