#ifndef TEST_UPTHREADLABELD_H
#define TEST_UPTHREADLABELD_H
#include "core/ALILabeldMethod.h"
class TrendLabeld:public ALILabeldMethod
{
    public:
        TrendLabeld(ALFLOAT up, ALFLOAT down):mUp(up), mDown(down){}
        virtual ~TrendLabeld(){}
        virtual ALFLOAT vLabel(const ALFloatData* data, bool &success) const
        {
            ALASSERT(NULL!=data);
            if (NULL == data->front())
            {
                success = false;
                return 0.0;
            }
            success = true;
            auto front = data->front();
            if (front->value(0) <= data->value(0))
            {
                return mDown;
            }
            return mUp;
        }
        ALFLOAT mUp;
        ALFLOAT mDown;
};
#endif
