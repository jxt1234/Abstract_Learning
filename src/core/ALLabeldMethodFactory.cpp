#include "core/ALLabeldMethodFactory.h"
class ALBasicLabeldMethod:public ALILabeldMethod
{
    public:
        virtual ALFLOAT vLabel(const ALFloatData* data, bool &success) const
        {
            success = true;
            return data->value(0);
        }
};

ALILabeldMethod* ALLabeldMethodFactory::createBasic()
{
    ALILabeldMethod* p = new ALBasicLabeldMethod;
    return p;
}
ALLabeldData* ALLabeldMethodFactory::delayLabel(const std::vector<const ALFloatData*>& data, const ALILabeldMethod* basic, int delay)
{
    auto res = new ALLabeldData;
    for (auto d : data)
    {
        auto origin = d;
        for (int i=0; i<delay; ++i)
        {
            d = d->front();
            if (NULL == d)
            {
                break;
            }
        }
        if (NULL == d)
        {
            continue;
        }
        bool success = true;
        ALFLOAT value = basic->vLabel(origin, success);
        if (!success)
        {
            continue;
        }
        res->insert(value, d);
    }
    return res;
}
