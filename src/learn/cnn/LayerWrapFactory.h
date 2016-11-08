#ifndef LEARN_CNN_LAYERWRAPFACTORY_H
#define LEARN_CNN_LAYERWRAPFACTORY_H
#include "LayerWrap.h"
#include "cJSON/cJSON.h"
namespace ALCNN {
    class LayerWrapFactory
    {
        public:
            static ALSp<LayerWrap> create(const cJSON* description);
    };
};
#endif
