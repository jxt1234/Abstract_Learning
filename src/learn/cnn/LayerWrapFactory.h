#include "LayerWrap.h"
#include "cJSON/cJSON.h"
namespace ALCNN {
    class LayerWrapFactory
    {
        public:
            static ALSp<LayerWrap> create(const cJSON* description);
    };
};
