#include "LayerWrapFactory.h"
#include "LayerFactoryRegistor.hpp"
#include "MeanPoolLayer.h"
#include "CNNLayer.h"
#include "SoftMaxLayer.h"
#include "ReluLayer.hpp"
#include "MaxPoolLayer.hpp"
#include "InnerProductLayer.hpp"
#include <fstream>
namespace ALCNN {
    static void _readLayerParameters(const cJSON* layer, LayerParameters& p, std::string& type)
    {
        ALASSERT(NULL!=layer);
        p.mIntValues.clear();
        for (auto c = layer->child; NULL!=c; c=c->next)
        {
            auto name = c->string;
            if (!strcmp(name, "type"))
            {
                type = c->valuestring;
                continue;
            }
            if (!strcmp(name, "input"))
            {
                p.uInputSize = c->valueint;
                continue;
            }
            if (!strcmp(name, "output"))
            {
                p.uOutputSize = c->valueint;
                continue;
            }
            if (!strcmp(name, "input_3D"))
            {
                auto ac = c->child;
                ALASSERT(NULL!=ac);
                p.mMatrixInfo.iWidth = ac->valueint;
                ac = ac->next;
                ALASSERT(NULL!=ac);
                p.mMatrixInfo.iHeight = ac->valueint;
                ac = ac->next;
                ALASSERT(NULL!=ac);
                p.mMatrixInfo.iDepth = ac->valueint;
                continue;
            }
            p.mIntValues.insert(std::make_pair(name, c->valueint));
        }
    }
    ALSp<LayerWrap> LayerWrapFactory::create(const cJSON* layer)
    {
        ALSp<LayerWrap> currentLayer;
        ALSp<LayerWrap> nextLayer;

        LayerParameters parameters;
        std::string type = "";
        /*Construct first layer*/
        _readLayerParameters(layer, parameters, type);
        ALSp<LayerWrap> firstLayer = new LayerWrap(LayerFactory::get()->create(type.c_str(), parameters));
        currentLayer = firstLayer;

        for (layer=layer->next;layer!=NULL; layer=layer->next)
        {
            _readLayerParameters(layer, parameters, type);
            nextLayer = new LayerWrap(LayerFactory::get()->create(type.c_str(), parameters));
            currentLayer->connectOutput(nextLayer);
            nextLayer->connectInput(currentLayer.get());
            currentLayer = nextLayer;
        }
        return firstLayer;
    }
};
