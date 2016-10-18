//
//  LayerFactoryRegistor.cpp
//  abs
//
//  Created by jiangxiaotang on 16/10/2016.
//  Copyright © 2016 jiangxiaotang. All rights reserved.
//

#include "LayerFactoryRegistor.hpp"
namespace ALCNN {
    LayerFactory* LayerFactory::gInstance = NULL;
    
    void LayerFactory::insert(CREATOR creator, const char* name)
    {
        mCreators.insert(std::make_pair(name, creator));
    }
    LayerFactory* LayerFactory::get()
    {
        if (NULL == gInstance)
        {
            gInstance = new LayerFactory;
        }
        return gInstance;
    }
    ILayer* LayerFactory::create(const char* name, const LayerParameters& layerParamters) const
    {
        auto iter = mCreators.find(name);
        ALASSERT(iter != mCreators.end());//TODO
        return (iter->second)(layerParamters);
    }
}
