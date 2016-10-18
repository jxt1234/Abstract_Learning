//
//  LayerFactoryRegistor.hpp
//  abs
//
//  Created by jiangxiaotang on 16/10/2016.
//  Copyright © 2016 jiangxiaotang. All rights reserved.
//

#ifndef LayerFactoryRegistor_hpp
#define LayerFactoryRegistor_hpp

#include <stdio.h>

#include <vector>
#include <string>
#include <map>
#include <functional>
#include "ILayer.h"
#include "math/ALIMatrix4DOp.h"

namespace ALCNN {
    struct LayerParameters
    {
        uint32_t uInputSize;
        uint32_t uOutputSize;
        ALIMatrix4DOp::Matrix4D mMatrixInfo;
        std::map<std::string, int> mIntValues;
        
        int get(const std::string& name) const {return mIntValues.find(name)->second;}
    };
    class LayerFactory
    {
    public:
        typedef std::function<ILayer*(const LayerParameters& layerParamters)> CREATOR;
        void insert(CREATOR creator, const char* name);
        static LayerFactory* get();
        ILayer* create(const char* name, const LayerParameters& layerParamters) const;
        //
    private:
        LayerFactory(){}
        ~LayerFactory(){}
        static LayerFactory* gInstance;
        std::map<std::string, CREATOR> mCreators;
    };
    
    class LayerFactoryRegister
    {
    public:
        LayerFactoryRegister(LayerFactory::CREATOR creator, const char* claim)
        {
            LayerFactory* ts = LayerFactory::get();
            ts->insert(creator, claim);
        }
        ~LayerFactoryRegister(){}
    };
}
#endif /* LayerFactoryRegistor_hpp */
