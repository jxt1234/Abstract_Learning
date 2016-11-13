#ifndef LEARN_CNN_SELECTLASTLAYER_H
#define LEARN_CNN_SELECTLASTLAYER_H
#include "ILayer.h"
namespace ALCNN {
    class SelectLastLayer : public ILayer
    {
        public:
            SelectLastLayer(size_t iw, size_t ow);
            virtual ~ SelectLastLayer();
            virtual void vForward(const ALFloatMatrix* before, ALFloatMatrix* after, const ALFloatMatrix* parameters, ALFloatMatrix* cache) const override;
            virtual void vBackward(const ALFloatMatrix* after_diff, const ALFloatMatrix* after, const ALFloatMatrix* parameters, const ALFloatMatrix* before, ALFloatMatrix* before_diff, ALFloatMatrix* parameters_diff, ALFloatMatrix* cache) const override;
        private:
            size_t mOw;
    };
};
#endif
