#ifndef SRC_LEARN_SVM_SMO_H
#define SRC_LEARN_SVM_SMO_H
#include "math/ALFloatMatrix.h"
namespace ALSMO
{
    class SMO
    {
        public:
            SMO(int n=1);
            ~SMO();
            /*max a: sum(a) - 0.5*sum(ai*aj*yi*yj*k(xi,xj)), 0<=ai<=C, sum(ai*yi)=0*/
            void sovle(ALFloatMatrix* alpha/*output*/, ALFLOAT& b/*output*/, const ALFloatMatrix* CT, const ALFloatMatrix* KX, const ALFloatMatrix* YT, int maxIters);
        private:
    };
};
#endif
