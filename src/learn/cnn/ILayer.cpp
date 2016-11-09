#include "ILayer.h"
namespace ALCNN{
    ILayer::ILayer(size_t iw, size_t ow, size_t pw, size_t ph, size_t cw, size_t ch)
    {
        mInfo.iw = iw;
        mInfo.ow = ow;
        mInfo.pw = pw;
        mInfo.ph = ph;
        mInfo.cw = cw;
        mInfo.ch = ch;

        //FIXME
        //ALASSERT(pw < 10000000);
        //ALASSERT(ph < 10000000);
    }
}
