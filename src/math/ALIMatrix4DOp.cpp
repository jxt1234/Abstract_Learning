#include "math/ALIMatrix4DOp.h"
#include "ALBasicMatrix4DOp.h"
#include "ALOpenCLMatrix4DOp.hpp"
ALIMatrix4DOp* ALIMatrix4DOp::create(TYPE t)
{
    //TODO
    ALIMatrix4DOp* result = NULL;
    switch (t) {
        case BASIC:
            result = new ALBasicMatrix4DOp;
            break;
        case OPENCL:
            result = new ALOpenCLMatrix4DOp;
            break;
        default:
            break;
    }
    return result;
}

bool ALIMatrix4DOp::Matrix4D::valid() const
{
    return pOrigin != NULL && pOrigin->width() == iWidth*iHeight*iDepth+iExpand;
}

int ALIMatrix4DOp::Matrix4D::getTotalWidth() const
{
    return iDepth*iWidth*iHeight+iExpand;
}
