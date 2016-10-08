#include "math/ALIMatrix4DOp.h"
#include "ALBasicMatrix4DOp.h"
ALIMatrix4DOp* ALIMatrix4DOp::create(TYPE t)
{
    //TODO
    return new ALBasicMatrix4DOp;
}

bool ALIMatrix4DOp::Matrix4D::valid() const
{
    return pOrigin != NULL && pOrigin->width() == iWidth*iHeight*iDepth+iExpand;
}

int ALIMatrix4DOp::Matrix4D::getTotalWidth() const
{
    return iDepth*iWidth*iHeight+iExpand;
}
