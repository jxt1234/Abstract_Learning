#include "learn/ALKMeans.h"
static ALFLOAT distance_2(ALFLOAT* x1, ALFLOAT* x2, size_t w)
{
    ALFLOAT sum = 0;
    for (size_t i=0; i<w; ++i)
    {
        ALFLOAT dif = x1[i] - x2[i];
        sum+=(dif*dif);
    }
    return sum;
}

ALKMeans::ALKMeans(size_t class_number, size_t iter):mNumber(class_number), mIter(iter)
{
}
ALKMeans::~ALKMeans()
{
}
ALIMatrixPredictor* ALKMeans::vLearn(const ALFloatMatrix* X) const
{
    class predicter:public ALIMatrixPredictor
    {
    public:
        predicter(ALSp<ALFloatMatrix> center):mCenter(center){}
        virtual ~predicter(){}
        virtual void vPredict(const ALFloatMatrix* X, ALFloatMatrix* Y/*Output*/) const override
        {
            ALKMeans::predict(X, mCenter.get(), Y);
        }
        virtual void vPrint(std::ostream& output) const override
        {
            output << "<ALKMeansCenter>"<<std::endl;
            ALFloatMatrix::print(mCenter.get(),output);
            output << "</ALKMeansCenter>"<<std::endl;
        }
    private:
        ALSp<ALFloatMatrix> mCenter;
    };
    ALSp<ALFloatMatrix> center = learn(X, mNumber, mIter);
    if (NULL == center.get())
    {
        return new ALDummyMatrixPredictor;
    }
    return new predicter(center);
}
void ALKMeans::predict(const ALFloatMatrix* X, const ALFloatMatrix* Center, ALFloatMatrix* R)
{
    ALASSERT(NULL!=X);
    ALASSERT(NULL!=Center);
    ALASSERT(NULL!=R);
    ALASSERT(X->width() == Center->width());
    ALASSERT(R->height() == X->height());
    ALASSERT(1 == R->width());
    for (size_t i=0; i<R->height(); ++i)
    {
        auto r = R->vGetAddr(i);
        ALFLOAT* x = X->vGetAddr(i);
        ALFLOAT mindistanse = distance_2(x, Center->vGetAddr(0), X->width());
        size_t type = 0;
        for (auto j=1; j<Center->height(); ++j)
        {
            ALFLOAT* c = Center->vGetAddr(j);
            ALFLOAT dis = distance_2(x, c, X->width());
            if (mindistanse > dis)
            {
                mindistanse = dis;
                type = j;
            }
        }
        r[0] = type;
    }
}
ALSp<ALFloatMatrix> ALKMeans::learn(const ALFloatMatrix* X, size_t number, size_t maxiter)
{
    ALASSERT(NULL!=X);
    ALASSERT(number>1);
    if(X->height() <= number)
    {
        return NULL;
    }
    ALSp<ALFloatMatrix> Center = ALFloatMatrix::create(X->width(), number);
    /*Initial origin center by x[0], x[d], x[2d], ..., x[kd]*/
    {
        auto d = X->height()/number;
        for (size_t i=0; i<number; ++i)
        {
            ALFLOAT* c = Center->vGetAddr(i);
            ALFLOAT* x = X->vGetAddr(i*d);
            ::memcpy(c, x, (X->width())*sizeof(ALFLOAT));
        }
    }
    ALSp<ALFloatMatrix> newCenter = ALFloatMatrix::create(X->width(), number);
    ALAutoStorage<size_t> _datanumbers(number);
    for (auto iter=0; iter<maxiter; ++iter)
    {
        size_t* n = _datanumbers.get();
        ::memset(n, 0, sizeof(size_t)*number);
        ALFloatMatrix::zero(newCenter.get());
        for (auto i=0; i<X->height(); ++i)
        {
            ALFLOAT* x = X->vGetAddr(i);
            ALFLOAT mindistanse = distance_2(x, Center->vGetAddr(), X->width());
            size_t type = 0;
            for (auto j=1; j<Center->height(); ++j)
            {
                ALFLOAT* c = Center->vGetAddr(j);
                ALFLOAT dis = distance_2(x, c, X->width());
                if (mindistanse > dis)
                {
                    mindistanse = dis;
                    type = j;
                }
            }
            ++n[type];
            ALFLOAT* newc = newCenter->vGetAddr(type);
            for (auto j=0; j<X->width(); ++j)
            {
                newc[j] += x[j];
            }
        }
        for (auto i=0; i<number; ++i)
        {
            if(0 == n[i])
            {
                //ALASSERT(0);
                continue;
            }
            ALFLOAT* c = newCenter->vGetAddr(i);
            for (auto k=0; k<newCenter->width(); ++k)
            {
                c[k] = c[k]/(ALFLOAT)(n[i]);
            }
        }
        bool ok = true;
        /*Compare new Center and old Center, if they are the same, break*/
        for (auto i=0; i<number; ++i)
        {
            ALFLOAT* c = newCenter->vGetAddr(i);
            ALFLOAT dis = distance_2(c, Center->vGetAddr(i), Center->width());
            if (!ZERO(dis))
            {
                ok = false;
            }
        }
        if (ok)
        {
            break;
        }
        else
        {
            ALFloatMatrix::copy(Center.get(), newCenter.get());
        }
    }
    return Center;
}
