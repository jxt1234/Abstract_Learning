#include "learn/ALGMMClassify.h"
#include "learn/ALGMM.h"


class ALGMMClassifyPredictor:public ALIMatrixPredictor
{
    public:
        ALGMMClassifyPredictor()
        {
        }
        virtual ~ALGMMClassifyPredictor()
        {
        }
        void addModel(ALSp<ALIMatrixPredictor> model, ALFLOAT class_type)
        {
            mGMMModels.push_back(model);
            mTypes.push_back(class_type);
        }
        void finish()
        {
            mTypeMatirx = ALFloatMatrix::create(mTypes.size(), 1);
            auto m = mTypeMatirx->vGetAddr(0);
            for (int i=0; i<mTypes.size(); ++i)
            {
                m[i] = mTypes[i];
            }
        }

        virtual const ALFloatMatrix* vGetPossiableValues() const override
        {
            return mTypeMatirx.get();
        }
        virtual void vPredictProbability(const ALFloatMatrix* X, ALFloatMatrix* Y/*Output*/) const override
        {
            ALASSERT(NULL!=X);
            ALASSERT(NULL!=Y);
            ALASSERT(mTypeMatirx->width() == Y->width());
            ALASSERT(mGMMModels.size()>1);
            ALASSERT(mGMMModels.size() == mTypeMatirx->width());
            ALASSERT(Y->height() == X->height());
            auto h = Y->height();
            for (int i=0; i<mGMMModels.size(); ++i)
            {
                ALSp<ALFloatMatrix> tempY = ALFloatMatrix::createCropVirtualMatrix(Y, i, 0, i, h-1);
                mGMMModels[i]->vPredict(X, tempY.get());
            }
            auto w = Y->width();
            for (int y=0; y<h; ++y)
            {
                auto _y = Y->vGetAddr(y);
                ALFLOAT sum = 0.0;
                for (int x=0; x<w; ++x)
                {
                    sum += _y[x];
                }
                for (int x=0; x<w; ++x)
                {
                    _y[x]/=sum;
                }
            }
        }


        virtual void vPredict(const ALFloatMatrix* X, ALFloatMatrix* Y/*Output*/) const override
        {
            ALASSERT(NULL!=X);
            ALASSERT(NULL!=Y);
            ALASSERT(1 == Y->width());
            ALASSERT(mGMMModels.size()>1);
            ALASSERT(Y->height() == X->height());
            auto h = Y->height();
            ALSp<ALFloatMatrix> maxY = ALFloatMatrix::create(1, h);
            mGMMModels[0]->vPredict(X, maxY.get());
            for (int j=0; j<h; ++j)
            {
                Y->vGetAddr(j)[0] = mTypes[0];
            }
            ALSp<ALFloatMatrix> tempY = ALFloatMatrix::create(1, h);
            for (int i=1; i<mGMMModels.size(); ++i)
            {
                mGMMModels[i]->vPredict(X, tempY.get());
                for (int j=0; j<h; ++j)
                {
                    auto _maxy = maxY->vGetAddr(j);
                    auto _ty = tempY->vGetAddr(j);
                    if (_maxy[0] < _ty[0])
                    {
                        *(Y->vGetAddr(j)) = mTypes[i];
                        _maxy[0] = _ty[0];
                    }
                }
            }
        }
        virtual void vPrint(std::ostream& output) const override
        {
            output << "<ALGMMClassify>\n";
            for (int i=0; i<mGMMModels.size(); ++i)
            {
                output << "<GMMModel>\n";
                mGMMModels[i]->vPrint(output);
                output << "</GMMModel>\n";
            }
            output << "</ALGMMClassify>\n";
        }
    private:
        std::vector<ALSp<ALIMatrixPredictor>> mGMMModels;
        std::vector<ALFLOAT> mTypes;
        ALSp<ALFloatMatrix> mTypeMatirx;

};

ALGMMClassify::ALGMMClassify(int centers)
{
    ALASSERT(centers>1);
    mCenters = centers;
}
ALGMMClassify::~ALGMMClassify()
{
}

static bool _contain(const std::vector<ALFLOAT>& classes, ALFLOAT value)
{
    for (auto c : classes)
    {
        if (ZERO(c-value))
        {
            return true;
        }
    }
    return false;
}

ALIMatrixPredictor* ALGMMClassify::vLearn(const ALFloatMatrix* X, const ALFloatMatrix* Y) const
{
    ALASSERT(NULL!=X);
    ALASSERT(NULL!=Y);
    ALASSERT(X->height() == Y->height());
    ALASSERT(1 == Y->width());
    ALSp<ALFloatMatrix> YT = ALFloatMatrix::transpose(Y);
    auto _yt = YT->vGetAddr(0);
    auto w = X->width();
    auto h = X->height();
    std::vector<ALFLOAT> classes;
    //Find max class
    for (int i=0; i<h; ++i)
    {
        if (!_contain(classes, _yt[i]))
        {
            classes.push_back(_yt[i]);
        }
    }
    ALASSERT(classes.size()>0);
    ALGMMClassifyPredictor* result = new ALGMMClassifyPredictor;
    ALGMM learner(mCenters);
    for (auto c : classes)
    {
        /*Collect*/
        //Measure
        size_t sum = 0;
        for (int i=0; i<h; ++i)
        {
            if(ZERO(_yt[i]-c))
            {
                ++sum;
            }
        }
        ALASSERT(sum>0);
        //Copy
        ALSp<ALFloatMatrix> current = ALFloatMatrix::create(w, sum);
        size_t offset = 0;
        for (int i=0; i<h; ++i)
        {
            if(ZERO(_yt[i]-c))
            {
                ::memcpy(current->vGetAddr(offset), X->vGetAddr(i), w*sizeof(ALFLOAT));
                ++offset;
            }
        }
        //Make model
        ALSp<ALIMatrixPredictor> model = learner.vLearn(current.get());
        result->addModel(model, c);
    }
    result->finish();
    return result;
}
