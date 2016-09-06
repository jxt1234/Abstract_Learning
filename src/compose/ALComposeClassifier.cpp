#include "compose/ALComposeClassifier.h"
ALComposeClassifier::ALComposeClassifier(const ALClassifierSet* set)
{
    ALASSERT(NULL!=set);
    std::vector<ALFLOAT> possibles;
    long sumCount = 0;
    /*Merge Possible and sum for count*/
    for (auto p : set->get())
    {
        mPredictors.push_back(p.first);
        sumCount += p.second;
        auto subPossibleM = p.first->vGetPossiableValues();
        ALASSERT(NULL!=subPossibleM);
        auto subPossibleM_ = subPossibleM->vGetAddr();
        auto w_subPossible = subPossibleM->width();
        /*Merge the possibles*/
        for (int i=0; i<w_subPossible; ++i)
        {
            bool find = false;
            for (auto f : possibles)
            {
                if (ZERO(f-subPossibleM_[i]))
                {
                    find = true;
                    break;
                }
            }
            if (!find)
            {
                possibles.push_back(subPossibleM_[i]);
            }
        }
    }
    ALASSERT(!possibles.empty());
    mPossibles = ALFloatMatrix::create(possibles.size(), 1);
    auto _p = mPossibles->vGetAddr();
    for (int i=0; i<possibles.size(); ++i)
    {
        _p[i] = possibles[i];
    }

    /*Compute for metas*/
    for (auto p : set->get())
    {
        auto count = p.second;
        ALFLOAT weight = (ALFLOAT)count/(ALFLOAT)sumCount;
        auto subPossibleM = p.first->vGetPossiableValues();
        ALASSERT(NULL!=subPossibleM);
        auto subPossibleM_ = subPossibleM->vGetAddr();
        auto w_subPossible = subPossibleM->width();
        std::vector<int> possibleMaps;
        for (int i=0; i<w_subPossible; ++i)
        {
            int pos = -1;
            for (int j=0; j<possibles.size(); ++j)
            {
                if (ZERO(possibles[j]-subPossibleM_[i]))
                {
                    pos = j;
                    break;
                }
            }
            ALASSERT(pos>=0);
            possibleMaps.push_back(pos);
        }
        mMetas.insert(std::make_pair(p.first.get(), std::make_pair(weight, possibleMaps)));
    }
}
ALComposeClassifier::~ALComposeClassifier()
{
}

void ALComposeClassifier::vPredict(const ALFloatMatrix* X, ALFloatMatrix* Y/*Output*/) const
{
    ALASSERT(NULL!=X);
    ALASSERT(NULL!=Y);
    ALSp<ALFloatMatrix> YP = ALFloatMatrix::create(mPossibles->width(), X->height());
    this->vPredictProbability(X, YP.get());
    auto pw = mPossibles->width();
    auto p = mPossibles->vGetAddr();
    auto h = X->height();
    for (int i=0; i<h; ++i)
    {
        auto y = Y->vGetAddr(i);
        auto yp = YP->vGetAddr(i);
        int maxPos = -1;
        ALFLOAT maxValue = -1.0f;
        for (int j=0; j<pw; ++j)
        {
            if (yp[j] > maxValue)
            {
                maxPos = j;
                maxValue = yp[j];
            }
        }
        ALASSERT(maxPos>=0);
        y[0] = p[maxPos];
    }
}

void ALComposeClassifier::vPredictProbability(const ALFloatMatrix* X, ALFloatMatrix* Y/*Output*/) const
{
    ALASSERT(NULL!=X);
    ALASSERT(NULL!=Y);
    ALASSERT(X->height() == Y->height());
    ALASSERT(Y->width() == mPossibles->width());
    auto h = X->height();
    ALFloatMatrix::zero(Y);
    for (auto p : mPredictors)
    {
        auto w = p->vGetPossiableValues()->width();
        auto map = mMetas.find(p.get())->second.second;
        auto weight = mMetas.find(p.get())->second.first;
        ALSp<ALFloatMatrix> subY = ALFloatMatrix::create(w, h);
        p->vPredictProbability(X, subY.get());
        for (int i=0; i<h; ++i)
        {
            auto y = Y->vGetAddr(i);
            auto suby = subY->vGetAddr(i);
            for (int j=0; j<w; ++j)
            {
                y[map[j]] += weight*suby[j];
            }
        }
    }
}
const ALFloatMatrix* ALComposeClassifier::vGetPossiableValues() const
{
    return mPossibles.get();
}
void ALComposeClassifier::vPrint(std::ostream& output) const
{
    for (auto p : mMetas)
    {
        p.first->vPrint(output);
        output << p.second.first<<"\n";
    }
}
