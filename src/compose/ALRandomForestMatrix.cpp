#include "compose/ALRandomForestMatrix.h"
#include "math/ALIMatrixTransformer.h"
#include "learn/ALMatrixSelector.h"
#include "learn/ALDecisionTree.h"
#include "math/ALStatistics.h"
#include <math.h>
#include "compose/ALComposeClassifier.h"
class ALRandomForestMatrix_Predictor:public ALIMatrixPredictor
{
public:
    ALRandomForestMatrix_Predictor(const std::vector<ALFLOAT>& values):mValues(values)
    {
        mValuesM = ALFloatMatrix::create(values.size(), 1);
        auto _v = mValuesM->vGetAddr();
        for (size_t i=0; i<values.size(); ++i)
        {
            _v[i] = values[i];
        }
    }
    virtual ~ALRandomForestMatrix_Predictor(){}
    virtual void vPredict(const ALFloatMatrix* X, ALFloatMatrix* Y/*Output*/) const override
    {
        ALASSERT(NULL!=X);
        ALASSERT(NULL!=Y);
        ALASSERT(1 == Y->width());
        ALASSERT(!mTrees.empty());
        ALASSERT(X->height() == Y->height());
        ALSp<ALFloatMatrix> YPS = ALFloatMatrix::create(mTrees.size(), Y->height());
        ALFloatMatrix::zero(YPS.get());
        for (size_t i=0; i<mTrees.size(); ++i)
        {
            auto iter = mTrees[i];
            ALSp<ALFloatMatrix> YPS_Unit = ALFloatMatrix::create(mValues.size(), Y->height());
            if (NULL!=iter.first.get())
            {
                ALSp<ALFloatMatrix> X_T = iter.first->vTransform(X);
                iter.second->vPredictProbability(X_T.get(), YPS_Unit.get());
            }
            else
            {
                iter.second->vPredictProbability(X, YPS_Unit.get());
            }
            for (size_t j=0;j<Y->height(); ++j)
            {
                auto yps = YPS->vGetAddr(j);
                auto yunit = YPS_Unit->vGetAddr(j);
                for (size_t k=0; k<mValues.size(); ++k)
                {
                    yps[k] += yunit[k];
                }
            }
        }
        auto h = Y->height();
        auto w = YPS->width();
        for (size_t i=0; i<h; ++i)
        {
            auto y_ = YPS->vGetAddr(i);
            auto y = Y->vGetAddr(i);
            auto pos = ALStatistics::max(y_, w);
            y[0] = mValues[pos];
        }
    }
    virtual void vPrint(std::ostream& output) const override
    {
        output << "<Forest>\n";
        for (auto iter :mTrees)
        {
            if (NULL!=iter.first.get())
            {
                output << "<Transformer>\n";
                iter.first->vPrint(output);
                output << "</Transformer>\n";
            }
            output << "<Predictor>\n";
            iter.second->vPrint(output);
            output << "</Predictor>\n";
        }
        output << "</Forest>\n";
    }
    virtual void vPredictProbability(const ALFloatMatrix* X, ALFloatMatrix* Y/*Output*/) const override
    {
        ALASSERT(NULL!=X);
        ALASSERT(NULL!=Y);
        ALASSERT(!mTrees.empty());
        ALASSERT(Y->width() == mValues.size());
        ALASSERT(X->height() == Y->height());
        ALFloatMatrix::zero(Y);
        for (size_t i=0; i<mTrees.size(); ++i)
        {
            auto iter = mTrees[i];
            ALSp<ALFloatMatrix> YPS_Unit = ALFloatMatrix::create(mValues.size(), Y->height());
            if (NULL!=iter.first.get())
            {
                ALSp<ALFloatMatrix> X_T = iter.first->vTransform(X);
                iter.second->vPredictProbability(X_T.get(), YPS_Unit.get());
            }
            else
            {
                iter.second->vPredictProbability(X, YPS_Unit.get());
            }
            for (size_t j=0;j<Y->height(); ++j)
            {
                auto yps = Y->vGetAddr(j);
                auto yunit = YPS_Unit->vGetAddr(j);
                for (size_t k=0; k<mValues.size(); ++k)
                {
                    yps[k] += yunit[k];
                }
            }
        }
        /*Normalize*/
        auto yw = mValues.size();
        for (size_t j=0;j<Y->height(); ++j)
        {
            auto yps = Y->vGetAddr(j);
            ALFLOAT sum = 0.0f;
            for (auto k=0; k<yw; ++k)
            {
                sum += yps[k];
            }
            for (auto k=0; k<yw; ++k)
            {
                yps[k]/=sum;
            }
        }
    }
    virtual const ALFloatMatrix* vGetPossiableValues() const override
    {
        return mValuesM.get();
    }

    void insert(ALSp<ALIMatrixTransformer> transform, ALSp<ALIMatrixPredictor> tree)
    {
        ALASSERT(NULL!=tree->vGetPossiableValues());
        ALASSERT(tree->vGetPossiableValues()->width() == mValues.size());
        mTrees.push_back(std::make_pair(transform, tree));
    }
private:
    std::vector<ALFLOAT> mValues;
    ALSp<ALFloatMatrix> mValuesM;
    std::vector<std::pair<ALSp<ALIMatrixTransformer>, ALSp<ALIMatrixPredictor>>> mTrees;
};



ALRandomForestMatrix::ALRandomForestMatrix(int treeNum, bool discrete)
{
    ALRandom::init();
    ALASSERT(treeNum>0);
    mTree = treeNum;
    mDiscrete = discrete;
}
ALRandomForestMatrix::~ALRandomForestMatrix()
{
}

ALIMatrixPredictor* ALRandomForestMatrix::vLearn(const ALFloatMatrix* X, const ALFloatMatrix* Y) const
{
    ALASSERT(NULL!=X);
    ALASSERT(NULL!=Y);
    ALASSERT(X->height() == Y->height());
    auto h = X->height();
    auto w = X->width();
    ALSp<ALFloatMatrix> YT = ALFloatMatrix::transpose(Y);
    ALSp<ALFloatMatrix> types = ALFloatMatrix::genTypes(YT.get());
    
    size_t random_h = 0.65*h;
    if (random_h < 5)
    {
        return new ALDummyMatrixPredictor;
    }
    size_t random_w = sqrt(w);
    if (random_w < 1)
    {
        return new ALDummyMatrixPredictor;
    }
    std::vector<ALFLOAT> values;
    for (size_t i=0; i<types->width(); ++i)
    {
        values.push_back(types->vGetAddr()[i]);
    }
    ALRandomForestMatrix_Predictor* predictor = new ALRandomForestMatrix_Predictor(values);
    /*Random Set the Tree number*/
    int number = mTree;
    ALAUTOSTORAGE(X_Random_Addr, ALFLOAT*, random_h);
    ALSp<ALFloatMatrix> Y_Random_T = ALFloatMatrix::create(random_h, 1);
    ALSp<ALDecisionTree> learner = new ALDecisionTree(10, false, mDiscrete);
    learner->setFixTypes(types);
    auto y_random_t = Y_Random_T->vGetAddr();
    auto y_t = YT->vGetAddr();
    ALSp<ALClassifierSet> set = new ALClassifierSet;
    for (int i=0; i<number; ++i)
    {
        /*Random Select TrainData*/
        for (size_t k=0; k<random_h; ++k)
        {
            auto select = ALRandom::mid(0, h);
            X_Random_Addr[k] = X->vGetAddr(select);
            y_random_t[k] = y_t[select];
        }
        /*Random Select Feature*/
        std::vector<int> pos;
        for (size_t k=0; k<random_w; ++k)
        {
            pos.push_back(ALRandom::mid(0, w));
        }
        ALSp<ALFloatMatrix> X_Random = ALFloatMatrix::createIndexVirtualMatrix(X_Random_Addr, w, random_h);
        ALSp<ALIMatrixTransformer> transformer = new ALMatrixSelector(pos);
        ALSp<ALFloatMatrix> X_transform = transformer->vTransform(X_Random.get());
        ALSp<ALIMatrixPredictor> tree = learner->learnT(X_transform.get(), Y_Random_T.get());
        predictor->insert(transformer, tree);
    }
    return predictor;
}

