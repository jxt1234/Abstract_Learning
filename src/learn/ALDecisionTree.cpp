#include "learn/ALDecisionTree.h"
#include "math/ALStatistics.h"
#include <vector>
#include <algorithm>

class ALDecisionTree::Tree::Node
{
public:
    Node(size_t k, ALFLOAT d, size_t typesn):mK(k), mD(d)
    {
        ALASSERT(typesn>0);
        mLeft = NULL, mRight = NULL;mLV=0.0;mRV=0.0;
        mLeftProbability = ALFloatMatrix::create(typesn, 1);
        mRightProbability = ALFloatMatrix::create(typesn, 1);
        ALFloatMatrix::zero(mLeftProbability.get());
        ALFloatMatrix::zero(mRightProbability.get());
    }
    ~Node()
    {
        if (NULL!=mLeft)
        {
            delete mLeft;
        }
        if (NULL!=mRight)
        {
            delete mRight;
        }
    }
    void reduce()
    {
        if (NULL == mLeft && NULL == mRight)
        {
            return;
        }
        /*Reduce for single node*/
        if (NULL==mLeft)
        {
            _shallow_copy(mRight);
            mRight->mLeft = NULL;
            mRight->mRight = NULL;
            delete mRight;
            mRight = NULL;
            reduce();
            return;
        }
        else if (NULL==mRight)
        {
            _shallow_copy(mLeft);
            mLeft->mLeft = NULL;
            mLeft->mRight = NULL;
            delete mLeft;
            mLeft = NULL;
            reduce();
            return;
        }
        mLeft->reduce();
        mRight->reduce();
        /*Reduce For two same value*/
        if ((!mLeft->_hasChild()) && ZERO(mLeft->mLV - mLeft->mRV))
        {
            mLV = mLeft->mLV;
            mLeftProbability = mLeft->mLeftProbability;
            delete mLeft;
            mLeft = NULL;
        }
        if ((!mRight->_hasChild()) && ZERO(mRight->mLV - mRight->mRV))
        {
            mRV = mRight->mLV;
            mRightProbability = mRight->mRightProbability;
            delete mRight;
            mRight = NULL;
        }
    }
    ALFLOAT predict(const ALFLOAT* X) const
    {
        if (X[mK]>mD)
        {
            if (NULL!=mRight)
            {
                return mRight->predict(X);
            }
            return mRV;
        }
        if (NULL!=mLeft)
        {
            return mLeft->predict(X);
        }
        return mLV;
    }
    
    void predictProbability(const ALFLOAT* X, ALFLOAT* dst) const
    {
        if (X[mK]>mD)
        {
            if (NULL!=mRight)
            {
                mRight->predictProbability(X, dst);
                return;
            }
            ::memcpy(dst, mRightProbability->vGetAddr(0), sizeof(ALFLOAT)*mRightProbability->width());
            return;
        }
        if (NULL!=mLeft)
        {
            mLeft->predictProbability(X, dst);
            return;
        }
        ::memcpy(dst, mLeftProbability->vGetAddr(0), sizeof(ALFLOAT)*mRightProbability->width());
    }
    void setNode(Node* l, Node* r)
    {
        mLeft = l;
        mRight = r;
    }
    void set(ALFLOAT left, ALFLOAT right)
    {
        mLV = left;
        mRV = right;
    }
    void setProbability(size_t* left, size_t* right, size_t n)
    {
        ALASSERT(n == mLeftProbability->width());
        auto l = mLeftProbability->vGetAddr(0);
        auto r = mRightProbability->vGetAddr(0);
        size_t l_n = 0;
        size_t r_n = 0;
        for (size_t i=0; i<n; ++i)
        {
            l[i] = left[i];
            l_n += left[i];
            r[i] = right[i];
            r_n += right[i];
        }
//        ALASSERT(l_n>0);
//        ALASSERT(r_n>0);
        if (l_n>0)
        {
            for (size_t i=0; i<n; ++i)
            {
                l[i] = l[i]/(ALFLOAT)l_n;
            }
        }
        if (r_n >0)
        {
            for (size_t i=0; i<n; ++i)
            {
                r[i] = r[i]/(ALFLOAT)r_n;
            }
        }
    }
    void printAsJSON(std::ostream& out) const
    {
        out << "{";
        out << "\"divide\":" << mD;
        out << ","<<"\"index\":"<<mK;
        out << ","<<"\"left\":"<<mLV;
        out << ","<<"\"right\":"<<mRV;
        out << ","<<"\"leftProbability\":"<<"[";
        for (int i=0; i<mLeftProbability->width(); ++i)
        {
            out <<mLeftProbability->vGetAddr()[i];
            if (i!=mLeftProbability->width()-1)
            {
                out << ",";
            }
        }
        out <<"]";
        out << ","<<"\"rightProbability\":"<<"[";
        for (int i=0; i<mRightProbability->width(); ++i)
        {
            out <<mRightProbability->vGetAddr()[i];
            if (i!=mRightProbability->width()-1)
            {
                out << ",";
            }
        }
        out <<"]";
        if (NULL!= mLeft)
        {
            out <<",\"leftNode\":";
            mLeft->printAsJSON(out);
        }
        if (NULL!= mRight)
        {
            out <<",\"rightNode\":";
            mRight->printAsJSON(out);
        }
        out << "}";
    }

    
    void printAsXml(std::ostream& out) const
    {
        out << "<Condition:";
        out << "\"x"<<mK<<"<="<<mD<<"\">\n";
        if (NULL != mLeft)
        {
            mLeft->printAsXml(out);
        }
        else
        {
            ALFloatMatrix::print(mLeftProbability.get(), out);
        }
        out << "</Condition>\n";
        out << "<Condition:";
        out << "\"x"<<mK<<">"<<mD<<"\">\n";
        if (NULL != mRight)
        {
            mRight->printAsXml(out);
        }
        else
        {
            ALFloatMatrix::print(mRightProbability.get(), out);
        }
        out << "</Condition>\n";
        
    }
    void print(std::ostream& out) const
    {
        out << "(x"<<mK<<" <="<<mD<<" ? ";
        if (mLeft != NULL)
        {
            mLeft->print(out);
        }
        else
        {
            out << mLV;
        }
        out << " : ";
        if (mRight != NULL)
        {
            mRight->print(out);
        }
        else
        {
            out << mRV;
        }
        out << ")";
    }
private:
    bool _hasChild() const
    {
        return (NULL!=mLeft)||(NULL!=mRight);
    }
    void _shallow_copy(Node* n)
    {
        ALASSERT(NULL!=n);
        mK = n->mK;
        mD = n->mD;
        mLV = n->mLV;
        mRV = n->mRV;
        mLeft = n->mLeft;
        mRight = n->mRight;
        mLeftProbability = n->mLeftProbability;
        mRightProbability = n->mRightProbability;
    }
    size_t mK;
    ALFLOAT mD;
    ALFLOAT mLV;
    ALFLOAT mRV;
    Node* mLeft;
    Node* mRight;
    ALSp<ALFloatMatrix> mLeftProbability;
    ALSp<ALFloatMatrix> mRightProbability;
};

void ALDecisionTree::Tree::print(std::ostream& out)
{
    mRoot->printAsXml(out);
}

ALDecisionTree::Tree::Tree(Node* r)
{
    ALASSERT(NULL!=r);
    mRoot = r;
}
ALDecisionTree::Tree::~Tree()
{
    ALASSERT(NULL!=mRoot);
    delete mRoot;
}
ALFLOAT ALDecisionTree::Tree::predict(const ALFLOAT* X)
{
    ALASSERT(NULL!=X);
    ALASSERT(NULL!=mRoot);
    return mRoot->predict(X);
}

void ALDecisionTree::Tree::predictProbability(const ALFLOAT* X, ALFLOAT* Y)
{
    ALASSERT(NULL!=X);
    ALASSERT(NULL!=mRoot);
    mRoot->predictProbability(X, Y);
}


ALDecisionTree::ALDecisionTree(size_t maxdepth, bool needreduce, bool discrete)
{
    mMaxDepth = maxdepth;
    mNeedReduce = needreduce;
    mDiscrete = discrete;
}
ALDecisionTree::~ALDecisionTree()
{
}

ALIMatrixPredictor* ALDecisionTree::vLearn(const ALFloatMatrix* X, const ALFloatMatrix* Y) const
{
    ALSp<ALFloatMatrix> YT = ALFloatMatrix::transpose(Y);
    return learnT(X, YT.get());
}
ALDecisionTree::Tree* ALDecisionTree::train(const ALFloatMatrix* X, const ALFloatMatrix* YT, const ALFloatMatrix* types) const
{
    ALASSERT(NULL!=types);
    Tree::Node* n = _train(X, YT, 0, types);
    if (NULL == n)
    {
        n = new Tree::Node(0, 0, types->width());
    }
    ALASSERT(NULL!=n);
    if (mNeedReduce)
    {
        //n->reduce();
    }
    return new ALDecisionTree::Tree(n);
}

/*return left size*/
static size_t collectLR(const std::vector<ALFLOAT>& types, const ALFloatMatrix* YT, const ALFloatMatrix* XT, ALFLOAT v, size_t k, size_t* l/*OUTPUT*/, size_t* r/*OUTPUT*/)
{
    ALAUTOTIME;
    size_t leftsize = 0;
    auto w = YT->width();
    ALASSERT(w == XT->width());
    auto y = YT->vGetAddr();
    auto x = XT->vGetAddr();
    /*Compute prune*/
    for (size_t i=0; i<w; ++i)
    {
        auto yv = y[i];
        auto xv = x[i];
        for (size_t j=0; j<types.size(); ++j)
        {
            if (ZERO(types[j]-yv))
            {
                if (xv > v)
                {
                    r[j]++;
                }
                else
                {
                    l[j]++;
                    leftsize++;
                }
                break;
            }
        }
    }
    return leftsize;
}

static void _GenerateSpliteValues(const ALFloatMatrix* X, std::vector<ALFLOAT>* target, size_t w)
{
    ALASSERT(NULL!=X);
    ALASSERT(NULL!=target);
    ALASSERT(w == X->width());
    const int step = 1;
    ALSp<ALFloatMatrix> stas = ALStatistics::statistics(X);
    auto Max = stas->vGetAddr(1);
    auto Min = stas->vGetAddr(2);
    for (int k=0; k<w; ++k)
    {
        auto _max = Max[k];
        auto _min = Min[k];
        std::vector<ALFLOAT> values;
//        FUNC_PRINT_ALL(_max, f);
//        FUNC_PRINT_ALL(_min, f);
        if (ZERO(_max-_min))
        {
            target[k] = values;
            continue;
        }
        for (int i=0; i<step; ++i)
        {
            ALFLOAT v = _min + (_max - _min)*(i+1)/(ALFLOAT)(step+1);
            values.push_back(v);
        }
        target[k] = values;
    }
}

/*Use Gini formula: Gini=1-sum(Pi*Pi)*/
ALDecisionTree::Tree::Node* ALDecisionTree::_train(const ALFloatMatrix* X, const ALFloatMatrix* YT, size_t depth, const ALFloatMatrix* typesM) const
{
    ALASSERT(NULL!=X);
    ALASSERT(NULL!=YT);
    ALASSERT(X->height() == YT->width());
    ALASSERT(1 == YT->height());
    ALASSERT(NULL!=typesM);
    std::vector<ALFLOAT> types;
    {
        for (size_t i=0; i<typesM->width(); ++i)
        {
            types.push_back(typesM->vGetAddr()[i]);
        }
    }
    auto w = X->width();
    auto h = X->height();
    /*Compute for the min prune*/
    size_t maxk = 0;
    ALFLOAT maxv = 0;
    ALFLOAT prune = 1.0;
    ALAUTOSTORAGE(numbersleft, size_t, types.size());
    ALAUTOSTORAGE(numbersright, size_t, types.size());
    
    ALAUTOSTORAGE(possibleValues, std::vector<ALFLOAT>, w);
    _GenerateSpliteValues(X, possibleValues, w);
    {
        //ALFORCEAUTOTIME;
        for (size_t k=0; k<w; ++k)
        {
            ALSp<ALFloatMatrix> XT = ALFloatMatrix::createCropVirtualMatrix(X, k, 0, k, YT->width()-1);
            XT = ALFloatMatrix::transpose(XT.get());
            for (auto v : possibleValues[k])
            {
                /*Collect*/
                ::memset(numbersleft, 0, sizeof(size_t)*types.size());
                ::memset(numbersright, 0, sizeof(size_t)*types.size());
                auto leftsize = collectLR(types, YT, XT.get(), v, k, numbersleft, numbersright);
                ALFLOAT suml = 1.0f;
                ALFLOAT sumr = 1.0f;
                for (size_t i=0; i<types.size(); ++i)
                {
                    suml = suml - (ALFLOAT)numbersleft[i]*(ALFLOAT)numbersleft[i]/(ALFLOAT)leftsize/(ALFLOAT)leftsize;
                    sumr = sumr - (ALFLOAT)numbersright[i]*(ALFLOAT)numbersright[i]/(ALFLOAT)(h-leftsize)/(ALFLOAT)(h-leftsize);
                }
                auto newprune = (suml*leftsize+sumr*(h-leftsize))/(ALFLOAT)h;
                if (newprune < prune)
                {
                    maxk = k;
                    maxv = v;
                    prune = newprune;
                }
            }
        }
    }
    if (prune >= 1.0f)
    {
        return NULL;
    }
    auto result = new Tree::Node(maxk, maxv, types.size());
    ::memset(numbersleft, 0, sizeof(size_t)*types.size());
    ::memset(numbersright, 0, sizeof(size_t)*types.size());
    
    ALSp<ALFloatMatrix> XT = ALFloatMatrix::createCropVirtualMatrix(X, maxk, 0, maxk, YT->width()-1);
    XT = ALFloatMatrix::transpose(XT.get());
    auto leftsize = collectLR(types, YT, XT.get(), maxv, maxk, numbersleft, numbersright);
    XT = NULL;
    //FUNC_PRINT_ALL(prune, f);
    size_t l = ALStatistics::max(numbersleft, types.size());
    size_t r = ALStatistics::max(numbersright, types.size());
    result->set(types[l], types[r]);
    result->setProbability(numbersleft, numbersright, types.size());
    if (0==leftsize || leftsize==h || depth >= mMaxDepth || prune <= 0.0001f)
    {
        return result;
    }
    ALSp<ALFloatMatrix> XL;
    ALSp<ALFloatMatrix> XR;
    ALSp<ALFloatMatrix> YL;
    ALSp<ALFloatMatrix> YR;
    auto rightSize = h - leftsize;
    YL = ALFloatMatrix::create(leftsize, 1);
    YR = ALFloatMatrix::create(rightSize, 1);
    ALAUTOSTORAGE(xl, ALFLOAT*, leftsize);
    auto yl = YL->vGetAddr();
    ALAUTOSTORAGE(xr, ALFLOAT*, h-leftsize);
    auto yr = YR->vGetAddr();
    size_t left=0;
    size_t right=0;
    auto y = YT->vGetAddr();
    for (size_t i=0; i<X->height();++i)
    {
        auto x = X->vGetAddr(i);
        auto y_v = y[i];
        if (x[maxk]>maxv)
        {
            xr[right] = x;
            yr[right] = y_v;
            ++right;
        }
        else
        {
            xl[left] = x;
            yl[left] = y_v;
            ++left;
        }
    }
    XL = ALFloatMatrix::createIndexVirtualMatrix(xl, X->width(), leftsize);
    XR = ALFloatMatrix::createIndexVirtualMatrix(xr, X->width(), h-leftsize);

    Tree::Node* leftNode = NULL;
    Tree::Node* rightNode = NULL;
    if (l < leftsize)
    {
        leftNode = _train(XL.get(), YL.get(), depth+1, typesM);
    }
    if (r < h-leftsize)
    {
        rightNode = _train(XR.get(), YR.get(), depth+1, typesM);
    }
    result->setNode(leftNode, rightNode);
    return result;
}


void ALDecisionTree::setFixTypes(ALSp<ALFloatMatrix> types)
{
    mFixTypes = types;
}

ALIMatrixPredictor* ALDecisionTree::learnT(const ALFloatMatrix* X, const ALFloatMatrix* YT) const
{
    ALSp<ALFloatMatrix> types = mFixTypes;
    if (NULL == types.get())
    {
        types = ALFloatMatrix::genTypes(YT);
    }
    ALSp<Tree> t = train(X, YT, types.get());
    class DecitonTreeP:public ALIMatrixPredictor
    {
    public:
        DecitonTreeP(ALSp<ALDecisionTree::Tree> t, const ALFloatMatrix* types):mTree(t)
        {
            mTypes = ALFloatMatrix::create(types->width(), 1);
            ALFloatMatrix::copy(mTypes.get(), types);
        }
        virtual ~DecitonTreeP(){}
        virtual void vPredict(const ALFloatMatrix* X, ALFloatMatrix* Y/*Output*/) const override
        {
            ALASSERT(NULL!=X);
            ALASSERT(NULL!=Y);
            ALASSERT(X->height() == Y->height());
            auto h = Y->height();
            for (int i=0; i<h; ++i)
            {
                auto x = X->vGetAddr(i);
                auto y = Y->vGetAddr(i);
                y[0] = mTree->predict(x);
            }
        }
        virtual void vPredictProbability(const ALFloatMatrix* X, ALFloatMatrix* Y/*Output*/) const override
        {
            ALASSERT(NULL!=X);
            ALASSERT(NULL!=Y);
            ALASSERT(X->height() == Y->height());
            ALASSERT(Y->width() == mTypes->width());
            auto h = Y->height();
            for (int i=0; i<h; ++i)
            {
                auto x = X->vGetAddr(i);
                auto y = Y->vGetAddr(i);
                mTree->predictProbability(x, y);
            }
        }
        virtual const ALFloatMatrix* vGetPossiableValues() const override
        {
            return mTypes.get();
        }
        
        virtual void vPrint(std::ostream& out) const override
        {
            out << "{";
            out << "\"types\":" << "[";
            for (int i=0; i<mTypes->width(); ++i)
            {
                out <<mTypes->vGetAddr()[i];
                if (i!=mTypes->width()-1)
                {
                    out <<",";
                }
            }
            out << "]";
            out << ",\"root\":";
            mTree->root()->printAsJSON(out);
            out << "}";
        }
    private:
        ALSp<ALDecisionTree::Tree> mTree;
        ALSp<ALFloatMatrix> mTypes;
    };
    return new DecitonTreeP(t, types.get());
}
