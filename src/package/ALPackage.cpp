#include "package/ALPackage.h"
#include "core/ALExpanderFactory.h"
#include "learn/ALGMMClassify.h"
#include "learn/ALDecisionTree.h"
#include "learn/ALLogicalRegress.h"
#include "compose/ALRandomForestMatrix.h"
#include "compose/ALComposeClassifier.h"
#include "learn/ALCGPRegressor.h"
#include "learn/ALNaiveBayesianLearner.h"
#include "math/ALIGradientDecent.h"
#include "cJSON/cJSON.h"
#include "learn/ALCNNLearner.h"

ALFloatPredictor* ALPackageLearn(ALIChainLearner* l, ALLabeldData* c)
{
    ALFloatPredictor* p = l->vLearn(c);
    return p;
}
ALIChainLearner* ALPackageCreateDivider(ALIChainLearner* l, ALIChainLearner* r, ALDividerParameter* p)
{
    ALDivider* d = new ALDivider(p->per, p->step, l, r);
    return d;
}
ALISuperviseLearner* ALPackageCreateRegress()
{
    return new ALRegressor;
}
ALFloatDataChain* ALPackageLoadData(char* file)
{
    ALStandardLoader s;
    ALFloatDataChain* c = s.load(file);
    return c;
}

double ALPackageCrossValidate(ALIChainLearner* l, ALLabeldData* c)
{
    auto fres = ALLearnFactory::crossValidate(l, c);
    return fres;
}
ALLabeldData* ALPackageLabled(ALFloatDataChain* c, double delay)
{
    ALSp<ALILabeldMethod> m = ALLabeldMethodFactory::createBasic();
    return ALLabeldMethodFactory::delayLabel(c->get(), m.get(), delay);
}

ALIChainLearner* ALPackageCombine(ALARStructure* ar, ALISuperviseLearner* l)
{
    l->addRef();
    ALSp<ALISuperviseLearner> spl = l;
    return ALIChainLearner::createFromBasic(spl, ALExpanderFactory::createAR(*ar));
}
ALISuperviseLearner* ALPackageCreateSVM(ALSVMParameter* p)
{
    return new ALSVMLearner(p);
}

static ALSp<ALFloatMatrix> _extractX(ALFloatMatrix* m)
{
    ALASSERT(NULL!=m);
    ALASSERT(m->width()>1);
    ALFloatMatrix* x = ALFloatMatrix::create(m->width()-1, m->height());
    auto w = m->width() -1;
    for (size_t i=0; i<m->height(); ++i)
    {
        auto src = m->vGetAddr(i);
        auto dst = x->vGetAddr(i);
        ::memcpy(dst, src+1, w*sizeof(ALFLOAT));
    }
    return x;
}

static ALSp<ALFloatMatrix> _extractY(ALFloatMatrix* m)
{
    ALASSERT(NULL!=m);
    ALASSERT(m->width()>1);
    ALFloatMatrix* y = ALFloatMatrix::create(1, m->height());
    for (size_t i=0; i<m->height(); ++i)
    {
        auto src = m->vGetAddr(i);
        auto dst = y->vGetAddr(i);
        dst[0] = src[0];
    }
    return y;
}


ALFloatMatrix* ALPackageValidateMatrix(ALIMatrixPredictor* l, ALFloatMatrix* m)
{
    ALASSERT(NULL!=m);
    ALASSERT(NULL!=l);
    ALSp<ALFloatMatrix> YP = ALFloatMatrix::create(1, m->height());
    ALSp<ALFloatMatrix> X = _extractX(m);
    ALSp<ALFloatMatrix> Y = _extractY(m);
    l->vPredict(X.get(), YP.get());
    ALFloatMatrix* res = ALFloatMatrix::create(3, m->height());
    for (size_t i=0; i<m->height(); ++i)
    {
        auto yp = YP->vGetAddr(i);
        auto y = Y->vGetAddr(i);
        auto dst = res->vGetAddr(i);
        dst[0] = *y;
        dst[1] = *yp;
        dst[2] = *yp - *y;
    }
    return res;
}
ALFloatMatrix* ALPackageValidateChain(ALFloatPredictor* l, ALLabeldData* c)
{
    ALFloatMatrix* res = ALFloatMatrix::create(3, c->size());
    auto data = c->get();
    for (size_t i=0; i<c->size(); ++i)
    {
        auto d = data[i];
        auto dst = res->vGetAddr(i);
        dst[0] = d.first;
        dst[1] = l->vPredict(d.second);
        dst[2] = dst[1]-dst[0];
    }
    return res;
}
ALIMatrixPredictor* ALPackageSuperLearning(ALISuperviseLearner* l, ALFloatMatrix* m)
{
    return l->vLearn(_extractX(m).get(), _extractY(m).get());
}
ALIMatrixPredictor* ALPackageUnSuperLearning(ALIUnSuperLearner* l, ALFloatMatrix* m)
{
    return l->vLearn(_extractX(m).get());
}

ALISuperviseLearner* ALPackageCreateGMM()
{
    return new ALGMMClassify;
}

ALISuperviseLearner* ALPackageCreateDecisionTree(ALDecisionTreeParameter* p)
{
    return new ALDecisionTree(p->maxDepth);
}
ALISuperviseLearner* ALPackageCreateLogicalRegress()
{
    return new ALLogicalRegress;
}
ALClassifier* ALPackageCreateClassify(ALClassifierCreator* l, ALFloatMatrix* m)
{
    ALASSERT(NULL!=m);
    ALASSERT(NULL!=l);
    ALASSERT(m->width() > 1);
    ALSp<ALFloatMatrix> X = ALFloatMatrix::createCropVirtualMatrix(m, 1, 0, m->width()-1, m->height()-1);
    ALSp<ALFloatMatrix> Y = ALFloatMatrix::createCropVirtualMatrix(m, 0, 0, 0, m->height()-1);
    return l->vLearn(X.get(), Y.get());
}
ALFloatMatrix* ALPackageClassify(ALClassifier* l, ALFloatMatrix* m)
{
    ALFloatMatrix* Y = ALFloatMatrix::create(1, m->height());
    l->vPredict(m, Y);
    return Y;
}

ALFloatMatrix* ALPackageClassifyProb(ALClassifier* l, ALFloatMatrix* m)
{
    ALASSERT(NULL!=l->vGetPossiableValues());
    ALFloatMatrix* Y = ALFloatMatrix::create(l->vGetPossiableValues()->width(), m->height());
    l->vPredictProbability(m, Y);
    return Y;
}

ALFloatMatrix* ALPackageClassifyProbValues(ALClassifier* l)
{
    const ALFloatMatrix* p = l->vGetPossiableValues();
    ALFloatMatrix* p_copy = ALFloatMatrix::create(p->width(), p->height());
    ALFloatMatrix::copy(p_copy, p);
    return p_copy;
}

double ALPackageCrossValidateClassify(ALClassifierCreator* l, ALFloatMatrix* m)
{
    ALASSERT(m->width() > 1);
    ALSp<ALFloatMatrix> X = ALFloatMatrix::createCropVirtualMatrix(m, 1, 0, m->width()-1, m->height()-1);
    ALSp<ALFloatMatrix> Y = ALFloatMatrix::createCropVirtualMatrix(m, 0, 0, 0, m->height()-1);
    return ALLearnFactory::crossValidateForClassify(l, X.get(), Y.get());
}

ALFloatMatrix* ALPackageMatrixMerge(ALFloatMatrix* A, ALFloatMatrix* B, double aleft, double aright, double bleft, double bright)
{
    ALASSERT(NULL!=A);
    ALASSERT(NULL!=B);
    ALASSERT(A->height() == B->height());
    ALASSERT(aleft>=0);
    ALASSERT(aright>=aleft);
    ALASSERT(aright<A->width());
    ALASSERT(bleft>=0);
    ALASSERT(bright>=bleft);
    ALASSERT(bright<B->width());
    int a_l = aleft;
    int a_w = aright-aleft+1;
    int b_l = bleft;
    int b_w = bright-bleft+1;

    int w = a_w + b_w;
    int h = A->height();
    ALFloatMatrix* result = ALFloatMatrix::create(w, h);
    for (size_t y=0; y<h; ++y)
    {
        auto r = result->vGetAddr(y);
        auto a = A->vGetAddr(y)+a_l;
        auto b = B->vGetAddr(y)+b_l;
        ::memcpy(r, a, a_w*sizeof(ALFLOAT));
        ::memcpy(r+a_w, b, b_w*sizeof(ALFLOAT));
    }
    return result;
}
ALFloatMatrix* ALPackageMatrixCrop(ALFloatMatrix* A, double aleft, double aright)
{
    ALASSERT(NULL!=A);
    ALASSERT(aleft>=0 && aleft < A->width());
    int ar = aright;
    int al = aleft;
    int w = A->width();
    int h = A->height();
    if (0 > ar)
    {
        ar = w-1;
    }
    ALASSERT(ar>=al && ar < A->width());
    int rw = ar-al+1;
    ALFloatMatrix* result = ALFloatMatrix::create(rw, h);
    for (int y=0; y<h; ++y)
    {
        auto dst = result->vGetAddr(y);
        auto src = A->vGetAddr(y);
        ::memcpy(dst, src+al, sizeof(ALFLOAT)*rw);
    }
    return result;
}

ALClassifier* ALPackageRandomForest(ALFloatMatrix* m)
{
    ALASSERT(NULL!=m);
    ALASSERT(m->width() > 1);
    ALSp<ALClassifierCreator> l = new ALRandomForestMatrix;
    ALSp<ALFloatMatrix> X = ALFloatMatrix::createCropVirtualMatrix(m, 1, 0, m->width()-1, m->height()-1);
    ALSp<ALFloatMatrix> Y = ALFloatMatrix::createCropVirtualMatrix(m, 0, 0, 0, m->height()-1);
    return l->vLearn(X.get(), Y.get());
}
ALIChainLearner* ALPackageCreateCGP(ALCGPParameter* p)
{
    ALCGPRegressor* r = new ALCGPRegressor(p->nW, p->nH);
    r->map(p->pValues, p->nW*p->nH*2);
    return r;
}

ALClassifier* ALPackageC45Tree(ALFloatMatrix* m, ALDecisionTreeParameter* p)
{
    ALSp<ALISuperviseLearner> learner = new ALDecisionTree(p->maxDepth);
    ALSp<ALFloatMatrix> X = ALFloatMatrix::createCropVirtualMatrix(m, 1, 0, m->width()-1, m->height()-1);
    ALSp<ALFloatMatrix> Y = ALFloatMatrix::createCropVirtualMatrix(m, 0, 0, 0, m->height()-1);
    return learner->vLearn(X.get(), Y.get());
}
ALClassifier* ALPackageNBM(ALFloatMatrix* m)
{
    ALSp<ALISuperviseLearner> learner = new ALNaiveBayesianLearner;
    ALSp<ALFloatMatrix> X = ALFloatMatrix::createCropVirtualMatrix(m, 1, 0, m->width()-1, m->height()-1);
    ALSp<ALFloatMatrix> Y = ALFloatMatrix::createCropVirtualMatrix(m, 0, 0, 0, m->height()-1);
    return learner->vLearn(X.get(), Y.get());
}

ALClassifierSet* ALPackageC45TreeUnit(ALFloatMatrix* m, ALDecisionTreeParameter* p)
{
    ALSp<ALIMatrixPredictor> pre = ALPackageC45Tree(m, p);
    ALClassifierSet* set = new ALClassifierSet;
    set->push(pre, m->height());
    return set;
}
ALClassifierSet* ALPackageMergeClassifierSet(ALClassifierSet* A, ALClassifierSet* B)
{
    return ALClassifierSet::merge(A, B);
}

ALClassifier* ALPackageComposeClassifierSet(ALClassifierSet* A)
{
    return new ALComposeClassifier(A);
}

ALFloatMatrix* ALPackageGDCompute(ALFloatMatrix* X, ALGradientMethod* decent, ALFloatMatrix* P)
{
    ALFloatMatrix* PD = ALFloatMatrix::create(P->width(), P->height());
    ALFloatMatrix::copy(PD, P);
    //TODO
    ALSp<ALCNNLearner> learner = new ALCNNLearner((cJSON*)decent->other);
    ALGradientMethod* newGd = learner->getGDMethod();
    newGd->gd->vOptimize(PD, X, newGd->det.get(), decent->alpha, decent->iteration);
    ALFloatMatrix::linear(PD, PD, 1.0, P, -1.0);
    delete newGd;
    return PD;
}
ALFloatMatrix* ALPackageMatrixPlus(ALFloatMatrix* X1, ALFloatMatrix* X2)
{
    ALASSERT(X1->width() == X2->width());
    ALASSERT(X1->height() == X2->height());
    auto res = ALFloatMatrix::create(X1->width(), X1->height());
    ALFloatMatrix::linear(res, X1, 1.0, X2, 1.0);
    return res;
}
ALFloatMatrix* ALPackageParameterInit(ALGradientMethod* decent)
{
    auto size = decent->det->vInitParameters(NULL);
    ALFloatMatrix* c = ALFloatMatrix::create(size, 1);
    decent->det->vInitParameters(c);
    return c;
}
ALFloatMatrix* ALPackageGDMatrixPrepare(ALFloatMatrix* X, ALFloatMatrix* Y, ALGradientMethod* grad)
{
    ALSp<ALFloatMatrix> YT = ALFloatMatrix::transpose(Y);
    ALSp<ALFloatMatrix> Y_E = ALFloatMatrix::create(grad->typeNumber, YT->width());
    ALFloatMatrix::typeExpand(Y_E.get(), YT.get());
    return ALFloatMatrix::unionHorizontal(Y_E.get(), X);
}
ALFloatMatrix* ALPackageMatrixPlusM(ALFloatMatrix* X1, ALFloatMatrix* X2)
{
    ALASSERT(X1->width() == X2->width());
    ALASSERT(X1->height() == X2->height());
    auto res = ALFloatMatrix::create(X1->width(), X1->height());
    ALFloatMatrix::linear(res, X1, 1.0, X2, 0.166667);
    return res;
}
