#ifndef INCLUDE_PACKAGE_ALPACKAGE_H
#define INCLUDE_PACKAGE_ALPACKAGE_H
#include "learn/ALRegressor.h"
#include "learn/ALDivider.h"
#include "learn/ALLearnFactory.h"
#include "loader/ALStandardLoader.h"
#include "compose/ALClassifierSet.h"
#include "compose/ALComposeClassifier.h"
#include "core/ALLabeldMethodFactory.h"
#include "learn/ALSVMLearner.h"

class ALGradientMethod;

class ALDividerParameter:public ALRefCount
{
public:
    float per;
    int step;
};
class ALDecisionTreeParameter:public ALRefCount
{
public:
//    size_t minGroup;
//    ALFLOAT targetPrune;
    size_t maxDepth;
};
class ALCGPParameter:public ALRefCount
{
public:
    ALCGPParameter(int w, int h)
    {
        pValues = new double[w*h*2];
        nW = w;
        nH = h;
    }
    virtual ~ ALCGPParameter()
    {
        delete [] pValues;
    }
    double* pValues;
    int nW;
    int nH;
};

typedef ALISuperviseLearner ALClassifierCreator;
typedef ALIMatrixPredictor ALClassifier;

/*GP FUNCTION*/ALClassifier* ALPackageCreateClassify(ALClassifierCreator* l, ALFloatMatrix* m);
/*GP FUNCTION*/ALClassifier* ALPackageRandomForest(ALFloatMatrix* m);
/*GP FUNCTION*/ALFloatMatrix* ALPackageClassify(ALClassifier* l, ALFloatMatrix* m);
/*GP FUNCTION*/ALFloatMatrix* ALPackageClassifyProb(ALClassifier* l, ALFloatMatrix* m);
/*GP FUNCTION*/ALFloatMatrix* ALPackageClassifyProbValues(ALClassifier* l);
/*GP FUNCTION*/double ALPackageCrossValidateClassify(ALClassifierCreator* l, ALFloatMatrix* c);
/*GP FUNCTION*/ALClassifier* ALPackageC45Tree(ALFloatMatrix* m, ALDecisionTreeParameter* p/*S*/);
/*GP FUNCTION*/ALClassifier* ALPackageNBM(ALFloatMatrix* m);


/*GP FUNCTION*/ALClassifierSet* ALPackageC45TreeUnit(ALFloatMatrix* m, ALDecisionTreeParameter* p/*S*/);
/*GP FUNCTION*/ALClassifierSet* ALPackageMergeClassifierSet(ALClassifierSet* A, ALClassifierSet* B);
/*GP FUNCTION*/ALClassifier* ALPackageComposeClassifierSet(ALClassifierSet* A);



/*GP FUNCTION*/ALFloatMatrix* ALPackageValidateMatrix(ALIMatrixPredictor* l, ALFloatMatrix* m);
/*GP FUNCTION*/ALFloatMatrix* ALPackageValidateChain(ALFloatPredictor* l, ALLabeldData* c);
/*GP FUNCTION*/ALIMatrixPredictor* ALPackageSuperLearning(ALISuperviseLearner* l, ALFloatMatrix* m);
/*GP FUNCTION*/ALIMatrixPredictor* ALPackageUnSuperLearning(ALIUnSuperLearner* l, ALFloatMatrix* m);
/*GP FUNCTION*/ALFloatPredictor* ALPackageLearn(ALIChainLearner* l, ALLabeldData* d);
/*GP FUNCTION*/ALIChainLearner* ALPackageCreateDivider(ALIChainLearner* l, ALIChainLearner* r, ALDividerParameter* p/*S*/);
/*GP FUNCTION*/ALIChainLearner* ALPackageCreateCGP(ALCGPParameter* p/*S*/);
/*GP FUNCTION*/ALISuperviseLearner* ALPackageCreateRegress();
/*GP FUNCTION*/ALLabeldData* ALPackageLabled(ALFloatDataChain* c, double delay);
/*GP FUNCTION*/double ALPackageCrossValidate(ALIChainLearner* l, ALLabeldData* c);
/*GP FUNCTION*/ALIChainLearner* ALPackageCombine(ALARStructure* ar/*S*/, ALISuperviseLearner* l);
/*GP FUNCTION*/ALClassifierCreator* ALPackageCreateSVM(ALSVMParameter* p/*S*/);
/*GP FUNCTION*/ALClassifierCreator* ALPackageCreateGMM();
/*GP FUNCTION*/ALClassifierCreator* ALPackageCreateLogicalRegress();
/*GP FUNCTION*/ALClassifierCreator* ALPackageCreateDecisionTree(ALDecisionTreeParameter* p/*S*/);
/*GP FUNCTION*/ALFloatMatrix* ALPackageMatrixMerge(ALFloatMatrix* A, ALFloatMatrix* B, double aleft, double aright, double bleft, double bright);
/*GP FUNCTION*/ALFloatMatrix* ALPackageMatrixCrop(ALFloatMatrix* A, double aleft, double aright);
/*GP FUNCTION*/ALFloatMatrix* ALPackageGDCompute(ALFloatMatrix* X, ALGradientMethod* decent, ALFloatMatrix* P);
/*GP FUNCTION*/ALFloatMatrix* ALPackageMatrixPlus(ALFloatMatrix* X1, ALFloatMatrix* X2);
/*GP FUNCTION*/ALFloatMatrix* ALPackageMatrixPlusM(ALFloatMatrix* X1, ALFloatMatrix* X2);
/*GP FUNCTION*/ALFloatMatrix* ALPackageGDMatrixPrepare(ALFloatMatrix* X, ALFloatMatrix* Y, ALGradientMethod* grad);
/*GP FUNCTION*/ALFloatMatrix* ALPackageParameterInit(ALGradientMethod* decent);

#endif
