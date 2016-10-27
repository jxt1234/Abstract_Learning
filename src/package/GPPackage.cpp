#include "GPPackage.h"
#include "GPTypes.h"
#include <assert.h>
#include "package/ALPackage.h"
GPContents* ALPackageCreateClassify_GPpackage(GPContents* inputs)
{
assert(2 == inputs->size());
assert(inputs->getType(0)->name() == gALClassifierCreator->name());
assert(inputs->getType(1)->name() == gALFloatMatrix->name());
GPContents* out =  new GPContents;
ALClassifierCreator* X0 = (ALClassifierCreator*)inputs->get(0);
ALFloatMatrix* X1 = (ALFloatMatrix*)inputs->get(1);
ALClassifier* result = ALPackageCreateClassify(X0, X1);
out->push(result,gALClassifier);
return out;
}
GPContents* ALPackageRandomForest_GPpackage(GPContents* inputs)
{
assert(1 == inputs->size());
assert(inputs->getType(0)->name() == gALFloatMatrix->name());
GPContents* out =  new GPContents;
ALFloatMatrix* X0 = (ALFloatMatrix*)inputs->get(0);
ALClassifier* result = ALPackageRandomForest(X0);
out->push(result,gALClassifier);
return out;
}
GPContents* ALPackageClassify_GPpackage(GPContents* inputs)
{
assert(2 == inputs->size());
assert(inputs->getType(0)->name() == gALClassifier->name());
assert(inputs->getType(1)->name() == gALFloatMatrix->name());
GPContents* out =  new GPContents;
ALClassifier* X0 = (ALClassifier*)inputs->get(0);
ALFloatMatrix* X1 = (ALFloatMatrix*)inputs->get(1);
ALFloatMatrix* result = ALPackageClassify(X0, X1);
out->push(result,gALFloatMatrix);
return out;
}
GPContents* ALPackageClassifyProb_GPpackage(GPContents* inputs)
{
assert(2 == inputs->size());
assert(inputs->getType(0)->name() == gALClassifier->name());
assert(inputs->getType(1)->name() == gALFloatMatrix->name());
GPContents* out =  new GPContents;
ALClassifier* X0 = (ALClassifier*)inputs->get(0);
ALFloatMatrix* X1 = (ALFloatMatrix*)inputs->get(1);
ALFloatMatrix* result = ALPackageClassifyProb(X0, X1);
out->push(result,gALFloatMatrix);
return out;
}
GPContents* ALPackageClassifyProbValues_GPpackage(GPContents* inputs)
{
assert(1 == inputs->size());
assert(inputs->getType(0)->name() == gALClassifier->name());
GPContents* out =  new GPContents;
ALClassifier* X0 = (ALClassifier*)inputs->get(0);
ALFloatMatrix* result = ALPackageClassifyProbValues(X0);
out->push(result,gALFloatMatrix);
return out;
}
GPContents* ALPackageCrossValidateClassify_GPpackage(GPContents* inputs)
{
assert(2 == inputs->size());
assert(inputs->getType(0)->name() == gALClassifierCreator->name());
assert(inputs->getType(1)->name() == gALFloatMatrix->name());
GPContents* out =  new GPContents;
ALClassifierCreator* X0 = (ALClassifierCreator*)inputs->get(0);
ALFloatMatrix* X1 = (ALFloatMatrix*)inputs->get(1);
double* result = new double;
*result = ALPackageCrossValidateClassify(X0, X1);
out->push(result,gdouble);
return out;
}
GPContents* ALPackageC45Tree_GPpackage(GPContents* inputs)
{
assert(2 == inputs->size());
assert(inputs->getType(0)->name() == gALFloatMatrix->name());
assert(inputs->getType(1) == gALDecisionTreeParameter);
GPContents* out =  new GPContents;
ALFloatMatrix* X0 = (ALFloatMatrix*)inputs->get(0);
ALDecisionTreeParameter* X1 = (ALDecisionTreeParameter*)inputs->get(1);
ALClassifier* result = ALPackageC45Tree(X0, X1);
out->push(result,gALClassifier);
return out;
}
GPContents* ALPackageNBM_GPpackage(GPContents* inputs)
{
assert(1 == inputs->size());
assert(inputs->getType(0)->name() == gALFloatMatrix->name());
GPContents* out =  new GPContents;
ALFloatMatrix* X0 = (ALFloatMatrix*)inputs->get(0);
ALClassifier* result = ALPackageNBM(X0);
out->push(result,gALClassifier);
return out;
}
GPContents* ALPackageC45TreeUnit_GPpackage(GPContents* inputs)
{
assert(2 == inputs->size());
assert(inputs->getType(0)->name() == gALFloatMatrix->name());
assert(inputs->getType(1) == gALDecisionTreeParameter);
GPContents* out =  new GPContents;
ALFloatMatrix* X0 = (ALFloatMatrix*)inputs->get(0);
ALDecisionTreeParameter* X1 = (ALDecisionTreeParameter*)inputs->get(1);
ALClassifierSet* result = ALPackageC45TreeUnit(X0, X1);
out->push(result,gALClassifierSet);
return out;
}
GPContents* ALPackageMergeClassifierSet_GPpackage(GPContents* inputs)
{
assert(2 == inputs->size());
assert(inputs->getType(0)->name() == gALClassifierSet->name());
assert(inputs->getType(1)->name() == gALClassifierSet->name());
GPContents* out =  new GPContents;
ALClassifierSet* X0 = (ALClassifierSet*)inputs->get(0);
ALClassifierSet* X1 = (ALClassifierSet*)inputs->get(1);
ALClassifierSet* result = ALPackageMergeClassifierSet(X0, X1);
out->push(result,gALClassifierSet);
return out;
}
GPContents* ALPackageComposeClassifierSet_GPpackage(GPContents* inputs)
{
assert(1 == inputs->size());
assert(inputs->getType(0)->name() == gALClassifierSet->name());
GPContents* out =  new GPContents;
ALClassifierSet* X0 = (ALClassifierSet*)inputs->get(0);
ALClassifier* result = ALPackageComposeClassifierSet(X0);
out->push(result,gALClassifier);
return out;
}
GPContents* ALPackageValidateMatrix_GPpackage(GPContents* inputs)
{
assert(2 == inputs->size());
assert(inputs->getType(0)->name() == gALIMatrixPredictor->name());
assert(inputs->getType(1)->name() == gALFloatMatrix->name());
GPContents* out =  new GPContents;
ALIMatrixPredictor* X0 = (ALIMatrixPredictor*)inputs->get(0);
ALFloatMatrix* X1 = (ALFloatMatrix*)inputs->get(1);
ALFloatMatrix* result = ALPackageValidateMatrix(X0, X1);
out->push(result,gALFloatMatrix);
return out;
}
GPContents* ALPackageValidateChain_GPpackage(GPContents* inputs)
{
assert(2 == inputs->size());
assert(inputs->getType(0)->name() == gALFloatPredictor->name());
assert(inputs->getType(1)->name() == gALLabeldData->name());
GPContents* out =  new GPContents;
ALFloatPredictor* X0 = (ALFloatPredictor*)inputs->get(0);
ALLabeldData* X1 = (ALLabeldData*)inputs->get(1);
ALFloatMatrix* result = ALPackageValidateChain(X0, X1);
out->push(result,gALFloatMatrix);
return out;
}
GPContents* ALPackageSuperLearning_GPpackage(GPContents* inputs)
{
assert(2 == inputs->size());
assert(inputs->getType(0)->name() == gALISuperviseLearner->name());
assert(inputs->getType(1)->name() == gALFloatMatrix->name());
GPContents* out =  new GPContents;
ALISuperviseLearner* X0 = (ALISuperviseLearner*)inputs->get(0);
ALFloatMatrix* X1 = (ALFloatMatrix*)inputs->get(1);
ALIMatrixPredictor* result = ALPackageSuperLearning(X0, X1);
out->push(result,gALIMatrixPredictor);
return out;
}
GPContents* ALPackageUnSuperLearning_GPpackage(GPContents* inputs)
{
assert(2 == inputs->size());
assert(inputs->getType(0)->name() == gALIUnSuperLearner->name());
assert(inputs->getType(1)->name() == gALFloatMatrix->name());
GPContents* out =  new GPContents;
ALIUnSuperLearner* X0 = (ALIUnSuperLearner*)inputs->get(0);
ALFloatMatrix* X1 = (ALFloatMatrix*)inputs->get(1);
ALIMatrixPredictor* result = ALPackageUnSuperLearning(X0, X1);
out->push(result,gALIMatrixPredictor);
return out;
}
GPContents* ALPackageLearn_GPpackage(GPContents* inputs)
{
assert(2 == inputs->size());
assert(inputs->getType(0)->name() == gALIChainLearner->name());
assert(inputs->getType(1)->name() == gALLabeldData->name());
GPContents* out =  new GPContents;
ALIChainLearner* X0 = (ALIChainLearner*)inputs->get(0);
ALLabeldData* X1 = (ALLabeldData*)inputs->get(1);
ALFloatPredictor* result = ALPackageLearn(X0, X1);
out->push(result,gALFloatPredictor);
return out;
}
GPContents* ALPackageCreateDivider_GPpackage(GPContents* inputs)
{
assert(3 == inputs->size());
assert(inputs->getType(0)->name() == gALIChainLearner->name());
assert(inputs->getType(1)->name() == gALIChainLearner->name());
assert(inputs->getType(2) == gALDividerParameter);
GPContents* out =  new GPContents;
ALIChainLearner* X0 = (ALIChainLearner*)inputs->get(0);
ALIChainLearner* X1 = (ALIChainLearner*)inputs->get(1);
ALDividerParameter* X2 = (ALDividerParameter*)inputs->get(2);
ALIChainLearner* result = ALPackageCreateDivider(X0, X1, X2);
out->push(result,gALIChainLearner);
return out;
}
GPContents* ALPackageCreateCGP_GPpackage(GPContents* inputs)
{
assert(1 == inputs->size());
assert(inputs->getType(0) == gALCGPParameter);
GPContents* out =  new GPContents;
ALCGPParameter* X0 = (ALCGPParameter*)inputs->get(0);
ALIChainLearner* result = ALPackageCreateCGP(X0);
out->push(result,gALIChainLearner);
return out;
}
GPContents* ALPackageCreateRegress_GPpackage(GPContents* inputs)
{
assert(0 == inputs->size());
GPContents* out =  new GPContents;
ALISuperviseLearner* result = ALPackageCreateRegress();
out->push(result,gALISuperviseLearner);
return out;
}
GPContents* ALPackageLabled_GPpackage(GPContents* inputs)
{
assert(2 == inputs->size());
assert(inputs->getType(0)->name() == gALFloatDataChain->name());
assert(inputs->getType(1)->name() == gdouble->name());
GPContents* out =  new GPContents;
ALFloatDataChain* X0 = (ALFloatDataChain*)inputs->get(0);
double X1 = *(double*)inputs->get(1);
ALLabeldData* result = ALPackageLabled(X0, X1);
out->push(result,gALLabeldData);
return out;
}
GPContents* ALPackageCrossValidate_GPpackage(GPContents* inputs)
{
assert(2 == inputs->size());
assert(inputs->getType(0)->name() == gALIChainLearner->name());
assert(inputs->getType(1)->name() == gALLabeldData->name());
GPContents* out =  new GPContents;
ALIChainLearner* X0 = (ALIChainLearner*)inputs->get(0);
ALLabeldData* X1 = (ALLabeldData*)inputs->get(1);
double* result = new double;
*result = ALPackageCrossValidate(X0, X1);
out->push(result,gdouble);
return out;
}
GPContents* ALPackageCombine_GPpackage(GPContents* inputs)
{
assert(2 == inputs->size());
assert(inputs->getType(0)->name() == gALISuperviseLearner->name());
assert(inputs->getType(1) == gALARStructure);
GPContents* out =  new GPContents;
ALISuperviseLearner* X1 = (ALISuperviseLearner*)inputs->get(0);
ALARStructure* X0 = (ALARStructure*)inputs->get(1);
ALIChainLearner* result = ALPackageCombine(X0, X1);
out->push(result,gALIChainLearner);
return out;
}
GPContents* ALPackageCreateSVM_GPpackage(GPContents* inputs)
{
assert(1 == inputs->size());
assert(inputs->getType(0) == gALSVMParameter);
GPContents* out =  new GPContents;
ALSVMParameter* X0 = (ALSVMParameter*)inputs->get(0);
ALClassifierCreator* result = ALPackageCreateSVM(X0);
out->push(result,gALClassifierCreator);
return out;
}
GPContents* ALPackageCreateGMM_GPpackage(GPContents* inputs)
{
assert(0 == inputs->size());
GPContents* out =  new GPContents;
ALClassifierCreator* result = ALPackageCreateGMM();
out->push(result,gALClassifierCreator);
return out;
}
GPContents* ALPackageCreateLogicalRegress_GPpackage(GPContents* inputs)
{
assert(0 == inputs->size());
GPContents* out =  new GPContents;
ALClassifierCreator* result = ALPackageCreateLogicalRegress();
out->push(result,gALClassifierCreator);
return out;
}
GPContents* ALPackageCreateDecisionTree_GPpackage(GPContents* inputs)
{
assert(1 == inputs->size());
assert(inputs->getType(0) == gALDecisionTreeParameter);
GPContents* out =  new GPContents;
ALDecisionTreeParameter* X0 = (ALDecisionTreeParameter*)inputs->get(0);
ALClassifierCreator* result = ALPackageCreateDecisionTree(X0);
out->push(result,gALClassifierCreator);
return out;
}
GPContents* ALPackageMatrixMerge_GPpackage(GPContents* inputs)
{
assert(6 == inputs->size());
assert(inputs->getType(0)->name() == gALFloatMatrix->name());
assert(inputs->getType(1)->name() == gALFloatMatrix->name());
assert(inputs->getType(2)->name() == gdouble->name());
assert(inputs->getType(3)->name() == gdouble->name());
assert(inputs->getType(4)->name() == gdouble->name());
assert(inputs->getType(5)->name() == gdouble->name());
GPContents* out =  new GPContents;
ALFloatMatrix* X0 = (ALFloatMatrix*)inputs->get(0);
ALFloatMatrix* X1 = (ALFloatMatrix*)inputs->get(1);
double X2 = *(double*)inputs->get(2);
double X3 = *(double*)inputs->get(3);
double X4 = *(double*)inputs->get(4);
double X5 = *(double*)inputs->get(5);
ALFloatMatrix* result = ALPackageMatrixMerge(X0, X1, X2, X3, X4, X5);
out->push(result,gALFloatMatrix);
return out;
}
GPContents* ALPackageMatrixCrop_GPpackage(GPContents* inputs)
{
assert(3 == inputs->size());
assert(inputs->getType(0)->name() == gALFloatMatrix->name());
assert(inputs->getType(1)->name() == gdouble->name());
assert(inputs->getType(2)->name() == gdouble->name());
GPContents* out =  new GPContents;
ALFloatMatrix* X0 = (ALFloatMatrix*)inputs->get(0);
double X1 = *(double*)inputs->get(1);
double X2 = *(double*)inputs->get(2);
ALFloatMatrix* result = ALPackageMatrixCrop(X0, X1, X2);
out->push(result,gALFloatMatrix);
return out;
}
GPContents* ALPackageGDCompute_GPpackage(GPContents* inputs)
{
assert(3 == inputs->size());
assert(inputs->getType(0)->name() == gALFloatMatrix->name());
assert(inputs->getType(1)->name() == gALGradientMethod->name());
assert(inputs->getType(2)->name() == gALFloatMatrix->name());
GPContents* out =  new GPContents;
ALFloatMatrix* X0 = (ALFloatMatrix*)inputs->get(0);
ALGradientMethod* X1 = (ALGradientMethod*)inputs->get(1);
ALFloatMatrix* X2 = (ALFloatMatrix*)inputs->get(2);
ALFloatMatrix* result = ALPackageGDCompute(X0, X1, X2);
out->push(result,gALFloatMatrix);
return out;
}
GPContents* ALPackageMatrixPlus_GPpackage(GPContents* inputs)
{
assert(2 == inputs->size());
assert(inputs->getType(0)->name() == gALFloatMatrix->name());
assert(inputs->getType(1)->name() == gALFloatMatrix->name());
GPContents* out =  new GPContents;
ALFloatMatrix* X0 = (ALFloatMatrix*)inputs->get(0);
ALFloatMatrix* X1 = (ALFloatMatrix*)inputs->get(1);
ALFloatMatrix* result = ALPackageMatrixPlus(X0, X1);
out->push(result,gALFloatMatrix);
return out;
}
GPContents* ALPackageMatrixPlusM_GPpackage(GPContents* inputs)
{
assert(2 == inputs->size());
assert(inputs->getType(0)->name() == gALFloatMatrix->name());
assert(inputs->getType(1)->name() == gALFloatMatrix->name());
GPContents* out =  new GPContents;
ALFloatMatrix* X0 = (ALFloatMatrix*)inputs->get(0);
ALFloatMatrix* X1 = (ALFloatMatrix*)inputs->get(1);
ALFloatMatrix* result = ALPackageMatrixPlusM(X0, X1);
out->push(result,gALFloatMatrix);
return out;
}
GPContents* ALPackageGDMatrixPrepare_GPpackage(GPContents* inputs)
{
assert(3 == inputs->size());
assert(inputs->getType(0)->name() == gALFloatMatrix->name());
assert(inputs->getType(1)->name() == gALFloatMatrix->name());
assert(inputs->getType(2)->name() == gALGradientMethod->name());
GPContents* out =  new GPContents;
ALFloatMatrix* X0 = (ALFloatMatrix*)inputs->get(0);
ALFloatMatrix* X1 = (ALFloatMatrix*)inputs->get(1);
ALGradientMethod* X2 = (ALGradientMethod*)inputs->get(2);
ALFloatMatrix* result = ALPackageGDMatrixPrepare(X0, X1, X2);
out->push(result,gALFloatMatrix);
return out;
}
GPContents* ALPackageParameterInit_GPpackage(GPContents* inputs)
{
assert(1 == inputs->size());
assert(inputs->getType(0)->name() == gALGradientMethod->name());
GPContents* out =  new GPContents;
ALGradientMethod* X0 = (ALGradientMethod*)inputs->get(0);
ALFloatMatrix* result = ALPackageParameterInit(X0);
out->push(result,gALFloatMatrix);
return out;
}
GPContents* ALPackageGDPredictorLoad_GPpackage(GPContents* inputs)
{
assert(2 == inputs->size());
assert(inputs->getType(0)->name() == gALGradientMethod->name());
assert(inputs->getType(1)->name() == gALFloatMatrix->name());
GPContents* out =  new GPContents;
ALGradientMethod* X0 = (ALGradientMethod*)inputs->get(0);
ALFloatMatrix* X1 = (ALFloatMatrix*)inputs->get(1);
ALClassifier* result = ALPackageGDPredictorLoad(X0, X1);
out->push(result,gALClassifier);
return out;
}
GPContents* ALPackageMatrixLinear_GPpackage(GPContents* inputs)
{
assert(3 == inputs->size());
assert(inputs->getType(0)->name() == gALFloatMatrix->name());
assert(inputs->getType(1)->name() == gdouble->name());
assert(inputs->getType(2)->name() == gdouble->name());
GPContents* out =  new GPContents;
ALFloatMatrix* X0 = (ALFloatMatrix*)inputs->get(0);
double X1 = *(double*)inputs->get(1);
double X2 = *(double*)inputs->get(2);
ALFloatMatrix* result = ALPackageMatrixLinear(X0, X1, X2);
out->push(result,gALFloatMatrix);
return out;
}
