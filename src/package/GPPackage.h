#ifndef SRC_PACKAGE_GPFUNCTIONS_H
#define SRC_PACKAGE_GPFUNCTIONS_H
#include "lowlevelAPI/GPContents.h"
extern "C"{
GPContents* ALPackageCreateClassify_GPpackage(GPContents* inputs);
GPContents* ALPackageRandomForest_GPpackage(GPContents* inputs);
GPContents* ALPackageClassify_GPpackage(GPContents* inputs);
GPContents* ALPackageClassifyProb_GPpackage(GPContents* inputs);
GPContents* ALPackageClassifyProbValues_GPpackage(GPContents* inputs);
GPContents* ALPackageCrossValidateClassify_GPpackage(GPContents* inputs);
GPContents* ALPackageC45Tree_GPpackage(GPContents* inputs);
GPContents* ALPackageNBM_GPpackage(GPContents* inputs);
GPContents* ALPackageC45TreeUnit_GPpackage(GPContents* inputs);
GPContents* ALPackageMergeClassifierSet_GPpackage(GPContents* inputs);
GPContents* ALPackageComposeClassifierSet_GPpackage(GPContents* inputs);
GPContents* ALPackageValidateMatrix_GPpackage(GPContents* inputs);
GPContents* ALPackageValidateChain_GPpackage(GPContents* inputs);
GPContents* ALPackageSuperLearning_GPpackage(GPContents* inputs);
GPContents* ALPackageUnSuperLearning_GPpackage(GPContents* inputs);
GPContents* ALPackageLearn_GPpackage(GPContents* inputs);
GPContents* ALPackageCreateDivider_GPpackage(GPContents* inputs);
GPContents* ALPackageCreateCGP_GPpackage(GPContents* inputs);
GPContents* ALPackageCreateRegress_GPpackage(GPContents* inputs);
GPContents* ALPackageLabled_GPpackage(GPContents* inputs);
GPContents* ALPackageCrossValidate_GPpackage(GPContents* inputs);
GPContents* ALPackageCombine_GPpackage(GPContents* inputs);
GPContents* ALPackageCreateSVM_GPpackage(GPContents* inputs);
GPContents* ALPackageCreateGMM_GPpackage(GPContents* inputs);
GPContents* ALPackageCreateLogicalRegress_GPpackage(GPContents* inputs);
GPContents* ALPackageCreateDecisionTree_GPpackage(GPContents* inputs);
GPContents* ALPackageMatrixMerge_GPpackage(GPContents* inputs);
GPContents* ALPackageMatrixCrop_GPpackage(GPContents* inputs);
}
#endif
