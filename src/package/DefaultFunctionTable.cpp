#include "package/DefaultFunctionTable.h"
#include "GPPackage.h"
#include "GPTypes.h"
void* DefaultFunctionTable::vGetFunction(const std::string& name)
{
if (name == "ALPackageCreateClassify_GPpackage")
{
return (void*)ALPackageCreateClassify_GPpackage;
}
if (name == "ALPackageRandomForest_GPpackage")
{
return (void*)ALPackageRandomForest_GPpackage;
}
if (name == "ALPackageClassify_GPpackage")
{
return (void*)ALPackageClassify_GPpackage;
}
if (name == "ALPackageClassifyProb_GPpackage")
{
return (void*)ALPackageClassifyProb_GPpackage;
}
if (name == "ALPackageClassifyProbValues_GPpackage")
{
return (void*)ALPackageClassifyProbValues_GPpackage;
}
if (name == "ALPackageCrossValidateClassify_GPpackage")
{
return (void*)ALPackageCrossValidateClassify_GPpackage;
}
if (name == "ALPackageC45Tree_GPpackage")
{
return (void*)ALPackageC45Tree_GPpackage;
}
if (name == "ALPackageNBM_GPpackage")
{
return (void*)ALPackageNBM_GPpackage;
}
if (name == "ALPackageC45TreeUnit_GPpackage")
{
return (void*)ALPackageC45TreeUnit_GPpackage;
}
if (name == "ALPackageMergeClassifierSet_GPpackage")
{
return (void*)ALPackageMergeClassifierSet_GPpackage;
}
if (name == "ALPackageComposeClassifierSet_GPpackage")
{
return (void*)ALPackageComposeClassifierSet_GPpackage;
}
if (name == "ALPackageValidateMatrix_GPpackage")
{
return (void*)ALPackageValidateMatrix_GPpackage;
}
if (name == "ALPackageValidateChain_GPpackage")
{
return (void*)ALPackageValidateChain_GPpackage;
}
if (name == "ALPackageSuperLearning_GPpackage")
{
return (void*)ALPackageSuperLearning_GPpackage;
}
if (name == "ALPackageUnSuperLearning_GPpackage")
{
return (void*)ALPackageUnSuperLearning_GPpackage;
}
if (name == "ALPackageLearn_GPpackage")
{
return (void*)ALPackageLearn_GPpackage;
}
if (name == "ALPackageCreateDivider_GPpackage")
{
return (void*)ALPackageCreateDivider_GPpackage;
}
if (name == "ALPackageCreateCGP_GPpackage")
{
return (void*)ALPackageCreateCGP_GPpackage;
}
if (name == "ALPackageCreateRegress_GPpackage")
{
return (void*)ALPackageCreateRegress_GPpackage;
}
if (name == "ALPackageLabled_GPpackage")
{
return (void*)ALPackageLabled_GPpackage;
}
if (name == "ALPackageCrossValidate_GPpackage")
{
return (void*)ALPackageCrossValidate_GPpackage;
}
if (name == "ALPackageCombine_GPpackage")
{
return (void*)ALPackageCombine_GPpackage;
}
if (name == "ALPackageCreateSVM_GPpackage")
{
return (void*)ALPackageCreateSVM_GPpackage;
}
if (name == "ALPackageCreateGMM_GPpackage")
{
return (void*)ALPackageCreateGMM_GPpackage;
}
if (name == "ALPackageCreateLogicalRegress_GPpackage")
{
return (void*)ALPackageCreateLogicalRegress_GPpackage;
}
if (name == "ALPackageCreateDecisionTree_GPpackage")
{
return (void*)ALPackageCreateDecisionTree_GPpackage;
}
if (name == "ALPackageMatrixMerge_GPpackage")
{
return (void*)ALPackageMatrixMerge_GPpackage;
}
if (name == "ALPackageMatrixCrop_GPpackage")
{
return (void*)ALPackageMatrixCrop_GPpackage;
}
if (name == "libAbstract_learning_GP_IStatusType_Create")
{
return (void*)libAbstract_learning_GP_IStatusType_Create;
}
return NULL;
}
