#include "GPTypes.h"
#include "package/ALPackage.h"
#include "ALFloatMatrix_GPType.h"
IStatusType* gALFloatMatrix = new ALFloatMatrix_GPType();
#include "ALClassifierSet_GPType.h"
IStatusType* gALClassifierSet = new ALClassifierSet_GPType();
#include "ALDividerParameter_GPType.h"
IStatusType* gALDividerParameter = new ALDividerParameter_GPType();
IStatusType* gdouble = new GPDoubleType();
#include "ALDecisionTreeParameter_GPType.h"
IStatusType* gALDecisionTreeParameter = new ALDecisionTreeParameter_GPType();
#include "ALIUnSuperLearner_GPType.h"
IStatusType* gALIUnSuperLearner = new ALIUnSuperLearner_GPType();
#include "ALSVMParameter_GPType.h"
IStatusType* gALSVMParameter = new ALSVMParameter_GPType();
#include "ALCGPParameter_GPType.h"
IStatusType* gALCGPParameter = new ALCGPParameter_GPType();
#include "ALARStructure_GPType.h"
IStatusType* gALARStructure = new ALARStructure_GPType();
#include "ALFloatPredictor_GPType.h"
IStatusType* gALFloatPredictor = new ALFloatPredictor_GPType();
#include "ALClassifierCreator_GPType.h"
IStatusType* gALClassifierCreator = new ALClassifierCreator_GPType();
#include "ALClassifier_GPType.h"
IStatusType* gALClassifier = new ALClassifier_GPType();
#include "ALFloatDataChain_GPType.h"
IStatusType* gALFloatDataChain = new ALFloatDataChain_GPType();
#include "ALIMatrixPredictor_GPType.h"
IStatusType* gALIMatrixPredictor = new ALIMatrixPredictor_GPType();
#include "ALIChainLearner_GPType.h"
IStatusType* gALIChainLearner = new ALIChainLearner_GPType();
#include "ALISuperviseLearner_GPType.h"
IStatusType* gALISuperviseLearner = new ALISuperviseLearner_GPType();
#include "ALLabeldData_GPType.h"
IStatusType* gALLabeldData = new ALLabeldData_GPType();
IStatusType* libAbstract_learning_GP_IStatusType_Create(const std::string& name)
{
if (name == "ALFloatMatrix")
{
return gALFloatMatrix;
}
if (name == "ALClassifierSet")
{
return gALClassifierSet;
}
if (name == "ALDividerParameter")
{
return gALDividerParameter;
}
if (name == "double")
{
return gdouble;
}
if (name == "ALDecisionTreeParameter")
{
return gALDecisionTreeParameter;
}
if (name == "ALIUnSuperLearner")
{
return gALIUnSuperLearner;
}
if (name == "ALSVMParameter")
{
return gALSVMParameter;
}
if (name == "ALCGPParameter")
{
return gALCGPParameter;
}
if (name == "ALARStructure")
{
return gALARStructure;
}
if (name == "ALFloatPredictor")
{
return gALFloatPredictor;
}
if (name == "ALClassifierCreator")
{
return gALClassifierCreator;
}
if (name == "ALClassifier")
{
return gALClassifier;
}
if (name == "ALFloatDataChain")
{
return gALFloatDataChain;
}
if (name == "ALIMatrixPredictor")
{
return gALIMatrixPredictor;
}
if (name == "ALIChainLearner")
{
return gALIChainLearner;
}
if (name == "ALISuperviseLearner")
{
return gALISuperviseLearner;
}
if (name == "ALLabeldData")
{
return gALLabeldData;
}
return NULL;
}
