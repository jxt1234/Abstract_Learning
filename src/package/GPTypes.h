#ifndef SRC_PACKAGE_GPTYPES_H
#define SRC_PACKAGE_GPTYPES_H
#include "lowlevelAPI/IStatusType.h"
extern IStatusType* gALFloatMatrix;
extern IStatusType* gALClassifierSet;
extern IStatusType* gALDividerParameter;
extern IStatusType* gALGradientMethod;
extern IStatusType* gdouble;
extern IStatusType* gALDecisionTreeParameter;
extern IStatusType* gALIUnSuperLearner;
extern IStatusType* gALSVMParameter;
extern IStatusType* gALCGPParameter;
extern IStatusType* gALARStructure;
extern IStatusType* gALFloatPredictor;
extern IStatusType* gALClassifierCreator;
extern IStatusType* gALClassifier;
extern IStatusType* gALFloatDataChain;
extern IStatusType* gALIMatrixPredictor;
extern IStatusType* gALIChainLearner;
extern IStatusType* gALISuperviseLearner;
extern IStatusType* gALLabeldData;
extern "C"{
IStatusType* libAbstract_learning_GP_IStatusType_Create(const std::string& name);
}
#endif
