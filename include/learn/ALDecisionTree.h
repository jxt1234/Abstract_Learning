#ifndef INCLUDE_LEARN_ALDECISIONTREE_H
#define INCLUDE_LEARN_ALDECISIONTREE_H
#include <ostream>
#include "ALISuperviseLearner.h"
#include "core/ALIExpander.h"
#include "math/ALFloatMatrix.h"
class ALDecisionTree:public ALISuperviseLearner
{
public:
    ALDecisionTree(size_t maxDepth=10, size_t divideSize=2);
    virtual ~ALDecisionTree();
    void setFixTypes(ALSp<ALFloatMatrix> types);
    virtual ALIMatrixPredictor* vLearn(const ALFloatMatrix* X, const ALFloatMatrix* Y) const override;

    
    ALIMatrixPredictor* learnT(const ALFloatMatrix* X, const ALFloatMatrix* YT) const;
private:
    class Tree:public ALRefCount
    {
    public:
        class Node;
        Tree(Node* r);
        ~Tree();
        ALFLOAT predict(const ALFLOAT* X);
        void predictProbability(const ALFLOAT* X, ALFLOAT* Y);
        void print(std::ostream& out);
        
        inline const Node* root() const {return mRoot;}
    private:
        Node* mRoot;
    };
    Tree* train(const ALFloatMatrix* X, const ALFloatMatrix* YT, const ALFloatMatrix* types) const;
    Tree::Node* _train(const ALFloatMatrix* X, const ALFloatMatrix* YT, size_t depth, const ALFloatMatrix* types) const;
    size_t mMaxDepth;
    ALSp<ALFloatMatrix> mFixTypes;
    size_t mDivideSize;
};
#endif
