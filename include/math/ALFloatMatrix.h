#ifndef INCLUDE_CORE_ALFLOATMATRIX_H
#define INCLUDE_CORE_ALFLOATMATRIX_H
#include "ALHead.h"
#include <ostream>
/*Must assume the memory is continuous in row*/
class ALFloatMatrix:public ALRefCount
{
public:
    inline size_t width() const {return mWidth;}
    inline size_t height() const {return mHeight;}
    virtual ALFLOAT* vGetAddr(size_t y=0) const = 0;
    
    /*Functional API, Don't change inputs*/
    static ALFloatMatrix* create(size_t w, size_t h);
    static ALFloatMatrix* createRefMatrix(ALFLOAT* base, size_t w, size_t h);
    static ALFLOAT norm(const ALFloatMatrix* X);
    static ALFloatMatrix* createIdentity(size_t n);
    static ALFloatMatrix* createDiag(const ALFloatMatrix* X);
    static ALFloatMatrix* linear(const ALFloatMatrix* A, ALFLOAT a, const ALFloatMatrix* B, ALFLOAT b);
    static ALFloatMatrix* product(const ALFloatMatrix* A, const ALFloatMatrix* B);
    static ALFloatMatrix* productSS(const ALFloatMatrix* A, const ALFloatMatrix* B);//For AT=A and BT=B
    static ALFloatMatrix* productT(const ALFloatMatrix* A, const ALFloatMatrix* BT);
    
    /*If element is NAN, set as c*/
    static void checkAndSet(ALFloatMatrix* X, ALFLOAT c);

    static void product(ALFloatMatrix* C, const ALFloatMatrix* A, const ALFloatMatrix* B);
    static void productT(ALFloatMatrix* C, const ALFloatMatrix* A, const ALFloatMatrix* BT);
    
    static bool theSame(const ALFloatMatrix* X, const ALFloatMatrix* Y, ALFLOAT error=0.001);

    /*H = HT, n is the number of E, return
     [E,  * A *[E,
     H]         H]*/
    static ALFloatMatrix* HAH(const ALFloatMatrix* A, const ALFloatMatrix* H);
    
    static ALFloatMatrix* inverse(const ALFloatMatrix* A);
    static ALFloatMatrix* transpose(const ALFloatMatrix* A);
    static void transpose(const ALFloatMatrix* src, ALFloatMatrix* dst);

    static ALFloatMatrix* sts(const ALFloatMatrix* A, bool transpose = false);
    
    
    /*Merge two matrix into one:
     (Y, X) -> [Y, X]
     */
    static ALFloatMatrix* unionHorizontal(const ALFloatMatrix* Y, const ALFloatMatrix* X);
    
    /*
     X -> [E, 0
     [0, X]
     X->width() must be the same as X->height
     n must be larger than X->width()
     */
    static ALFloatMatrix* enlarge(size_t n, const ALFloatMatrix* X);
    
    /*These API may change the value of input pointer*/
    static void zero(ALFloatMatrix* X/*Output*/);
    static void copy(ALFloatMatrix* dst/*Output*/, const ALFloatMatrix* src);
    static void linear(ALFloatMatrix* C, const ALFloatMatrix* A, ALFLOAT a, const ALFloatMatrix* B, ALFLOAT b);

    /*x[i][j] = a*x[i][j]+b*/
    static void linearDirect(ALFloatMatrix* X, ALFLOAT a, ALFLOAT b);
    
    static void print(const ALFloatMatrix* A, std::ostream& os/*Output*/);
    static ALFLOAT inverse_basic(const ALFloatMatrix* A, ALFloatMatrix* dst);
    
    /*For basic save and load*/
    static void save(const ALFloatMatrix* m, ALWStream* f);
    static ALFloatMatrix* load(ALStream* f);
    /*For quick save and load*/
    static void quickSave(const ALFloatMatrix* m, ALWStream* f);
    static ALFloatMatrix* quickLoad(ALStream* f);
    static ALFloatMatrix* quickLoadLarge(ALStream* f);
    
    /*For Virtual Matrix*/
    static ALFloatMatrix* createCropVirtualMatrix(const ALFloatMatrix* base, size_t l, size_t t, size_t r, size_t b);
    static ALFloatMatrix* createIndexVirtualMatrix(ALFLOAT** indexes, size_t w, size_t h);
    static ALFloatMatrix* randomeSelectMatrix(const ALFloatMatrix* base, size_t height);
    
    /*General function*/
    static void productBasic(ALFLOAT* c, size_t c_stride, const ALFLOAT* a, size_t a_stride, const ALFLOAT* b, size_t b_stride, size_t w, size_t h, size_t k);
    static void productBasicT(ALFLOAT* c, size_t c_stride, const ALFLOAT* a, size_t a_stride, const ALFLOAT* b, size_t b_stride, size_t w, size_t h, size_t k);
    /*w, h is the size of src */
    static void transposeBasic(const ALFLOAT* src, size_t src_stride, ALFLOAT* dst, size_t dst_stride, size_t w, size_t h);
    
    
    /*For Discrete Matrix*/
    static ALFloatMatrix* genTypes(const ALFloatMatrix* Y);
    
    /*Return YT*/
    static ALFloatMatrix* getTypes(const ALFloatMatrix* YP, const ALFloatMatrix* prop);

    /*Turn classify sign to vector*/
    static void typeExpand(ALFloatMatrix* Y_Expand/*Output*/, const ALFloatMatrix* YT/*Input*/);

    
protected:
    ALFloatMatrix(size_t w, size_t h):mWidth(w), mHeight(h){}
    virtual ~ALFloatMatrix(){}
    size_t mWidth;
    size_t mHeight;
};

#endif
