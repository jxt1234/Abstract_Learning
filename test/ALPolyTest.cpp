#include "test/GPTest.h"
#include "math/ALPolynomial.h"
#include <iostream>
using namespace std;

static void testmain()
{
    ALFLOAT y[] = {64,48,44,25,10,1};
    ALFLOAT x[] = {4,3,1};
    ALSp<ALFloatMatrix> Y = ALPolynomial::construct(y, sizeof(y)/sizeof(ALFLOAT));
    ALSp<ALFloatMatrix> X = ALPolynomial::construct(x, sizeof(x)/sizeof(ALFLOAT));
    ALSp<ALFloatMatrix> Xp = ALPolynomial::divide(Y.get(), X.get());
    ALFloatMatrix::print(Xp.get(), cout);

    Xp = ALPolynomial::multi(Y.get(), X.get());
    ALFloatMatrix::print(Xp.get(), cout);

    Xp = ALPolynomial::det(Y.get());
    ALFloatMatrix::print(Xp.get(), cout);

    cout << ALPolynomial::compute(Y.get(), 2.3)<<endl;
    cout << ALPolynomial::compute(X.get(), 2.3)<<endl;
    cout << ALPolynomial::compute(Xp.get(), 2.3)<<endl;

    ALFLOAT root = ALPolynomial::NewTonSolve(Y.get());
    cout << root <<": "<<ALPolynomial::compute(Y.get(), root)<<endl;

    ALFLOAT __y[] = {-8,-22,-7,1};
    ALSp<ALFloatMatrix> _Y = ALPolynomial::construct(__y, sizeof(__y)/sizeof(ALFLOAT));

    ALSp<ALFloatMatrix> allroot = ALPolynomial::solve(_Y.get());
    for (int i=0; i<allroot->width(); ++i)
    {
        ALFLOAT v = *(allroot->vGetAddr(0)+i);
        cout << v << " : " << ALPolynomial::compute(_Y.get(), v)<<endl;
    }
}
class ALPolyTest:public GPTest
{
    public:
        virtual void run()
        {
            testmain();
        }
        ALPolyTest(){}
        virtual ~ALPolyTest(){}
};

static GPTestRegister<ALPolyTest> a("ALPolyTest");
