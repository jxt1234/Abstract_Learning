#include "learn/ALSVMHandleFactory.h"
#include <sstream>
#include "svm/OneClass.h"
#include "svm/MultiClass.h"
using namespace std;
typedef map<string, string>::const_iterator ITER;
ALSp<ALSVM::Reportor> ALSVMHandleFactory::create(const std::map<std::string, std::string>& heads)
{
    ITER iter = heads.find("svm_type");
    assert(iter!=heads.end());
    istringstream is(iter->second);
    string type;
    is >> type;
    const char* one[] = {
        "one_class",
        "epsilon_svr",
        "nu_svr"
    };
    const char* multi[] = {
        "c_svc",
        "nu_svc"
    };
    for (int i=0; i<sizeof(one)/sizeof(const char*); ++i)
    {
        if (type == one[i])
        {
            return new OneClass;
        }
    }
    for (int i=0; i<sizeof(multi)/sizeof(const char*); ++i)
    {
        if (type == multi[i])
        {
            return new MultiClass(heads);
        }
    }

    assert(false);
    return NULL;
}
