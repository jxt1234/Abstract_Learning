#ifndef ABSTRACT_LEARNING_DEBUG_H
#define ABSTRACT_LEARNING_DEBUG_H

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
/*Print method*/
#ifdef BUILD_FOR_ANDROID
#include <android/log.h>
#define ALPRINT(format, ...) __android_log_print(ANDROID_LOG_INFO, "AL", format,##__VA_ARGS__)
#define ALPRINT_FL(format,...) __android_log_print(ANDROID_LOG_INFO, "AL", format", FUNC: %s, LINE: %d \n",##__VA_ARGS__, __func__, __LINE__)
#else
#define ALPRINT(format, ...) printf(format,##__VA_ARGS__)
#define ALPRINT_FL(format,...) printf(format", FUNC: %s, LINE: %d \n", ##__VA_ARGS__,__func__, __LINE__)
#endif
/*Add with line and function*/
#define FUNC_PRINT(x) ALPRINT(#x"=%d in %s, %d \n",(int)(x),  __func__, __LINE__);
#define FUNC_PRINT_ALL(x, type) ALPRINT(#x"= "#type" %"#type" in %s, %d \n",x,  __func__, __LINE__);

#define CHECK_POINTER(x) {if(NULL==x){FUNC_PRINT_ALL(x,p);break;}}

#ifndef BUILD_FOR_ANDROID
#define ALASSERT(x) \
    if (!(x)) al_dump_stack();assert(x);
#else
#define ALASSERT(x) \
    {bool ___result = (x);\
        if (!(___result))\
        FUNC_PRINT((___result));}
#endif


#define CHECK_POINTER(x) {if(NULL==x){FUNC_PRINT_ALL(x,p);break;}}

#ifdef __cplusplus
extern "C"{
#endif
void al_dump_stack();
#ifdef __cplusplus
}
#endif

#endif
