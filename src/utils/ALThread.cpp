#include "utils/ALThread.h"
#include "utils/ALDebug.h"

ALThread::ALThread(bool loop):mData(NULL)
{
}
ALThread::~ALThread()
{
    if (NULL != mData)
    {
        bool join_result = stop();
        assert(join_result);
    }
}

bool ALThread::start()
{
    return platform_create();
}

bool ALThread::stop()
{
    wake();
    return platform_destroy();
}

bool ALThread::wake()
{
    return platform_wake();
}

void* ALThread::threadLoop(void* t)
{
    assert(NULL!=t);
    ALThread* th = (ALThread*)t;
    do
    {
        th->run();
        if (th->mLoop)
        {
            th->platform_wait();
        }
    }while(th->mLoop);
    return NULL;
}

