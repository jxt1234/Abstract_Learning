#include "utils/ALThread.h"
#include <semaphore.h>
#include <pthread.h>
#include <assert.h>

struct TData
{
    pthread_t id;
    sem_t* lock;
};

bool ALThread::platform_create()
{
    TData* data = new TData;
    if (mLoop)
    {
        data->lock = sem_open("ALThread", 0);
    }
    mData = (void*)data;
    pthread_create(&(data->id), NULL, ALThread::threadLoop, this);
    return true;
}

bool ALThread::platform_wake()
{
    assert(NULL!=mData);
    TData* data = (TData*)mData;
    sem_post((data->lock));
    return true;
}

bool ALThread::platform_destroy()
{
    assert(NULL!=mData);
    TData* data = (TData*)mData;
    if (mLoop)
    {
        sem_close(data->lock);
    }
    delete data;
    mData = NULL;
    return true;
}

bool ALThread::platform_join()
{
    assert(NULL!=mData);
    TData* data = (TData*)mData;
    pthread_join(data->id, NULL);
    return true;
}

void ALThread::platform_wait()
{
    assert(NULL!=mData);
    TData* data = (TData*)mData;
    sem_wait((data->lock));
}


ALSema::ALSema()
{
    sem_t* s = sem_open("ALSema", 0);
    mData = (void*)s;
}

ALSema::~ALSema()
{
    assert(NULL!=mData);
    sem_t* s = (sem_t*)(mData);
    sem_close(s);
}

void ALSema::wait()
{
    assert(NULL!=mData);
    sem_t* s = (sem_t*)(mData);
    sem_wait(s);
}

void ALSema::post()
{
    assert(NULL!=mData);
    sem_t* s = (sem_t*)(mData);
    sem_post(s);
}

ALMutex::ALMutex()
{
    pthread_mutex_t* m = new pthread_mutex_t;
    pthread_mutex_init(m, NULL);
    mData = (void*)m;
}

ALMutex::~ALMutex()
{
    assert(NULL!=mData);
    pthread_mutex_t* m = (pthread_mutex_t*)mData;
    pthread_mutex_destroy(m);
    delete m;
}

void ALMutex::lock()
{
    assert(NULL!=mData);
    pthread_mutex_lock((pthread_mutex_t*)mData);
}

void ALMutex::unlock()
{
    assert(NULL!=mData);
    pthread_mutex_unlock((pthread_mutex_t*)mData);
}
