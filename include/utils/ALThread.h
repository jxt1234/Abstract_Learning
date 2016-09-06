#ifndef UTILS_THREAD_H
#define UTILS_THREAD_H
#include <stdlib.h>

class ALSema
{
    public:
        ALSema();
        ~ALSema();
        void wait();
        void post();
    private:
        void* mData;
};

class ALMutex
{
    public:
        ALMutex();
        ~ALMutex();
        void lock();
        void unlock();
    private:
        void* mData;
};

class ALAutoALMutex
{
    public:
        ALAutoALMutex(ALMutex& _m):m(_m){m.lock();}
        ~ALAutoALMutex(){m.unlock();}
    private:
        ALMutex& m;
};


class ALThread
{
    public:
        ALThread(bool loop = false);
        virtual ~ALThread();
        bool start();
        bool stop();
        bool wake();
        virtual void run() = 0;
    private:
        static void* threadLoop(void* t);
        bool platform_create();
        bool platform_destroy();
        bool platform_wake();
        void platform_wait();
        bool platform_join();
        void* mData;
        bool mLoop;
};


#endif
