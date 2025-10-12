#ifndef __SC_THREADS_H__
#define __SC_THREADS_H__

// cross platform thread handling
#ifdef _WIN32
#include <windows.h>
#include <process.h>

typedef HANDLE thread_t;
typedef CRITICAL_SECTION* mutex_t;
#else
#define min(a, b) ((a>b) ? (b) : (a))


#include <pthread.h>
#include <unistd.h>
typedef pthread_t thread_t;
typedef pthread_mutex_t mutex_t;
#endif

int get_cpu_count();
int create_thread(thread_t* thread, void* (*start_routine)(void*), void* arg);
int join_thread(thread_t thread);
int create_mutex(mutex_t* mutex);
int destroy_mutex(mutex_t mutex);
int lock_mutex(mutex_t mutex);
int unlock_mutex(mutex_t mutex);



#endif
