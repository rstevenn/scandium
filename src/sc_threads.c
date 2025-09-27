#include "sc_threads.h"




int get_cpu_count() {
    #ifdef _WIN32
        SYSTEM_INFO sysinfo;
        GetSystemInfo(&sysinfo);
        return sysinfo.dwNumberOfProcessors;
    #else
        return sysconf(_SC_NPROCESSORS_ONLN);
    #endif
}




int create_thread(thread_t* thread, void* (*start_routine)(void*), void* arg) {
    #ifdef _WIN32
        *thread = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)start_routine, arg, 0, NULL);
        if (*thread == NULL) {
            return -1;
        }
        return 0;

    #else
        int rc = pthread_create(thread, NULL, start_routine, arg);
        if (rc != 0) {
            return -1;
        }
        return 0;
    
    #endif
}



int join_thread(thread_t thread) {
    #ifdef _WIN32
        WaitForSingleObject(thread, INFINITE);
        CloseHandle(thread);
        return 0;
    #else
        int rc = pthread_join(thread, NULL);
        if (rc != 0) {
            return -1;
        }
        return 0;
    #endif
}



int create_mutex(mutex_t* mutex);
int destroy_mutex(mutex_t mutex);
int lock_mutex(mutex_t mutex);
int unlock_mutex(mutex_t mutex);