#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdatomic.h>
#include <x86intrin.h>
static _Atomic uint64_t last;
static _Atomic int lock;
static long warps; static int64_t worst;
static uint64_t rd(void){ unsigned a; uint64_t t=__rdtscp(&a); _mm_lfence(); return t; }
static void *run(void *arg){
  int cpu=(int)(long)arg; cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu,&s); pthread_setaffinity_np(pthread_self(),sizeof s,&s);
  for(long i=0;i<20000000;i++){
    while(atomic_exchange(&lock,1)) ;
    uint64_t prev=atomic_load(&last); uint64_t now=rd(); atomic_store(&last,now);
    atomic_store(&lock,0);
    if(now<prev){ warps++; if((int64_t)(prev-now)>worst) worst=prev-now; }
  }
  return 0;
}
int main(int c,char**v){ int a=atoi(v[1]),b=atoi(v[2]); pthread_t x,y;
  pthread_create(&x,0,run,(void*)(long)a); pthread_create(&y,0,run,(void*)(long)b); pthread_join(x,0); pthread_join(y,0);
  printf("cpus %d %d warps %ld worst %ld cycles\n",a,b,warps,(long)worst); }
