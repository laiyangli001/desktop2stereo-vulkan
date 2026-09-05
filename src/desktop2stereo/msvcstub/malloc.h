#ifndef __MALLOC_H_STUB
#define __MALLOC_H_STUB
#include <stddef.h>
void *malloc(size_t s);
void free(void *p);
void *calloc(size_t n, size_t s);
void *realloc(void *p, size_t s);
void * _aligned_malloc(size_t s, size_t a);
void _aligned_free(void *p);
size_t _msize(void *p);
int _heapmin(void);
#endif

#define alloca(s) _alloca(s)
void *_alloca(size_t s);
