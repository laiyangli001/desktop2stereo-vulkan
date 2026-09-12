#ifndef STDLIB_H_STUB
#define STDLIB_H_STUB

#include <stddef.h>
#include <stdint.h>
#define EXIT_SUCCESS 0
#define EXIT_FAILURE 1
void *malloc(size_t _s);
void free(void *p);
void *calloc(size_t _n, size_t _s);
void *realloc(void *p, size_t _s);
void abort(void);
int atexit(void (*f)(void));
extern int errno;
#define RAND_MAX 32767
int rand(void);
void srand(unsigned int _s);
double atof(const char *s);
int atoi(const char *s);
long atol(const char *s);
double strtod(const char *s, char **e);
long strtol(const char *s, char **e, int b);
unsigned long strtoul(const char *s, char **e, int b);
int abs(int x);
long labs(long x);
char *getenv(const char *n);
int system(const char *s);
void qsort(void *b, size_t n, size_t s, int (*c)(const void*, const void*));

#endif
