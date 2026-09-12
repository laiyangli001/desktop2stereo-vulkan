#ifndef STDIO_H_STUB
#define STDIO_H_STUB

#include <stddef.h>
#include <stdarg.h>
typedef struct _iobuf FILE;
extern FILE *stdin, *stdout, *stderr;
#define EOF (-1)
#if defined(_MSC_VER) || 1
static __inline int __crt_snprintf(char *b, size_t n, const char *fmt, ...) {
  int r; __builtin_va_list ap; __builtin_va_start(ap, fmt);
  extern int __stdio_common_vsnprintf_s(unsigned __int64 opt, char *b, size_t n, const char *fmt, void *loc, __builtin_va_list ap);
  r = __stdio_common_vsnprintf_s(0, b, n, fmt, 0, ap);
  __builtin_va_end(ap); return r;
}
#define snprintf __crt_snprintf
#else
int snprintf(char *b, size_t n, const char *fmt, ...);
#endif
int printf(const char *fmt, ...);
int fprintf(FILE *f, const char *fmt, ...);
#define sprintf __crt_snprintf
int snprintf(char *b, size_t n, const char *fmt, ...);



FILE *fopen(const char *p, const char *m);
int fclose(FILE *f);
size_t fread(void *d, size_t s, size_t n, FILE *f);
size_t fwrite(const void *d, size_t s, size_t n, FILE *f);
int fflush(FILE *f);
int fgetc(FILE *f);
char *fgets(char *b, int n, FILE *f);
int fputc(int c, FILE *f);
int puts(const char *s);
int fputs(const char *s, FILE *f);
int remove(const char *p);
long ftell(FILE *f);
int fseek(FILE *f, long o, int w);
void perror(const char *s);
int fscanf(FILE *f, const char *fmt, ...);
int sscanf(const char *b, const char *fmt, ...);

#endif
