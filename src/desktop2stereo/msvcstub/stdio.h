#ifndef STDIO_H_STUB
#define STDIO_H_STUB

#include <stddef.h>
#include <stdarg.h>
typedef struct _iobuf FILE;
extern FILE *stdin, *stdout, *stderr;
#define EOF (-1)
int printf(const char *fmt, ...);
int fprintf(FILE *f, const char *fmt, ...);
int sprintf(char *b, const char *fmt, ...);
int snprintf(char *b, size_t n, const char *fmt, ...);
int vprintf(const char *fmt, va_list ap);
int vfprintf(FILE *f, const char *fmt, va_list ap);
int vsnprintf(char *b, size_t n, const char *fmt, va_list ap);
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
