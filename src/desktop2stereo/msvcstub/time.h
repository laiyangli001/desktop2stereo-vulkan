#ifndef TIME_H_STUB
#define TIME_H_STUB

#include <stddef.h>
typedef long time_t;
typedef long clock_t;
struct tm { int tm_sec; int tm_min; int tm_hour; int tm_mday; int tm_mon; int tm_year; int tm_wday; int tm_yday; int tm_isdst; };
#define CLOCKS_PER_SEC 1000
time_t time(time_t *t);
clock_t clock(void);
double difftime(time_t a, time_t b);
time_t mktime(struct tm *t);
char *ctime(const time_t *t);
char *asctime(const struct tm *t);
struct tm *gmtime(const time_t *t);
struct tm *localtime(const time_t *t);
size_t strftime(char *s, size_t n, const char *f, const struct tm *t);

#endif
