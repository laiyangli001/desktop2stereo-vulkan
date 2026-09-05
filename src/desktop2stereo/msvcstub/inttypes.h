#ifndef __INTTYPES_H_STUB
#define __INTTYPES_H_STUB
#include <stdint.h>
#define PRId64 "lld"
#define PRIu64 "llu"
#define PRIx64 "llx"
#define PRId32 "d"
#define PRIu32 "u"
#define PRIdPTR "lld"
#define PRIuPTR "llu"
#define SCNd64 "lld"
#define SCNu64 "llu"
typedef struct { long quot; long rem; } imaxdiv_t;
intmax_t imaxabs(intmax_t v);
imaxdiv_t imaxdiv(intmax_t n, intmax_t d);
intmax_t strtoimax(const char *s, char **e, int b);
uintmax_t strtoumax(const char *s, char **e, int b);
#endif
