#ifndef ASSERT_H_STUB
#define ASSERT_H_STUB

#ifdef NDEBUG
#define assert(x) ((void)0)
#else
extern void __assert_fail(const char *e, const char *f, int l, const char *fn);
#define assert(x) ((x) ? (void)0 : __assert_fail(#x, __FILE__, __LINE__, 0))
#endif

#endif
