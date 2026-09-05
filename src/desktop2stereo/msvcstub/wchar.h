#ifndef WCHAR_H_STUB
#define WCHAR_H_STUB

#include <stddef.h>
typedef unsigned short wchar_t;
typedef struct { unsigned long __size; } mbstate_t;
wchar_t *wcscpy(wchar_t *d, const wchar_t *s);
wchar_t *wcsncpy(wchar_t *d, const wchar_t *s, size_t n);
size_t wcslen(const wchar_t *s);
int wcscmp(const wchar_t *a, const wchar_t *b);
int wcsncmp(const wchar_t *a, const wchar_t *b, size_t n);
wchar_t *wcschr(const wchar_t *s, wchar_t c);
wchar_t *wcsstr(const wchar_t *h, const wchar_t *n);
wchar_t *wmemcpy(wchar_t *d, const wchar_t *s, size_t n);
wchar_t *wmemmove(wchar_t *d, const wchar_t *s, size_t n);
int wmemcmp(const wchar_t *a, const wchar_t *b, size_t n);
wchar_t *wmemset(wchar_t *d, wchar_t c, size_t n);
size_t wcsxfrm(wchar_t *d, const wchar_t *s, size_t n);
int wctob(int c);

#endif
