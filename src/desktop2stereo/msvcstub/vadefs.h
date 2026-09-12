#ifndef __VADEFS_H__
#define __VADEFS_H__
#include <stddef.h>
typedef __builtin_va_list va_list;
typedef __builtin_va_list __crt_va_list;
typedef __builtin_va_list _ArgList;
#define _ADDRESSOF(v) (&(v))
#define __crt_va_start(ap, x) __builtin_va_start(ap, x)
#define __crt_va_arg(ap, t) __builtin_va_arg(ap, t)
#define __crt_va_end(ap) __builtin_va_end(ap)
#define __crt_va_copy(d, s) __builtin_va_copy(d, s)
#define _crt_va_start(ap, x) __builtin_va_start(ap, x)
#define _crt_va_arg(ap, t) __builtin_va_arg(ap, t)
#define _crt_va_end(ap) __builtin_va_end(ap)
#ifndef va_start
#define va_start(ap, x) __crt_va_start(ap, x)
#endif
#ifndef va_arg
#define va_arg(ap, t) __crt_va_arg(ap, t)
#endif
#ifndef va_end
#define va_end(ap) __crt_va_end(ap)
#endif
#endif
