#ifndef __VCRUNTIME_H__
#define __VCRUNTIME_H__
#include <sal.h>
#include <vadefs.h>
#ifdef __cplusplus
#define _CRT_BEGIN_C_HEADER extern "C" {
#define _CRT_END_C_HEADER }
#else
#define _CRT_BEGIN_C_HEADER
#define _CRT_END_C_HEADER
#endif
#ifndef _In_opt_z_
#define _In_opt_z_
#endif
#ifndef _Ret_z_
#define _Ret_z_
#endif
#ifndef _CRT_UNUSED
#define _CRT_UNUSED(x) (void)(x)
#endif
#ifndef __crt_bool
#define __crt_bool bool
#endif
#ifndef _CRT_STDIO_INLINE
#define _CRT_STDIO_INLINE __forceinline
#endif
#ifndef _ACRTIMP
#define _ACRTIMP __declspec(dllimport)
#endif
#ifndef _VCRT_DEFINE_IS_WCHAR_UNICODE
#define _VCRT_DEFINE_IS_WCHAR_UNICODE
#endif
#endif

#ifndef __CRTDECL
#define __CRTDECL __cdecl
#endif
#ifndef __cdecl
#define __cdecl
#endif

#ifndef _CRT_NONSTDC_DEPRECATE
#define _CRT_NONSTDC_DEPRECATE(x)
#endif
#ifndef _CRT_INSECURE_DEPRECATE
#define _CRT_INSECURE_DEPRECATE(x)
#endif
#ifndef _CRT_DEPRECATE_TEXT
#define _CRT_DEPRECATE_TEXT(x)
#endif
#ifndef _CRT_NOINLINE
#define _CRT_NOINLINE
#endif
#ifndef _CRT_CONST_CORRECT_OVERLOADS
#define _CRT_CONST_CORRECT_OVERLOADS
#endif
#ifndef _Check_return_
#define _Check_return_
#endif
#ifndef _Ret_notnull_
#define _Ret_notnull_
#endif

#ifndef _VCRTIMP
#define _VCRTIMP __declspec(dllimport)
#endif
#ifndef _VCRT_ALLOCATOR
#define _VCRT_ALLOCATOR __declspec(dllimport)
#endif
#ifndef _ACRTALLOC
#define _ACRTALLOC __declspec(dllimport)
#endif
#ifndef _UINTPTR_T_DEFINED
#define _UINTPTR_T_DEFINED
typedef unsigned __int64 uintptr_t;
typedef __int64 intptr_t;
typedef unsigned __int64 uintptr_t_alt;
#endif
#ifndef _SIZE_T_DEFINED
#define _SIZE_T_DEFINED
typedef unsigned __int64 size_t;
#endif
#ifndef _SSIZE_T_DEFINED
#define _SSIZE_T_DEFINED
typedef __int64 ssize_t;
#endif
