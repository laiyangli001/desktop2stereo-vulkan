#ifndef WINDOWS_H_STUB
#define WINDOWS_H_STUB

typedef unsigned long DWORD;
typedef unsigned long ULONG;
typedef int BOOL;
typedef void *HANDLE;
typedef void *HMODULE;
typedef void *HINSTANCE;
typedef char *LPSTR;
typedef const char *LPCSTR;
typedef const void *LPCVOID;
typedef unsigned short WCHAR;
typedef char CHAR;
#define TRUE 1
#define FALSE 0
#define NULL 0
#define FORMAT_MESSAGE_ALLOCATE_BUFFER 0x00000100
#define FORMAT_MESSAGE_IGNORE_INSERTS 0x00000200
#define FORMAT_MESSAGE_FROM_STRING 0x00000400
#define FORMAT_MESSAGE_FROM_HMODULE 0x00000800
#define FORMAT_MESSAGE_FROM_SYSTEM 0x00001000
#define FORMAT_MESSAGE_ARGUMENT_ARRAY 0x00002000
#define INVALID_HANDLE_VALUE ((HANDLE)(long long)-1)
HMODULE LoadLibraryA(LPCSTR name);
HMODULE GetModuleHandleA(LPCSTR name);
BOOL FreeLibrary(HMODULE h);
void *GetProcAddress(HMODULE h, LPCSTR name);
DWORD GetLastError(void);
DWORD FormatMessageA(DWORD flags, LPCVOID src, DWORD id, DWORD lang,
                     LPSTR buf, DWORD size, void *args);
void LocalFree(void *p);
#define WINAPI
typedef int (WINAPI *FARPROC)(void);

#endif
