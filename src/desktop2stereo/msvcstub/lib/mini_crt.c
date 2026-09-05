typedef int BOOL;
typedef void *HINSTANCE;
typedef unsigned long DWORD;
typedef void *LPVOID;
typedef void *HMODULE;
BOOL DllMain(HINSTANCE h, DWORD r, LPVOID p) { return 1; }
void DllMainCRTStartup(void) {}
void __assert_fail(const char *e, const char *f, int l, const char *fn) { for (;;) {} }
void __security_check_cookie(void *c) {}
void __report_gsfailure(void) { for (;;) {} }
void _fltused(void) {}
void _purecall(void) { for (;;) {} }
int _invalid_parameter(void) { return 0; }
int _CrtDbgReport(int a, const char *b, int c, const char *d, const char *e) { return 0; }
