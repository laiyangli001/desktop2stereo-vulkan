typedef int BOOL; typedef void *HINSTANCE; typedef unsigned long DWORD; typedef void *LPVOID;
BOOL DllMain(HINSTANCE h, DWORD r, LPVOID p) { return 1; }
void DllMainCRTStartup(void) {}
void _DllMainCRTStartup(void) {}
void __chkstk(void) {}
void __assert_fail(const char *e, const char *f, int l, const char *fn) { for (;;) {} }
int _fltused = 0;
void __security_check_cookie(void *c) {}
void *__security_cookie = (void*)0;
