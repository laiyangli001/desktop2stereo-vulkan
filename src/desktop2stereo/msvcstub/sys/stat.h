#ifndef __SYS_STAT_H_STUB
#define __SYS_STAT_H_STUB
#include <stddef.h>
#include <time.h>
#define S_IFMT 0xF000
#define S_IFDIR 0x4000
#define S_IFREG 0x8000
#define S_IREAD 0x100
#define S_IWRITE 0x80
#define S_IEXEC 0x40
struct stat { int st_dev; short st_ino; unsigned short st_mode; short st_nlink; short st_uid; short st_gid; short st_rdev; long st_size; long st_atime; long st_mtime; long st_ctime; };
int stat(const char *p, struct stat *s);
int fstat(int fd, struct stat *s);
int lstat(const char *p, struct stat *s);
int chmod(const char *p, int m);
int fchmod(int fd, int m);
int mkdir(const char *p, int m);
int umask(int m);
int _stat(const char *p, struct stat *s);
int _fstat(int fd, struct stat *s);
int _chmod(const char *p, int m);
int _mkdir(const char *p);
#define _S_IFMT 0xF000
#define _S_IFDIR 0x4000
#define _S_IFREG 0x8000
#define _S_IREAD 0x100
#define _S_IWRITE 0x80
#define _S_IEXEC 0x40
#endif
