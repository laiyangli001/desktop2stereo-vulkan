#ifndef IO_H_STUB
#define IO_H_STUB

#include <stddef.h>
#define SEEK_SET 0
#define SEEK_CUR 1
#define SEEK_END 2
#define _O_RDONLY 0x0000
#define _O_WRONLY 0x0001
#define _O_RDWR 0x0002
#define _O_APPEND 0x0008
#define _O_CREAT 0x0100
#define _O_TRUNC 0x0200
#define _O_BINARY 0x8000
int _open(const char *path, int flags, ...);
int _close(int fd);
long _lseek(int fd, long off, int whence);
int _read(int fd, void *buf, unsigned int count);
int _write(int fd, const void *buf, unsigned int count);
int _commit(int fd);
int _chsize(int fd, long size);
long _filelength(int fd);
int _fileno(void *stream);
int _isatty(int fd);
int _eof(int fd);
void _setmode(int fd, int mode);
int _access(const char *path, int mode);
int _mkdir(const char *path);
int _unlink(const char *path);
int _rename(const char *a, const char *b);
char *_strerror(const char *msg);
char *strerror(int err);

#endif
