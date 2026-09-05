#ifndef MATH_H_STUB
#define MATH_H_STUB

#include <stddef.h>
double sin(double x); double cos(double x); double tan(double x);
double asin(double x); double acos(double x); double atan(double x); double atan2(double y, double x);
double sinh(double x); double cosh(double x); double tanh(double x);
double exp(double x); double log(double x); double log10(double x); double log2(double x);
double pow(double x, double y); double sqrt(double x); double cbrt(double x);
double floor(double x); double ceil(double x); double round(double x); double trunc(double x);
double fabs(double x); double fmod(double x, double y); double fmin(double x, double y); double fmax(double x, double y);
double frexp(double x, int *e); double ldexp(double x, int e); double modf(double x, double *i);
double hypot(double x, double y); double erf(double x); double erfc(double x);
double lgamma(double x); double tgamma(double x);
int isnan(double x); int isinf(double x); int isfinite(double x);
float sinf(float x); float cosf(float x); float tanf(float x);
float sqrtf(float x); float powf(float x, float y); float fabsf(float x);
float floorf(float x); float ceilf(float x); float fminf(float x, float y); float fmaxf(float x, float y);

#endif
