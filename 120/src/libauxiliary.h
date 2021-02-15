#ifndef _LIBAUXILIARY
#define _LIBAUXILIARY

#ifdef __cplusplus
extern "C" {
#endif

/* libauxiliary.c */

void addgaussnoise(float *U, float *V, double sigma, unsigned long seed, 
		   size_t size);

#ifdef __cplusplus
}
#endif

#endif /* !_LIBAUXILIARY */
