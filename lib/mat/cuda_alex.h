#ifdef __cplusplus
extern "C" {
#endif

void testVectors(int rows);
int multCudaCSRKernelWarp(MatriceCsr *mat,Vector *vector,Vector *result,double *time,unsigned int threadsPerBlock);
int multCudaCSRKernelLinear(MatriceCsr *mat,Vector *vector,Vector *result,double *time,unsigned int threadsPerBlock);
#ifdef __cplusplus
}
#endif