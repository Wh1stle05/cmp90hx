// cuBLAS/bandwidth microbench for CMP 90HX (same as earlier)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA err %s\n",cudaGetErrorString(e));exit(1);} } while(0)
#define CUBLAS_CHECK(x) do { cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){printf("CUBLAS err %d\n",(int)s);exit(1);} } while(0)

static double now(){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }

int main(int argc,char**argv){
  if(argc<2){printf("usage: bench <fp32|tf32|fp16|bf16|int8|copyh2d|copyd2h|copydev>\n");return 1;}
  const char* mode=argv[1];
  const int n=8192;
  double run_s=10.0;

  cublasHandle_t h; CUBLAS_CHECK(cublasCreate(&h));

  if(strcmp(mode,"fp32")==0||strcmp(mode,"tf32")==0){
    size_t sz=(size_t)n*n*4;
    float *A,*B,*C; CHECK(cudaMalloc(&A,sz)); CHECK(cudaMalloc(&B,sz)); CHECK(cudaMalloc(&C,sz));
    CHECK(cudaMemset(A,0,sz)); CHECK(cudaMemset(B,0,sz)); CHECK(cudaMemset(C,0,sz));
    float alpha=1.0f,beta=0.0f;
    if(strcmp(mode,"tf32")==0) CUBLAS_CHECK(cublasSetMathMode(h,CUBLAS_TF32_TENSOR_OP_MATH));
    CUBLAS_CHECK(cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&alpha,A,n,B,n,&beta,C,n));
    CHECK(cudaDeviceSynchronize());
    double t0=now(); long long it=0;
    while(now()-t0<run_s){
      CUBLAS_CHECK(cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&alpha,A,n,B,n,&beta,C,n));
      if(++it%16==0){ CHECK(cudaDeviceSynchronize()); }
    }
    CHECK(cudaDeviceSynchronize()); double dt=now()-t0;
    double flops=2.0*(double)n*n*n*it;
    printf("RESULT mode=%s iters=%lld time=%.2fs rate=%.2f TFLOPS\n",mode,it,dt,flops/dt/1e12);
    cudaFree(A);cudaFree(B);cudaFree(C);
  }
  else if(strcmp(mode,"fp16")==0||strcmp(mode,"bf16")==0){
    cudaDataType_t abt = (strcmp(mode,"fp16")==0)?CUDA_R_16F:CUDA_R_16BF;
    size_t sz=(size_t)n*n*2;
    void *A,*B; float *C; CHECK(cudaMalloc(&A,sz)); CHECK(cudaMalloc(&B,sz)); CHECK(cudaMalloc(&C,(size_t)n*n*4));
    CHECK(cudaMemset(A,0,sz)); CHECK(cudaMemset(B,0,sz)); CHECK(cudaMemset(C,0,(size_t)n*n*4));
    float alpha=1.0f,beta=0.0f;
    CUBLAS_CHECK(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&alpha,A,abt,n,B,abt,n,&beta,C,CUDA_R_32F,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CHECK(cudaDeviceSynchronize());
    double t0=now(); long long it=0;
    while(now()-t0<run_s){
      CUBLAS_CHECK(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&alpha,A,abt,n,B,abt,n,&beta,C,CUDA_R_32F,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      if(++it%16==0) CHECK(cudaDeviceSynchronize());
    }
    CHECK(cudaDeviceSynchronize()); double dt=now()-t0;
    double flops=2.0*(double)n*n*n*it;
    printf("RESULT mode=%s iters=%lld time=%.2fs rate=%.2f TFLOPS\n",mode,it,dt,flops/dt/1e12);
    cudaFree(A);cudaFree(B);cudaFree(C);
  }
  else if(strcmp(mode,"int8")==0){
    size_t sz=(size_t)n*n;
    signed char *A,*B; int *C;
    CHECK(cudaMalloc(&A,sz)); CHECK(cudaMalloc(&B,sz)); CHECK(cudaMalloc(&C,(size_t)n*n*4));
    CHECK(cudaMemset(A,1,sz)); CHECK(cudaMemset(B,1,sz)); CHECK(cudaMemset(C,0,(size_t)n*n*4));
    float alpha=1.0f,beta=0.0f;
    CUBLAS_CHECK(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&alpha,A,CUDA_R_8I,n,B,CUDA_R_8I,n,&beta,C,CUDA_R_32I,n,CUDA_R_32I,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CHECK(cudaDeviceSynchronize());
    double t0=now(); long long it=0;
    while(now()-t0<run_s){
      CUBLAS_CHECK(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&alpha,A,CUDA_R_8I,n,B,CUDA_R_8I,n,&beta,C,CUDA_R_32I,n,CUDA_R_32I,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      if(++it%16==0) CHECK(cudaDeviceSynchronize());
    }
    CHECK(cudaDeviceSynchronize()); double dt=now()-t0;
    double ops=2.0*(double)n*n*n*it;
    printf("RESULT mode=%s iters=%lld time=%.2fs rate=%.2f TOPS\n",mode,it,dt,ops/dt/1e12);
    cudaFree(A);cudaFree(B);cudaFree(C);
  }
  else if(strcmp(mode,"copyh2d")==0||strcmp(mode,"copyd2h")==0||strcmp(mode,"copydev")==0){
    size_t sz=1024UL*1024UL*1024UL;
    unsigned char *d,*d2=0; CHECK(cudaMalloc(&d,sz));
    if(strcmp(mode,"copydev")==0) CHECK(cudaMalloc(&d2,sz));
    unsigned char *h=0; CHECK(cudaMallocHost(&h,sz));
    CHECK(cudaMemset(d,1,sz)); memset(h,2,sz);
    cudaMemcpyKind k; unsigned char *src=d,*dst=d2;
    if(strcmp(mode,"copyh2d")==0){ k=cudaMemcpyHostToDevice; src=h; dst=d; }
    else if(strcmp(mode,"copyd2h")==0){ k=cudaMemcpyDeviceToHost; src=d; dst=h; }
    else { k=cudaMemcpyDeviceToDevice; src=d; dst=d2; }
    for(int i=0;i<3;i++) CHECK(cudaMemcpy(dst,src,sz,k));
    CHECK(cudaDeviceSynchronize());
    double t0=now(); int reps=0;
    while(now()-t0<run_s){ CHECK(cudaMemcpy(dst,src,sz,k)); reps++; }
    double dt=now()-t0;
    printf("RESULT mode=%s bytes=%.2f GiB reps=%d time=%.2fs rate=%.2f GB/s\n",mode,(double)sz/1e9,reps,dt,(double)sz*reps/dt/1e9);
    cudaFree(d); if(d2) cudaFree(d2); cudaFreeHost(h);
  }
  else { printf("unknown mode %s\n",mode); return 1; }
  CUBLAS_CHECK(cublasDestroy(h));
  return 0;
}
