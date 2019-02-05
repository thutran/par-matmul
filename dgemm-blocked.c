/* 
    Please include compiler name below (you may also include any other modules you would like to be loaded)

COMPILER= gnu

    Please include All compiler flags and libraries as you want them run. You can simply copy this over from the Makefile's first few lines
 
CC = cc
OPT = -O3
CFLAGS = -Wall -std=gnu99 $(OPT)
MKLROOT = /opt/intel/composer_xe_2013.1.117/mkl
LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm

*/
#include <stdio.h>
const char* dgemm_desc = "Simple blocked dgemm.";


#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 16
#endif

#if !defined(CACHE_LINE_DOUBLE)
#define CACHE_LINE_DOUBLE 8 
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/*transpose lda-by-lda matrix*/ 
static void transpose(int lda, double* const src, double* restrict dst){
  int n_cache_lines = lda*lda/CACHE_LINE_DOUBLE;
  // #pragma omp parallel for
  for (int i=0; i<=n_cache_lines; ++i){
    for (int j=0; j<CACHE_LINE_DOUBLE && ((i*CACHE_LINE_DOUBLE+j)<lda*lda); ++j){
      // id = i+j
      int col = (i*CACHE_LINE_DOUBLE+j)%lda; // column in src
      int row = (i*CACHE_LINE_DOUBLE+j)/lda; // row in src
      dst[col*lda + row] = src[i*CACHE_LINE_DOUBLE+j];
    }
  }
}

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  /* order: jki (BCA) */
  // column in B
  for (int j=0; j<N; ++j)
    // A*B pair
    for (int k=0; k<K; ++k)
      // icc compiler loop unrolling
      #pragma unroll(BLOCK_SIZE)
      // row in A
      for (int i=0; i<M; ++i)
        C[i+j*lda] += A[i+k*lda] * B[k+j*lda];
}

/*cache oblivious*/
static void do_block_oblivious (int lda, int M, int N, int K, double* A, double* B, double* C){
  if (M+N+K <= BLOCK_SIZE*3){
    /* order: jki (BCA) */
    // column in B
    for (int j=0; j<N; ++j)
      // A*B pair
      for (int k=0; k<K; ++k)
        // row in A
        for (int i=0; i<M; ++i)
          C[i+j*lda] += A[i+k*lda] * B[k+j*lda];
    
  } else {
    // A1 | A3    B1 | B3   C1 | C3
    // -------    -------   -------
    // A2 | A4    B2 | B4   C2 | C4
    int m= (M%2==0) ? M/2 : (M-1)/2;
    int n= (N%2==0) ? N/2 : (N-1)/2;
    int k= (K%2==0) ? K/2 : (K-1)/2;
    // int m=M/2, n=N/2, k=K/2;

    // compute C1 = A1B1 + A3B2
    do_block(lda, m, n, k, A, B, C);
    do_block(lda, m, n, K-k, A+k*lda, B+k, C);

    // compute C2 = A2B1 + A4B2
    do_block(lda, M-m, n, k, A+m, B, C+m);
    do_block(lda, M-m, n, K-k, A+k*lda+m, B+k, C+m);

    // compute C3 = A1B3 + A3B4
    do_block(lda, m, N-n, k, A, B+n*lda, C+n*lda);
    do_block(lda, m, N-n, K-k, A+k*lda, B+n*lda+k, C+n*lda);

    // compute C4 = A2B3 + A4B4
    do_block(lda, M-m, N-n, k, A+m, B+n*lda, C+n*lda+k);
    do_block(lda, M-m, N-n, K-k, A+k*lda+m, B+n*lda+k, C+n*lda+k);
  }
}



/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{
  // // For each block-row of A 
  for (int i = 0; i < lda; i += BLOCK_SIZE)
    // For each block-column of B 
    for (int j = 0; j < lda; j += BLOCK_SIZE)    
      // Accumulate block dgemms into block of C 
      for (int k = 0; k < lda; k += BLOCK_SIZE)
      {
      	// Correct block dimensions if block "goes off edge of" the matrix 
      	int M = min (BLOCK_SIZE, lda-i);
      	int N = min (BLOCK_SIZE, lda-j);
      	int K = min (BLOCK_SIZE, lda-k);

      	/* Perform individual block dgemm */
      	do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
      }

  // cache-oblivious
  // do_block_oblivious(lda, lda, lda, lda, A, B, C);
}
