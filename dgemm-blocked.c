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
#include <immintrin.h>
#include <xmmintrin.h>
#include <mmintrin.h>
#include <pmmintrin.h>
#include <emmintrin.h>

const char* dgemm_desc = "Simple blocked dgemm.";


#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 128
#endif

#if !defined(CACHE_LINE_DOUBLE)
#define CACHE_LINE_DOUBLE 8 
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/*transpose lda-by-lda matrix*/ 
// TODO: review, loop unrolling
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

/* transpose 4x4 double matrix */
static void transpose_4x4_matrix(double* restrict src, double* restrict dst){
  __m256d row0 = _mm256_loadu_pd(src);
  __m256d row1 = _mm256_loadu_pd(src+4);
  __m256d row2 = _mm256_loadu_pd(src+8);
  __m256d row3 = _mm256_loadu_pd(src+12);

  __m256d tmp3, tmp2, tmp1, tmp0; 

  tmp0 = _mm256_permute4x64_pd(_mm256_unpacklo_pd(row0, row1), 0b11011000); 
  tmp2 = _mm256_permute4x64_pd(_mm256_unpacklo_pd(row2, row3), 0b11011000); 
  tmp1 = _mm256_permute4x64_pd(_mm256_unpackhi_pd(row0, row1), 0b11011000);
  tmp3 = _mm256_permute4x64_pd(_mm256_unpackhi_pd(row2, row3), 0b11011000); 

  row0 = _mm256_permute4x64_pd(_mm256_shuffle_pd(tmp0, tmp2, 0b0000), 0b11011000); 
  row1 = _mm256_permute4x64_pd(_mm256_shuffle_pd(tmp1, tmp3, 0b0000), 0b11011000); 
  row2 = _mm256_permute4x64_pd(_mm256_shuffle_pd(tmp0, tmp2, 0b1111), 0b11011000); 
  row3 = _mm256_permute4x64_pd(_mm256_shuffle_pd(tmp1, tmp3, 0b1111), 0b11011000); 

  _mm256_storeu_pd(dst, row0);
  _mm256_storeu_pd(dst+4, row1);
  _mm256_storeu_pd(dst+8, row2);
  _mm256_storeu_pd(dst+12, row3);
}

static void transpose_4rows(__m256d row0, __m256d row1 , __m256d row2 , __m256d row3 ){
  __m256d tmp3, tmp2, tmp1, tmp0; 

  tmp0 = _mm256_permute4x64_pd(_mm256_unpacklo_pd(row0, row1), 0b11011000); 
  tmp2 = _mm256_permute4x64_pd(_mm256_unpacklo_pd(row2, row3), 0b11011000); 
  tmp1 = _mm256_permute4x64_pd(_mm256_unpackhi_pd(row0, row1), 0b11011000);
  tmp3 = _mm256_permute4x64_pd(_mm256_unpackhi_pd(row2, row3), 0b11011000); 

  row0 = _mm256_permute4x64_pd(_mm256_shuffle_pd(tmp0, tmp2, 0b0000), 0b11011000); 
  row1 = _mm256_permute4x64_pd(_mm256_shuffle_pd(tmp1, tmp3, 0b0000), 0b11011000); 
  row2 = _mm256_permute4x64_pd(_mm256_shuffle_pd(tmp0, tmp2, 0b1111), 0b11011000); 
  row3 = _mm256_permute4x64_pd(_mm256_shuffle_pd(tmp1, tmp3, 0b1111), 0b11011000); 
}

/* SIMD intrinsics */
// dimensions of A 4xK, B Kx4, C 4x4
static void do_block_simd_4x4 (int lda, int K, double* A, double* B, double* restrict C){  
  //     c0 | c1 | c2 | c3 |
  //    --------------------
  // r0 |   |    |    |    |
  // r1 |   |    |    |    |
  //     ------------------
  // r2 |   |    |    |    |
  // r3 |   |    |    |    |
  //    --------------------

  // sets of 2x1
  __m128d A_r0_r1, A_r2_r3;
  // sets of 1x1 where the cell is duplicated to fit __m128d
  __m128d B_c0, B_c1, B_c2, B_c3;

  // load elements in C into sets of 2x1
  __m128d C_r0_r1_c0 = _mm_loadu_pd(C);
  // printf("%f %f\n", (double)C_r0_r1_c0[0], (double)C_r0_r1_c0[1]);
  __m128d C_r2_r3_c0 = _mm_loadu_pd(C+2);

  __m128d C_r0_r1_c1 = _mm_loadu_pd(C+lda);
  __m128d C_r2_r3_c1 = _mm_loadu_pd(C+lda+2);

  __m128d C_r0_r1_c2 = _mm_loadu_pd(C+2*lda);
  __m128d C_r2_r3_c2 = _mm_loadu_pd(C+2*lda+2);

  __m128d C_r0_r1_c3 = _mm_loadu_pd(C+3*lda);
  __m128d C_r2_r3_c3 = _mm_loadu_pd(C+3*lda+2);

  for (int k=0; k<K; ++k){
    // load elements in A into sets of 2x1
    A_r0_r1 = _mm_loadu_pd(A+k*lda);
    A_r2_r3 = _mm_loadu_pd(A+k*lda+2);

    // load elements in B, 1 cell a time, duplicate the cell to fit m128d
    B_c0 = _mm_loaddup_pd(B+k);
    B_c1 = _mm_loaddup_pd(B+k+lda);
    B_c2 = _mm_loaddup_pd(B+k+2*lda);
    B_c3 = _mm_loaddup_pd(B+k+3*lda);
  
    // do calculation
    C_r0_r1_c0 = _mm_add_pd(C_r0_r1_c0, _mm_mul_pd(A_r0_r1, B_c0));
    C_r0_r1_c1 = _mm_add_pd(C_r0_r1_c1, _mm_mul_pd(A_r0_r1, B_c1));
    C_r0_r1_c2 = _mm_add_pd(C_r0_r1_c2, _mm_mul_pd(A_r0_r1, B_c2));
    C_r0_r1_c3 = _mm_add_pd(C_r0_r1_c3, _mm_mul_pd(A_r0_r1, B_c3));

    C_r2_r3_c0 = _mm_add_pd(C_r2_r3_c0, _mm_mul_pd(A_r2_r3, B_c0));
    C_r2_r3_c1 = _mm_add_pd(C_r2_r3_c1, _mm_mul_pd(A_r2_r3, B_c1));
    C_r2_r3_c2 = _mm_add_pd(C_r2_r3_c2, _mm_mul_pd(A_r2_r3, B_c2));
    C_r2_r3_c3 = _mm_add_pd(C_r2_r3_c3, _mm_mul_pd(A_r2_r3, B_c3));
  }

  // store back to mem
  _mm_storeu_pd(C, C_r0_r1_c0);
  _mm_storeu_pd(C+2, C_r2_r3_c0);

  _mm_storeu_pd(C+lda, C_r0_r1_c1);
  _mm_storeu_pd(C+lda+2, C_r2_r3_c1);

  _mm_storeu_pd(C+2*lda, C_r0_r1_c2);
  _mm_storeu_pd(C+2*lda+2, C_r2_r3_c2);

  _mm_storeu_pd(C+3*lda, C_r0_r1_c3);
  _mm_storeu_pd(C+3*lda+2, C_r2_r3_c3);
}

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* restrict C){
  // bound to do simd_4x4
  int M_simd = (M>>2) << 2;
  int N_simd = (N>>2) << 2;
  // int K_simd = (K>>2) << 2;
  // printf("%d %d %d\n", M_simd, N_simd, K);
  for (int j = 0; j < N_simd; j+=4)
    for (int i = 0; i < M_simd; i+=4) 
        do_block_simd_4x4 (lda, K, A+i, B+j*lda, C+i+j*lda);

  // ----------
  // | simd |R2|
  // | simd |  |
  // ----------
  // | R1      |
  // ----------
  // remaining R1
  for (int i=M_simd; i<M; ++i)
    for (int j=0; j<N; ++j)
      for (int k=0; k<K; ++k)
        C[i+j*lda] += A[i+k*lda] * B[k+j*lda];  
  // remaining R2
  for (int j=N_simd; j<N; ++j)
    for (int i=0; i<M_simd; ++i)
      for (int k=0; k<K; ++k)
        C[i+j*lda] += A[i+k*lda] * B[k+j*lda];  
}

/*cache oblivious*/
static void do_block_oblivious (int lda, int M, int N, int K, double* A, double* B, double* restrict C){
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
void square_dgemm (int lda, double* const A, double* const B, double* restrict C)
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
