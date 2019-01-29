# On Bridges we will check versus your performance versus Intel MKL library's BLAS. 

CC = gcc

OPT = -O3

ifeq ($(CC), gcc)
	SIMD = -mavx2
else ifeq ($(CC), icc)
	SIMD = -axCORE-AVX2
else ifeq ($(CC), pgcc)
	SIMD = -tp=haswell-64
else
	SIMD = 
endif

ifdef ENABLE_OPENMP
	ifeq ($(CC), gcc)
		OMP = -fopenmp
	else ifeq ($(CC), icc)
		OMP = -qopenmp
	else ifeq ($(CC), pgcc)
		OMP = -mp
	else
		OMP = 
	endif
else
	OMP = 
endif


ifndef BLOCK_SIZE
	BLOCK_SIZE = 8
endif
# BLOCKED_NAME = benchmark-blocked-$(BLOCK_SIZE)


ifeq ($(CC), pgcc)
	CFLAGS = -c9x $(OPT) $(SIMD) $(OMP) -DBLOCK_SIZE=$(BLOCK_SIZE)
else
	CFLAGS = -Wall -std=gnu99 $(OPT) $(SIMD) $(OMP) -DBLOCK_SIZE=$(BLOCK_SIZE)
endif


#MKLROOT = /opt/intel/composer_xe_2013.1.117/mkl
#MKLROOT = /opt/intel/system_studio_2019/mkl
#LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm
LDLIBS = -lrt  -I$(MKLROOT)/include -Wl,-L$(MKLROOT)/lib/intel64/ -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl $(OMP)

# targets = benchmark-naive benchmark-blocked benchmark-blas
targets = benchmark-naive benchmark-blocked-$(BLOCK_SIZE) benchmark-blas
objects = benchmark.o dgemm-naive.o dgemm-blocked.o dgemm-blas.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)
#	echo $(MKLROOT)

benchmark-naive : benchmark.o dgemm-naive.o 
	$(CC) -o $@ $^ $(LDLIBS)
# benchmark-blocked : benchmark.o dgemm-blocked.o
benchmark-blocked-$(BLOCK_SIZE) : benchmark.o dgemm-blocked.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blas : benchmark.o dgemm-blas.o
	$(CC) -o $@ $^ $(LDLIBS)

%.o : %.c
	$(CC) -c $(CFLAGS) $<

.PHONY : clean
clean:
	rm -f $(targets) $(objects) *.stdout
