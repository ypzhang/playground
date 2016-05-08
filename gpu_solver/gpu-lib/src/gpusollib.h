#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cublas.h>
#include <assert.h>

#define IDX2C(i,j,ld) (((i)*(ld))+(j))
#define WARP      32
#define HALFWARP  16
#define BLOCKDIM  512
#define ALIGNMENT 512
#define MAXTHREADS (30 * 1024 * 5)
// number of half warp per block
#define NHW 32
// for LU kernel
#define BLOCKDIM2 512
// BLOCKDIM2/HALFWARP
#define NHW2 32 
#define ZERO 0.0
#define TRUE  1
#define FALSE 0
#define EPSILON   1.0e-18
#define EPSMAC    1.0e-16
/*--- max # of diags in DIA */
#define MAXDIAG 60
/*--- max # of color in 
 *--- multi-color reordering*/
#define MAXCOL 100
#include "datatype.h"
#include "protos.h"

