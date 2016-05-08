enum ACCE_TYPE {
  GMRES,
  CG
};

enum PREC_TYPE {
  ILUT,
  ILUK,
  BILUK,
  BILUT,
  LSPOLY,
  MCSOR,
  MCILU0,
  IC,
  NOPREC
};

enum MATRIX_TYPE {
  CSR,
  JAD,
  DIA
};

enum REORDER_TYPE {
  NONE,
  RCM,
  ND,
  MMD
};

/*-------------------*/
/*   Matrix Format   */
/*-------------------*/
/*------- COO format */
typedef struct coo_t_ {
  int n;
  int nnz;
  int *ir;
  int *jc;
  double *val;
} coo_t;

/*------- CSR format */
typedef struct csr_t_ {
  int n;
  int nnz;
  int *ia;
  int *ja;
  double *a;
} csr_t;

/*------- JAD format */
typedef struct jad_t_ {
  int n;
  int nnz;
  int *ia;
  int *ja;
  double *a;
  int njad;
  int *perm;
  double *w;
} jad_t;

/*------- DIA format */
typedef struct dia_t_ {
  int n;
  int nnz;
  int ndiags;
  int stride;
  double *diags;
  int *ioff;
} dia_t;

/*----------------------*/
/*    Precons Struct    */
/*----------------------*/
/*------- L-S Poly Prec */
typedef struct lspoly_t_ {
/*----- Stieltjes coeff */
  double *alpha;
  double *beta;
  double *gamma;
/*----- work array */
  double *d_v1;
  double *d_v0;
  double *d_v;
} lspoly_t;

/*--------- ILU prec */
typedef struct lu_t_ {
  csr_t *l;
  csr_t *u;
} lu_t;

/*---- level scheduling */
typedef struct level_t_ {
/*---------- L */
  int nlevL;
  int *jlevL;
  int *ilevL;
/*---------- U */
  int nlevU;
  int *jlevU;
  int *ilevU;
} level_t;

typedef struct ilu_prec_t_ {
  lu_t *h_lu, *d_lu;
/*---- cpu lusol */
  double *h_x;
  double *h_b;
/*--- Level Scheduling */
  level_t *h_lev, *d_lev;
} ilu_prec_t;

typedef struct ic_prec_t_ {
/*----- L^{T} in host */
  csr_t *LT;
/*----- D in host */
  double *D;
/*----- CPU LLT solve */
  double *h_x;
  double *h_b;
/*----- for GPU trian solve */
/*----- save LL^{T} in L/U  */
  lu_t *h_lu, *d_lu;
  level_t *h_lev, *d_lev;
} ic_prec_t;

/*------------*/
/*  block LU  */
/*------------*/
typedef struct bilu_dev_t_ {
/*--------------------------------- 
IN DEVICE MEMORY
================
L/U factors for each block
in csr format. All csrs saved 
in a single 3-array foramt.
===============
double array a:
l1_a,  u1_a,  | l2_a,  u2_a,  | ...
------------
int array ja:
l1_ja, u1_ja, | l2_ja, u2_ja, | ...
------
block i's 'ja' or 'a' starts at
postion nzinterval*i, 
which is the max number of
nz in all nz(li)+nz(ui)
==============
int array ia:
l1_ia, u1_ia, | l2_ia, u2_ia, | ...
------
length of block i's 'ia' is
nrow[i]+1 + nrow[i]+1
=2*nrow[i]+2

level info for all blocks
saved in 2-array: jlev, ilev
array nlev saves # of levels
for each block
-----------------------------------*/
  int nzinterval;
  double *a;
  int *ja;
  int *ia;
  //#rows
  int *nrow;
  //row offset
  int *noff;
  //level 
  int *jlev;
  int *ilev;
  int *nlev;
} bilu_dev_t;

typedef struct bilu_host_t_ {
/*--- # of rows of each block */
  int *nrow;
/*--- row of offset */
  int *noff;
  csr_t *bdiag;
  lu_t *blu;
  level_t *blev;
} bilu_host_t;

typedef struct bilu_prec_t_ {
  int nb;
  bilu_host_t *host;
  bilu_dev_t *dev;
/*---- cpu blusol */
  double *h_x;
  double *h_b;
} bilu_prec_t;

/*-- Multi-color SOR Prec */
typedef struct mcsor_prec_t_ {
/*---- num of color found */
  int ncol;
  int *kolrs;
  int *il;
/*---- work array */
  double *d_w;
/*---- diag(i) = 1/D(i) */
  double *d_diag;
} mcsor_prec_t;

typedef struct mcilu0_t_ {
  int ncol;
  int *kolrs;
  int *il;
/*----- L/U factors */
  lu_t *h_lu, *d_lu;
/*---- diag(i) = 1/D(i) */
  double *h_diag;
  double *d_diag;
/*----- work array */
  double *d_w;
} mcilu0_prec_t;

typedef struct result_t_ {
/*------ number of iters */
  int niters;
/*------ initial residue */
  double rnorm0;
/*------ residue norm */
  double rnorm;
/*------ error norm: ||sol-x|| */
  double enorm;
/*------ precon time */
  double tm_prec;
/*------ iter time */
  double tm_iter;
/*------ fill-factor */
  double filfact;
/*------ num of color */
  int ncol;
/*------ bilu avg fill-fact */
  double bfilfact;
/*------ metis time */
  double tm_dd;
/*------ num of lev */
  int ulev;
  int llev;
} result_t;

typedef struct ilut_opt_t_ {
  int lfil;
  double tol;
} ilut_opt_t;

typedef struct iluk_opt_t_ {
  int lfil;
} iluk_opt_t;

typedef struct ic_opt_t_ {
  int lfil;
  double tol;
  int modi;
} ic_opt_t;

typedef struct mcsor_opt_t_ {
/*--- max number of colors */
  int maxcol;
/*---- omega of SOR */
  double omega;
/*---- SOR(k) */
  int k;
/*---- sparsification*/
  int sp;
/*---- spars drop tol*/
  double tol;
} mcsor_opt_t;

typedef struct mcilu0_opt_t_ {
/*--- max number of colors */
  int maxcol;
/*---- sparsification*/
  int sp;
/*---- spars drop tol*/
  double tol;
} mcilu0_opt_t;

typedef struct bilu_opt_t_ {
/*---- num of blocks */
  int bn;
/*---- domain decomp */
  int dd;
/*---- level of fill
  ---- used in biluk bilut */
  int lfil;
/*---- drop tol
  ---- used in bilut */
  double tol;
} bilu_opt_t;

typedef struct lspoly_opt_t_ {
/*----- degree of polyn */
  int deg;
/*--- num of lanczos steps*/
/*--- for eig. value estimation */
  int nlan;
} lspoly_opt_t;

typedef struct options_t_ {
  char fmatname[200];
  char matname[50];
  MATRIX_TYPE mattype;
  ACCE_TYPE solver;
  PREC_TYPE prectype;
  REORDER_TYPE reord;
/*------ gpu lu sol */
  int lusolgpu;
/*--------- diag-scal */
  int ds;
/*------------------ */
/*------ Precon Opts */
/*--------- ilut */
  ilut_opt_t *ilut_opt;
/*--------- iluk */
  iluk_opt_t *iluk_opt;
/*--------- mcsor */
  mcsor_opt_t *mcsor_opt;
/*--------- mcilu0 */
  mcilu0_opt_t *mcilu0_opt;
/*--------- bilu */
  bilu_opt_t *bilu_opt;
/*--------- ic */
  ic_opt_t *ic_opt;
/*------- LS Polyn */
  lspoly_opt_t *lspoly_opt;
/*------------------ */
/*------ accelerator */
  int kdim;
  int maxits;
  double tol;
/*------ result */
  result_t result;
} options_t;

/*--------------------------*
        Matrix Wrapper
 *--------------------------*/
typedef struct matrix_t_ {
/*------------------------- *
  n is the size of matrix
  nnz is the number of non-zeros
  NOTE: nnz may be changed in 
  csr/jad->nnz for padding zeros
 *------------------------- */
  int n, nnz;
/*-------- csr */
  csr_t *h_csr, *d_csr;
/*-------- jad */
  jad_t *h_jad, *d_jad;
/*-------- dia */
  dia_t *h_dia, *d_dia;
/*---- spmv */
  void (*spmv)(struct matrix_t_*,double*,double*,int);
} matrix_t;

/*--------------------------*
    Preconditioner Wrapper
 *--------------------------*/
typedef struct precon_t_ {
/*------ ILU precon */
  ilu_prec_t *ilu;
/*------ L-S poly precon */
  lspoly_t *lspoly;
/*------ Block ILU */
  bilu_prec_t *bilu;
/*------ M-C SOR */
  mcsor_prec_t *mcsor;
/*------ M-C ILU0 */
  mcilu0_prec_t *mcilu0;
/*------ I-C */
  ic_prec_t *ic;
/*------ prec op */
  void (*op) (matrix_t*, struct precon_t_*, 
  options_t*, double*, double*);
} precon_t;

