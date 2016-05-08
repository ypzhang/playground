#include "gpusollib.h"
#include <string.h>
//#include <sys/time.h>
#include <time.h>

#define MAXLINE 500

double wall_timer() {
//  struct timeval tim;
//  gettimeofday(&tim, NULL);
//  double t = tim.tv_sec + tim.tv_usec/1e6;
  return 0.;
}

/*---------------------------------*/
void err_norm(int n, double *x,
double *sol, result_t *r) {
/*---------------------------------*/
  int i;
  double t, enrm=0.0;
/*---------------------------------*/
  for (i=0; i<n; i++) {
    t = x[i]-sol[i];
    enrm += t*t;
  }
  r->enorm = sqrt(enrm);
}

void resd_norm(csr_t *A, double *x,
double *b, result_t *r) {
/*---------------------------------*/
  int i,n;
  double *b2, t,rnrm=0.0;
/*---------------------------------*/
  n = A->n;
  Calloc(b2, n, double);
  spmv_csr_cpu(A, x, b2);
  for (i=0; i<n; i++) {
    t = b2[i]-b[i];
    rnrm += t*t;
  }
  r->rnorm = sqrt(rnrm);
  free(b2);
}

PREC_TYPE get_prectype(char *s) {
  if (strcmp(s, "ilut") == 0) return ILUT;
  if (strcmp(s, "iluk") == 0) return ILUK;
  if (strcmp(s, "bilut") == 0) return BILUT;
  if (strcmp(s, "biluk") == 0) return BILUK;
  if (strcmp(s, "lspoly") == 0) return LSPOLY;
  if (strcmp(s, "mcsor") == 0) return MCSOR;
  if (strcmp(s, "mcilu0") == 0) return MCILU0;
  if (strcmp(s, "ic") == 0) return IC;
  return NOPREC;
}

/*----------------------------------------*/
void read_input(char* fn, options_t *opts) {
  FILE *fp;
  char line[MAXLINE];
  char s1[50],s2[50],s3[50],s4[50],s5[50];
  int lfil,bn,deg,nlan;
  double tol;
  PREC_TYPE prectype;
/*----------------------------------------*/
  memset(s1, 0, 50*sizeof(char));
  memset(s2, 0, 50*sizeof(char));
  memset(s3, 0, 50*sizeof(char));
  memset(s4, 0, 50*sizeof(char));
  memset(s5, 0, 50*sizeof(char));
/*----------------------------------------*/
  fp = fopen(fn, "r");
  if (fp == NULL) {
    printf("Error in opening file %s\n",fn);
    exit(-1);
  }
/*----------------- mat file name */
  fgets(line, MAXLINE, fp);
  assert(line[0] == '#');
  fgets(line, MAXLINE, fp);
  sscanf(line, "%s %s", 
  opts->fmatname, opts->matname);
/*----------------- matrix type */
  fgets(line, MAXLINE, fp);
  assert(line[0] == '#');
  fgets(line, MAXLINE, fp);
  sscanf(line, "%s", s1);
  if (strcmp(s1, "csr") == 0)
    opts->mattype = CSR;
  else if (strcmp(s1, "jad") == 0)
    opts->mattype = JAD;
  else
    opts->mattype = DIA;
/*---------------- prec type */
  fgets(line, MAXLINE, fp);
  assert(line[0] == '#');
  fgets(line, MAXLINE, fp);
  sscanf(line,"%s %s %s %s %s",s1,s2,s3,s4,s5);
  prectype = get_prectype(s1);
  switch (prectype) {
    case ILUT:
    opts->prectype = ILUT;
    Calloc(opts->ilut_opt, 1, ilut_opt_t);
    lfil = atoi(s2); assert(lfil>0);
    opts->ilut_opt->lfil = lfil;
    tol = atof(s3); assert(tol>0 && tol<1);
    opts->ilut_opt->tol = tol;
    break;
    case ILUK:
    opts->prectype = ILUK;
    Calloc(opts->iluk_opt, 1, iluk_opt_t);
    lfil = atoi(s2); assert(lfil>0);
    opts->iluk_opt->lfil = lfil;
    break;
    case MCSOR:
    opts->prectype = MCSOR;
    Calloc(opts->mcsor_opt, 1, mcsor_opt_t);
    opts->mcsor_opt->maxcol = MAXCOL;
    opts->mcsor_opt->omega = 1.0;
    deg = atoi(s2); assert(deg>0);
    opts->mcsor_opt->k = deg;
    if (strcmp(s3, "yes") == 0) {
      opts->mcsor_opt->sp = 1;
      tol = atof(s4); assert(tol>0 && tol<1);
      opts->mcsor_opt->tol = tol;
    }
    else
      opts->mcsor_opt->sp = 0;
    break;
    case MCILU0:
    opts->prectype = MCILU0;
    Calloc(opts->mcilu0_opt, 1, mcilu0_opt_t);
    opts->mcilu0_opt->maxcol = MAXCOL;
    if (strcmp(s2, "yes") == 0) {
      opts->mcilu0_opt->sp = 1;
      tol = atof(s3); assert(tol>0 && tol<1);
      opts->mcilu0_opt->tol = tol;
    }
    else
      opts->mcilu0_opt->sp = 0;
    break;

    case BILUK:
    opts->prectype = BILUK;
    Calloc(opts->bilu_opt, 1, bilu_opt_t);
    lfil = atoi(s2); assert(lfil > 0);
    opts->bilu_opt->lfil = lfil;
    bn = atoi(s3); assert(bn>0);
    opts->bilu_opt->bn = bn;
    if (strcmp(s4, "dd") == 0)
      opts->bilu_opt->dd = 1;
    else
      opts->bilu_opt->dd = 0;
    break;

    case BILUT:
    opts->prectype = BILUT;
    Calloc(opts->bilu_opt, 1, bilu_opt_t);
/*----------------------*/
    lfil = atoi(s2); assert(lfil > 0);
    opts->bilu_opt->lfil = lfil;
/*----------------------*/
    tol = atof(s3);
    assert(tol>0 && tol <1);
    opts->bilu_opt->tol = tol;
/*----------------------*/
    bn = atoi(s4); assert(bn > 0);
    opts->bilu_opt->bn = bn;
/*----------------------*/
    if (strcmp(s5, "dd") == 0)
      opts->bilu_opt->dd = 1;
    else
      opts->bilu_opt->dd = 0;
    break;

    case IC:
    opts->prectype = IC;
    Calloc(opts->ic_opt, 1, ic_opt_t);
    lfil = atoi(s2);
    assert(lfil > 0);
    opts->ic_opt->lfil = lfil;
    tol = atof(s3);
    assert(tol>0 && tol <1);
    opts->ic_opt->tol = tol;
    if (strcmp(s4, "yes")==0)
      opts->ic_opt->modi = 1;
    else
      opts->ic_opt->modi = 0;
    break;

    case LSPOLY:
    opts->prectype = LSPOLY;
    Calloc(opts->lspoly_opt, 1, lspoly_opt_t);
    deg = atoi(s2); assert(deg>0);
    opts->lspoly_opt->deg = deg;
    nlan = atoi(s3); assert(nlan>0);
    opts->lspoly_opt->nlan = nlan;
    break;

    case NOPREC:
    opts->prectype = NOPREC;
/*---- nothing */
    break;
  }
/*------------- acceler type */
  fgets(line, MAXLINE, fp);
  assert(line[0] == '#');
  fgets(line, MAXLINE, fp);
  sscanf(line, "%s", s1);
  if (strcmp(s1, "gmres") == 0)
    opts->solver = GMRES;
  else
    opts->solver = CG;
/*------------ stopping tol */
  fgets(line, MAXLINE, fp);
  assert(line[0] == '#');
  fgets(line, MAXLINE, fp);
  sscanf(line, "%s", s1);
  opts->tol = atof(s1);
/*----------- max iters */
  fgets(line, MAXLINE, fp);
  assert(line[0] == '#');
  fgets(line, MAXLINE, fp);
  sscanf(line, "%s", s1);
  opts->maxits = atoi(s1);
/*---------- kdim */
  fgets(line, MAXLINE, fp);
  assert(line[0] == '#');
  fgets(line, MAXLINE, fp);
  sscanf(line, "%s", s1);
  opts->kdim = atoi(s1);
/*--------- lusol cpu/gpu */
  fgets(line, MAXLINE, fp);
  assert(line[0] == '#');
  fgets(line, MAXLINE, fp);
  sscanf(line, "%s", s1);
  if (strcmp(s1, "cpu") == 0)
    opts->lusolgpu = 0;
  else
    opts->lusolgpu = 1;
/*--------- reordering */
  fgets(line, MAXLINE, fp);
  assert(line[0] == '#');
  fgets(line, MAXLINE, fp);
  sscanf(line, "%s %s", s1,s2);
  if (strcmp(s1, "rcm") == 0)
    opts->reord = RCM;
  else if (strcmp(s1, "mmd") == 0)
    opts->reord = MMD;
  else if (strcmp(s1, "nd") == 0)
    opts->reord = ND;
  else
    opts->reord = NONE;
/*------- diag scaling */
  fgets(line, MAXLINE, fp);
  assert(line[0] == '#');
  fgets(line, MAXLINE, fp);
  sscanf(line, "%s", s1);
  int t = atoi(s1);
  assert(t>=-1 && t<=1);
  opts->ds = t;
  fclose(fp);
}

/*-----------------------------------------------*/
void output_result(matrix_t* mat, options_t *opts) {
/*-----------------------------------------------*/
  char fn[] = "OUT/RESULT";
  FILE *fp = fopen(fn,"w");
  fprintf(fp,"=========================================\n");
  fprintf(fp,"  MATRIX  %s\n",opts->matname);
  fprintf(fp,"-----------------------------------------\n");
  fprintf(fp,"  Size %d,  NonZeros %d\n",
  mat->n, mat->nnz);
  fprintf(fp,"-----------------------------------------\n");
  if (opts->solver == CG)
    fprintf(fp,"  Accelerator CG\n");
  else
    fprintf(fp,"  Accelerator GMRES(%d)\n",opts->kdim);
  fprintf(fp,"-----------------------------------------\n");
  fprintf(fp, "  PRECONDITIONER  ");
  switch (opts->prectype) {
    case ILUT:
      fprintf(fp, "ILUT(%d,%.2f)\n",
      opts->ilut_opt->lfil,
      opts->ilut_opt->tol);
    break;
    case ILUK:
      fprintf(fp, "ILUK(%d)\n",
      opts->iluk_opt->lfil);
    break;
    case MCSOR:
      fprintf(fp, "MCSSOR(%d)\n",
      opts->mcsor_opt->k);
    break;
    case MCILU0:
      fprintf(fp, "MCILU0\n");
    break;
    case BILUK:
      fprintf(fp, "BILUK(%d)\n",
      opts->bilu_opt->lfil);
    break;
    case BILUT:
      fprintf(fp, "BILUT(%d, %.2e)\n",
      opts->bilu_opt->lfil, opts->bilu_opt->tol);
    break;
    case LSPOLY:
      fprintf(fp, "LS-POLY(%d)\n",
      opts->lspoly_opt->deg);
    break;
    case IC:
      if (opts->ic_opt->modi)
        fprintf(fp, "MODIFIED IC");
      else
        fprintf(fp, "IC");
      fprintf(fp, "(%d, %.2e)\n",
      opts->ic_opt->lfil, opts->ic_opt->tol);
    break;
    case NOPREC:
      fprintf(fp, "NO-PRECON\n");
    break;
  }
  fprintf(fp,"=========================================\n");
  fprintf(fp,"  Its        %d\n",   opts->result.niters);
  fprintf(fp,"  P-T        %.2f\n", opts->result.tm_prec);
  fprintf(fp,"  I-T        %.2f\n", opts->result.tm_iter);
  if (opts->prectype == ILUT ||
      opts->prectype == ILUK ||
      opts->prectype == IC) {
  fprintf(fp,"  FillFact   %.2f\n", opts->result.filfact);
  }
  if (opts->prectype == BILUK ||
      opts->prectype == BILUT) {
  fprintf(fp,"  FillFact   %.2f\n", opts->result.bfilfact);
  }
  fprintf(fp,"  ErrNorm    %.2e\n", opts->result.enorm);
  fprintf(fp,"  RsdNorm    %.2e\n", opts->result.rnorm);
  fprintf(fp,"-----------------------------------------\n");
  fprintf(fp,"  MAX-ITS    %d\n",   opts->maxits);
  fprintf(fp,"  TOL        %.2e\n", opts->tol);
  switch (opts->mattype) {
    case CSR:
  fprintf(fp,"  MATRIX     CSR Format\n");
    break;
    case JAD:
  fprintf(fp,"  MATRIX     JAD Format, NJAD %d\n",
  mat->h_jad->njad);
    break;
    case DIA:
  fprintf(fp,"  MATRIX     DIA Format, NDIA %d\n",
  mat->h_dia->ndiags);
  }
  fprintf(fp,"  InitRsd    %.2e\n", 
  opts->result.rnorm0);
  switch (opts->reord) {
    case RCM:
  fprintf(fp,"  REORDER    RCM\n");
    break;
    case MMD:
  fprintf(fp,"  REORDER    MMD\n");
    break;
    case ND:
  fprintf(fp,"  REORDER    ND\n");
    break;
    case NONE:
  fprintf(fp,"  REORDER    NONE\n");
  }

  switch (opts->prectype) {
    case IC:
    case ILUT:
    case ILUK:
    if (opts->lusolgpu) {
  fprintf(fp,"  L/U SOL    GPU\n");
  fprintf(fp,"  LEVEL-L    %d\n", opts->result.llev);
  fprintf(fp,"  LEVEL-U    %d\n", opts->result.ulev);
    }
    else
  fprintf(fp,"  L/U SOL    CPU\n");
    break;

    case MCSOR:
/*
  fprintf(fp,"  MultiCol   Max # of color %d\n",\
  opts->mcsor_opt->maxcol);
*/
  fprintf(fp,"  MultiCol   # of color %d\n",
  opts->result.ncol);
    if (opts->mcsor_opt->sp)
  fprintf(fp,"  MultiCol   w/ Sparsification, tol=%.2e\n",
  opts->mcsor_opt->tol);
    else
  fprintf(fp,"  MultiCol   w/o Sparsification\n");
    break;

    case MCILU0:
/*
  //fprintf(fp,"  MultiCol   Max # of color %d\n",
  opts->mcsor_opt->maxcol);
*/
  fprintf(fp,"  MultiCol   # of color %d\n",
  opts->result.ncol);
    if (opts->mcilu0_opt->sp)
  fprintf(fp,"  MultiCol   w/ Sparsification, tol=%.2e\n",
  opts->mcilu0_opt->tol);
    else
  fprintf(fp,"  MultiCol   w/o Sparsification\n");
    break;

  case BILUK:
  fprintf(fp,"  #BLOCKS    %d\n", opts->bilu_opt->bn);
    if (opts->bilu_opt->dd)
  fprintf(fp,"  DD         METIS\n");
    else
  fprintf(fp,"  DD         NONE\n");
    if (opts->lusolgpu)
  fprintf(fp,"  L/U SOL    GPU\n");
    else
  fprintf(fp,"  L/U SOL    CPU\n");
    break;

  case BILUT:
  fprintf(fp,"  #BLOCKS    %d\n", opts->bilu_opt->bn);
    if (opts->bilu_opt->dd)
  fprintf(fp,"  DD         METIS\n");
    else
  fprintf(fp,"  DD         NONE\n");
    if (opts->lusolgpu)
  fprintf(fp,"  L/U SOL    GPU\n");
    else
  fprintf(fp,"  L/U SOL    CPU\n");
    break;

  case LSPOLY:
  fprintf(fp,"  #Lanczos   %d\n", 
  opts->lspoly_opt->nlan);
  break;

  case NOPREC:
  break;
  }

  if (opts->ds > -1)
  fprintf(fp,"  DIAG SCAL  yes, type %d\n", opts->ds);
  else
  fprintf(fp,"  DIAG SCAL  no\n");

  if (opts->result.niters < opts->maxits)
    printf("  converged in %d steps ...\n",\
    opts->result.niters);
  else
    printf("  NOT converged in %d steps ...\n",\
    opts->maxits);

  printf("results saved in %s ... \n\n", fn);

  fprintf(fp,"-----------------------------------------\n");
  fclose(fp);
}

/*-----------------------------------------*/
void dump_mat_coo(csr_t *A, char *fn) {
  int i,j,n;
  FILE *fp;
  fp = fopen(fn, "w");
  n = A->n;
  for (i=0; i<n; i++)
    for (j=(A->ia)[i]; j<(A->ia)[i+1]; j++)
      fprintf(fp, "%d %d %f\n", 
      i+1, (A->ja)[j-1], (A->a)[j-1]);
  fclose(fp);
}

void check_mc(csr_t *A, int ncol, int *il) {
  double rnrm,*a,t,maxt;
  int i,i1,i2,j,k,n,*ia,*ja,x,y;
  n = A->n;
  a = A->a;
  ia = A->ia;
  ja = A->ja;
  maxt = 0.0;
  x = y = -1;
  for (i=0; i<ncol; i++) {
    i1 = il[i]-1;
    i2 = il[i+1]-1;
    for (j=i1; j<i2; j++) {
/*---- row j's 2 norm */
      rnrm = 0.0;
      for (k=ia[j]; k<ia[j+1]; k++) 
        rnrm += a[k-1]*a[k-1];
      rnrm = sqrt(rnrm);
/*---------*/
      for (k=ia[j]; k<ia[j+1]; k++) {
        if (ja[k-1]-1 >= i1 &&
            ja[k-1]-1 < i2 && ja[k-1]-1 != j) {
          t = fabs(a[k-1]);
          t = t/rnrm;
          if (t > maxt) {
            x = j;
            y = ja[k-1]-1;
            maxt = t;
          }
        }
      }
    }
  }
/*---------*/
  printf("max entry (%d,%d) %.2e\n", x, y, maxt);
}

/*-------------------------------------------------*/
void lduSolCPU(int n, double *b, double *x, csr_t *L,
csr_t *U, double *diag) {
  int i,k,i1,i2;
  int *lia, *lja, *uia, *uja;
  double *la, *ua;
  lia = L->ia;
  lja = L->ja;
  la  = L->a;
  uia = U->ia;
  uja = U->ja;
  ua  = U->a;

  for (i=0; i<n; i++)
    x[i] = b[i];

  // Forward solve. Solve L*x = b
  for (i=0; i<n; i++)
  {
    i1 = lia[i];
    i2 = lia[i+1];
    for (k=i1; k<i2; k++)
      x[i] -= la[k-1]*x[lja[k-1]-1];
  }

  /* Backward slove. Solve x = U^{-1}*x */
  for (i=n-1; i>=0; i--)
  {
    double t = diag[i];
    i1 = uia[i];
    i2 = uia[i+1];
    for (k=i1; k<i2; k++)
      x[i] -= ua[k-1]*x[uja[k-1]-1];

    x[i] = t*x[i];
  }
}

/*--------------------------------------------*/
void test_mcilu0(matrix_t *mat, precon_t *prec, 
options_t *opts, csr_t *h_L, csr_t *h_U,
double *d2) {
/*-----------------------------------------*/
  double *y, *z, *x, *d_y, *d_x,t1,t2;
  int i,n;
/*-----------------------------------------*/
  n = mat->n;
  Malloc(x, n, double);
  Malloc(y, n, double);
  Malloc(z, n, double);
  d_x = (double*)cuda_malloc(n*sizeof(double));
  d_y = (double*)cuda_malloc(n*sizeof(double));
/*-----------------------------------------*/
  srand(200);
  for (i=0; i<n; i++)
    x[i] = ((double)rand())/((double)RAND_MAX);
  memcpyh2d(d_x, x, n*sizeof(double));
  t1 = wall_timer();
  for (i=0; i<100; i++)
    mcilu0op(mat, prec, opts, d_y, d_x);
  memcpyd2h(y, d_y, n*sizeof(double));
  t2 = wall_timer() - t1;
  printf("MC-ILU(0) Timing: GPU-time=%f  ", t2);
/*----- CPU -----*/
  t1 = wall_timer();
  for (i=0; i<100; i++)
    lduSolCPU(n, x, z, h_L, h_U, d2);
  t2 = wall_timer() - t1;
  printf("CPU-time=%f ", t2);
/*---------------*/
  double err = 0.0;
  double nrm = 0.0;
  for (i=0; i<n; i++) {
    nrm += z[i]*z[i];
    err += (z[i]-y[i])*(z[i]-y[i]);
  }
  nrm = sqrt(nrm);
  err = sqrt(err);

  printf("[error = %.5e(%.5e)]\n", err,err/nrm);

  free(y); free(z); free(x);
  cuda_free(d_y); cuda_free(d_x);
}

/*---------------------------------------------*/
void mcprune_lu(lu_t *lu, mcilu0_prec_t *mcilu0) {
/*---------------------------------------------*/
/*   This function drops entries in the L/U    */
/*   according to multi-color reordering       */
/*   L/U is strict lower/upper triang matrix   */
/*   i.e., diagonal blocks in L/U are zero     */
/*---------------------------------------------*/
  int n,i,i1,i2,j,k,ncol,*il,ctrL,ctrU,col;
  lu_t lu2;
  csr_t *L, *U, *L2, *U2;
/*---------------------------------*/
  L = lu->l;
  U = lu->u;
  n = L->n;
  ncol = mcilu0->ncol;
  il = mcilu0->il;
  malloc_lu(n, L->nnz, U->nnz, &lu2);
  L2 = lu2.l;
  U2 = lu2.u;
  ctrL = ctrU = 0;
  L2->ia[0] = U2->ia[0] = 1;
/*-------- loop for each color */
  for (i=0; i<ncol; i++) {
    i1 = il[i]-1;
    i2 = il[i+1]-1;
    for (j=i1; j<i2; j++) {
/*------- L: row j */
      for (k=L->ia[j]; k<L->ia[j+1]; k++) {
        col = L->ja[k-1]-1;
/*-------- if in diag block */
        if (col >= i1 && col <  i2)
          continue;
/*-------- else */
        L2->ja[ctrL] = col+1;
        L2->a[ctrL] = L->a[k-1];
        ctrL++;
      }
      L2->ia[j+1] = ctrL+1;
/*------ U: row j */
      for (k=U->ia[j]; k<U->ia[j+1]; k++) {
        col = U->ja[k-1]-1;
/*-------- if in diag block */
        if (col >= i1 && col <  i2)
          continue;
/*-------- else */
        U2->ja[ctrU] = col+1;
        U2->a[ctrU] = U->a[k-1];
        ctrU++;
      }
      U2->ia[j+1] = ctrU+1;
    }
  }
/*----- resize L2, U2 */
  realloc_csr(L2, ctrL);
  realloc_csr(U2, ctrU);
/*
  printf("L-dropped:%d, U-dropped:%d\n",
  L->nnz-ctrL, U->nnz-ctrU);
*/
/*------ free old L/U */
  free_csr(L);
  free_csr(U);
  lu->l = L2;
  lu->u = U2;
}

/*---------------------------------------------*/
csr_t* mcprune_csr(csr_t *A, mcilu0_prec_t *mcilu0) {
/*---------------------------------------------*/
/*   This function drops entries in  A         */
/*   according to multi-color reordering       */
/*   i.e., diagonal blocks in A are diags      */
/*---------------------------------------------*/
  int n,i,i1,i2,j,k,ncol,*il,ctr,col;
  csr_t *B;
/*---------------------------------*/
  n = A->n;
  ncol = mcilu0->ncol;
  il = mcilu0->il;
  Calloc(B, 1, csr_t);
  malloc_csr(n, A->nnz, B);
  ctr = 0;
  B->ia[0] = 1;
/*-------- loop for each color */
  for (i=0; i<ncol; i++) {
    i1 = il[i]-1;
    i2 = il[i+1]-1;
    for (j=i1; j<i2; j++) {
/*------- A: row j */
      for (k=A->ia[j]; k<A->ia[j+1]; k++) {
        col = A->ja[k-1]-1;
/*-------- if in diag block */
        if (col >= i1 && col <  i2 && col != j)
          continue;
/*-------- else */
        B->ja[ctr] = col+1;
        B->a[ctr] = A->a[k-1];
        ctr++;
      }
      B->ia[j+1] = ctr+1;
    }
  }
/*----- resize B */
  realloc_csr(B, ctr);
  //printf("A-dropped:%d\n", A->nnz-ctr);
  return B;
}

