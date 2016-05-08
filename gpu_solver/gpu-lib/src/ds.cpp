#include "gpusollib.h"

void ds1(csr_t *A, double *b, double *d) {
/*-------------------------------*/
  int n,*ia,*ja,i,j;
  double *a;
/*-------------------------------*/
  ia = A->ia;
  ja = A->ja;
  a = A->a;
  n = A->n;
/*-------------------------------*/
  for (i=0; i<n; i++) {
    d[i] = 0.0;
    for (j=ia[i]; j<ia[i+1]; j++) {
      if (i == ja[j-1]-1) {
/*----- diag element */
        if (a[j-1] != 0.0)
          d[i] = 1.0/sqrt(fabs(a[j-1]));
        break;
      }
    }
    if (d[i] == 0.0) d[i] = 1.0;
  }
/* A:=|D|^{-1/2}*A*|D|^{-1/2} */
  for (i=0; i<n; i++)
    for (j=ia[i]; j<ia[i+1]; j++) {
      a[j-1] *= d[i];
      a[j-1] *= d[ja[j-1]-1];
    }
/*----- b = |D|^{-1/2}*b */
  for (i=0; i<n; i++)
    b[i] *= d[i];
/*--------------------*/
}

int roscal(csr_t *A, double *diag) {
/*----------------------------------
| This routine scales each row of A 
|  so that the 2-norm is 1.
| on return
| diag  = diag[j] = 1/norm(row[j])
|----------------------------------*/
  int i,j,n,*ia,*ja;
  double *a, scal;
/*----------------------*/
  ia = A->ia;
  ja = A->ja;
   a = A->a;
   n = A->n;
/*----------------------*/
  for (i=0; i<n; i++) {
    scal = 0.0;
    for (j=ia[i]; j<ia[i+1]; j++)
      scal += a[j-1]*a[j-1];
    scal = sqrt(scal);
    if (scal == 0.0)
      scal = 1.0; 
    else 
      scal = 1.0 / scal;
    diag[i] = scal;
    for (j=ia[i]; j<ia[i+1]; j++)
      a[j-1] *= scal;
   }
   return 0;
}

int coscal(csr_t *A, double *diag) {
/*----------------------------------
| This routine scales each row of A 
|  so that the 2-norm is 1.
| on return
| diag  = diag[j] = 1/norm(col[j])
|----------------------------------*/
  int i,j,n,*ia,*ja,col;
  double *a, t;
/*----------------------*/
  ia = A->ia;
  ja = A->ja;
   a = A->a;
   n = A->n;
/*----------------------*/
  for (i=0; i<n; i++)
    diag[i] = 0.0;
/*---------------------------
|   2-norm of each column
|----------------------------*/
  for (i=0; i<n; i++)
    for (j=ia[i]; j<ia[i+1]; j++) {
      col = ja[j-1]-1;
      t = a[j-1];
      diag[col] += t*t;
    }
  for (i=0; i<n; i++) {
    if (diag[i] == 0.0)
      diag[i] = 1.0;
    else
      diag[i] = 1.0 / sqrt(diag[i]);
  }
/*---------- A * D */
  for (i=0; i<n; i++) {
    for (j=ia[i]; j<ia[i+1]; j++)
      a[j-1] *= diag[ja[j-1]-1];
  }
  return 0;
}

void ds2(csr_t *A, double *b, double *d) {
/*---- A := D1*A*D2 */
/*---- b := D1*b    */
/*-------------------------------*/
  int i,n;
/*-------------------------------*/
  n = A->n;
/*----- A1 = D1*A ---*/
  roscal(A, d);
/*----- b1 = D1*b ---*/
  for (i=0; i<n; i++)
    b[i] *= d[i];
/*----- A2 = A1*D2 --*/
  coscal(A, d);
}

void diag_scal(csr_t *A, double *b, 
options_t *opts, double **diag) {
  int n;
  double *d;
/*-----------------------*/
  if (opts->ds == -1) { 
    *diag = NULL;
    return;
  }
/*-----------------------*/
  n = A->n;
  Calloc(d, n, double);
  if (opts->ds == 0)
    ds1(A, b, d);
  else
    ds2(A, b, d);
/*-----------------*/
  *diag = d;
  return;
}

void scal_vec(int n, double *diag, double *x) {
/*-------------------------------------*/
  int i;
/*-------------------------------------*/
  if (diag == NULL) return;
/*-------------------------------------*/
  for (i=0; i<n; i++)
    x[i] *= diag[i];
}

