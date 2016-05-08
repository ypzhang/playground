#include "gpusollib.h"

#define PI 3.141592653589793

/* support at most 5 intervals */
double scal[5] = {1.0, 0.1, 0.1, 0.1, 1.0};

/*------------------------------------------*
 * input:
 * p: of size m*nintv p1 = m, p2 = nintv
 * intv: intervals
 * nintv: len(intv)-1
 * output:
 * q: of size (m+1)*nintv q1=(m+1), q2=nintv
 *-------------------------------------------*/
void xmul(double *p, int p1, int p2,
          double *intv, int nintv,
          double *q, int q1, int q2) {
  int i,j;
  double c,h,h2;

  if (p1 == 0) return;

  for (i=0; i<nintv; i++) {
    c = 0.5 * (intv[i+1] + intv[i]);
    h = 0.5 * (intv[i+1] - intv[i]);

    for (j=0; j<p1; j++)
      q[j*q2+i] = c * p[j*p2+i];

    q[p1*q2+i] = 0.0;

    q[q2+i] += h * p[i];

    h2 = 0.5 * h;

    for (j=1; j<p1; j++)
    {
      q[(j+1)*q2+i] += h2 * p[j*p2+i];
      q[(j-1)*q2+i] += h2 * p[j*p2+i]; 
    }
  }
}

/*---------------------------------------------*
 * input:
 * p: of size p1*p2
 * q: of size q1*q2
 * output:
 * r
 *---------------------------------------------*/
double dotp(double *p, int p1, int p2, 
          double *q, int q1, int q2) {
  int i,j,m,nintv;
  double r = 0.0;

  m = min(q1, p1);
  nintv = q2;

  if (m != 0)
    for (i=0; i<nintv; i++)
      r += scal[i] * q[i] * p[i];

  for (j=0; j<m; j++)
    for (i=0; i<nintv; i++)
      r += scal[i] * q[j*q2+i] * p[j*p2+i];

  return (r*PI/2.0);
}

/*---------------------------------------------*
 * input:
 * p
 * gam	
 * q
 * output:
 * p (overwrite)
 *---------------------------------------------*/
void polsum(double *p, int p1, int p2, double gam,
            double *q, int q1, int q2) {
  int i,j;

  for (j=0; j<q1; j++)
    for (i=0; i<p2; i++)
      p[j*p2+i] += gam * q[j*q2+i];
}

/*---------------------------------------------*
 * This function computes coefficients of L-S  *
 * polynomial by Stieltjes procedure           *
 * The function to be approximated is 1/x      *
 * i.e., 1/x ~ gamma_1*P_1 + gamma_2*P_2 + ... *
 * alpha, beta define the 3-term recurrence    *
 * input:                                      *
 * nintv = len(intv) -1                        *
 * iter: degree                                *
 * output:                                     *
 * alpha, beta, gamma                          *
 *---------------------------------------------*/
void ggexp(int nintv, double *intv, int iter,
           double *alpha, double *beta, double *gamma) {
  int i,j,k,m;
  double *p0,*ppol,*appol,*qpol,bet1,bet,alp;

  m = 2;
  Malloc(p0, m*nintv, double);

  // init p0
  for (i=0; i<nintv; i++) {
    p0[i] = 1.0;
    p0[nintv+i] = 0.0;
  }

  Calloc(ppol, (m+1+iter)*nintv, double);

  xmul(p0, m, nintv, intv, nintv, ppol, m+1, nintv);

  bet1 = sqrt(dotp(ppol,m+1,nintv,ppol,m+1,nintv));

  for (i=0; i<(m+1)*nintv; i++)
    ppol[i] /= bet1;

  gamma[0] = dotp(ppol, m+1, nintv, p0, m, nintv);
  bet = 0.0;
  beta[0] = bet1;

  Malloc(appol, (m+1+iter)*nintv, double);
  Malloc(qpol,  (m+iter)*nintv,   double);

  j = m;
  for (i=0; i<iter; i++) {
    xmul(ppol, j+1, nintv, intv, nintv, appol, j+2,
         nintv);
    alp = dotp(appol, j+2, nintv, ppol, j+1, nintv);
    alpha[i] = alp;
    polsum(appol, j+2, nintv, -alp, ppol, j+1, nintv);

    if (i > 0)
      polsum(appol,j+2,nintv,-bet,qpol,j,nintv);

    bet = sqrt(dotp(appol, j+2, nintv, appol, j+2,
                    nintv));

    beta[i+1] = bet;

    if (bet == 0) {
      perror("divided by zero\n");
      exit(-1);
    }

    memcpy(qpol, ppol, (j+1)*nintv*sizeof(double));

    for (k=0; k<(j+2)*nintv; k++)
      ppol[k] = appol[k]/bet;

    j++;
    gamma[i+1] = dotp(ppol, j+1, nintv, p0, m, nintv);
  }
}

/*-----------------------------------------*/
void lspol(int deg, int nintv, double *intv,
double *alpha, double *beta, double *gamma){
/*-----------------------------------------*/
  if (nintv > 5) {
    perror("ninterval should <= 5\n");
    exit(-1);
  }
  ggexp(nintv, intv, deg, alpha, beta, gamma);
}

