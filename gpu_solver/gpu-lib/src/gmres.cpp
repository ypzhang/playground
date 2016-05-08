#include "gpusollib.h"

#define PRINT 0
#define PRINT2 0

/*--------------------------------------*
|      FGMRES method
| n     : matrix dimension
| kdim  : max Krylov subspace dimension
| maxits: max iter number
| eps   : relative tolerance
 *--------------------------------------*/
void fgmres(matrix_t *mat, precon_t *prec,
options_t *opts, double *d_x, double *d_b) {
/*-----------------------------------------*/
  int i,j,k,k1,ii,out_flag,in_flag,n;
  double gam, eps1, *VV, *w, eps;
  int Kdim, maxits;
 /*------------------------------------------*/
  n = mat->n;  Kdim = opts->kdim;
  maxits = opts->maxits;  eps = opts->tol;
 /*------------------------------------------*
  |        Initialization                     |
  *------------------------------------------*/
  printf("begin GMRES(%d) ... \n", Kdim);

  // ALIGNMENT OF DEVICE MEM
  int m = ALIGNMENT/sizeof(double);
  int n2 = (n+m-1)/m*m;

  /*-----------------------------*
  |         DEVICE MEM           |
   *-----------------------------*/
  /* Allocate Device mem for matrix Vm */
  VV = (double*) cuda_malloc((Kdim+1)*n2*sizeof(double));
  /* work array */
  w = (double*) cuda_malloc(Kdim*n2*sizeof(double));
  
  /*-----------------------------*
   |           HOST MEM          |
   *-----------------------------*/
  /* Allocate Device mem for Hessenberg matrix H */
  double *HH;
  Malloc(HH, Kdim*(Kdim+1), double);  
  /* Givens rotation */
  double *c;
  Malloc(c, Kdim, double);
  double *s;
  Malloc(s, Kdim, double);
  double *rs;
  Malloc(rs, Kdim+1, double);
  
  /*-----------------------------------*
  |          Iteration                 |
  *------------------------------------*/
  out_flag = TRUE;
  int iters = 0;
  double ro, t;

  /* outer loop */
  while (out_flag)
  {
    /*----------------------*
     |   VV(0,:)= -Ax,      |
     *----------------------*/
    (*(mat->spmv))(mat, d_x, VV, 1);

    /* VV(0,:) = rhs + VV(0,:) */
    CUAXPY(n, 1.0, d_b, 1, VV, 1);
 
    /*----------------------*
     |     Norm(VV(0,:))    |
     *----------------------*/
    ro = CUDOT(n, VV, 1, VV, 1);

    ro = sqrt(ro);

    if (fabs(ro-ZERO) <= EPSILON) {
      out_flag = FALSE;
      break;
    }   
    t = 1.0 / ro;    
        
    /*-------------------*
    |    v1=VV(0,:)*t
     *-------------------*/
    CUSCAL(n, t, VV, 1);

    if (iters == 0) {
      opts->result.rnorm0 = ro;
      //printf("Initial Residual: %e\n", ro);
      eps1 = eps*ro;      
    }

    /* initialize 1-st term of rhs of hessenberg system */
    rs[0] = ro;
    i = -1;
    in_flag = TRUE;
    
    /* Inner loop */
    while (in_flag) {
      i++;
      iters ++;
      
      /*--------------------------*
      |    Right precon operation
      |    w[i]=M^{-1} * VV[i,:];
       *--------------------------*/
      (*(prec->op))(mat, prec, opts, &w[i*n2], &VV[i*n2]);

      /*---------------------------*
      |      VV[i+1,:]=A*w[i]
       *---------------------------*/
      (*(mat->spmv))(mat, &w[i*n2], &VV[(i+1)*n2], 0);

      /*---------------------------*
      |    Modified Gram-schmidt
       *---------------------------*/
      for (j=0; j<=i; j++) {
        HH[IDX2C(i,j,Kdim+1)] = 
        CUDOT(n, &VV[j*n2], 1, &VV[(i+1)*n2], 1);            

        CUAXPY(n, -HH[IDX2C(i,j,Kdim+1)], 
                    &VV[j*n2], 1, &VV[(i+1)*n2], 1);
      }

      /*---------------------------*
       |      norm(VV(i+1,:))
       *---------------------------*/
      t = CUDOT(n, &VV[(i+1)*n2], 1, &VV[(i+1)*n2], 1);

      t = sqrt(t);

      HH[IDX2C(i,i+1,Kdim+1)] = t;
            
      if (fabs(t-ZERO) > EPSILON) {
        t = 1.0 / t;
	/*----------------*
	 |    VV(i+1)*t
	 *----------------*/
	CUSCAL(n, t, &VV[(i+1)*n2], 1);
      }
      
      /* Least square problem of HH */
      if (i !=0 )
        for (k=1; k<=i; k++) {
          k1 = k-1;

          t  = HH[IDX2C(i,k1,Kdim+1)];

          HH[IDX2C(i,k1,Kdim+1)] =
          c[k1]*t + s[k1]*HH[IDX2C(i,k,Kdim+1)];

          HH[IDX2C(i,k, Kdim+1)] =
          -s[k1]*t + c[k1]*HH[IDX2C(i,k,Kdim+1)];
        }

      double Hii  = HH[IDX2C(i,i,Kdim+1)];
      double Hii1 = HH[IDX2C(i,i+1,Kdim+1)];
      
      gam = sqrt(Hii*Hii + Hii1*Hii1);
            
      if (fabs(gam-ZERO) <= EPSILON)
        gam = EPSMAC;
		
      /* next Given's rotation */
      c[i] = Hii  / gam;
      s[i] = Hii1 / gam;
      rs[i+1] = -s[i] * rs[i];
      rs[i]   =  c[i] * rs[i];
      
      /* residue norm */
      HH[IDX2C(i,i,Kdim+1)] = c[i]*Hii + s[i]*Hii1;
      ro = fabs(rs[i+1]);

      if (PRINT2)
        printf("%d %e\n", iters, ro);

      /* test convergence */
      if (i+1 >=Kdim || ro <=eps1 || iters >= maxits)
        in_flag = FALSE;
	
    } /* end of inner loop */
              
    /*-------------------------------*
    | Solve upper triangular system  |
     *-------------------------------*/
    rs[i] = rs[i]/HH[IDX2C(i,i,Kdim+1)];
    for (ii=2; ii<=i+1; ii++)
    {
      k  = i-ii+1;
      k1 = k+1;
      t  = rs[k];
      for (j=k1; j<=i; j++)
        t = t - HH[IDX2C(j,k,Kdim+1)]*rs[j];
	
      rs[k] = t / HH[IDX2C(k,k,Kdim+1)];
    }
        
    /*---------------*
    |  Get solution  
     *---------------*/    
    for (j=0; j<=i; j++)
      CUAXPY(n, rs[j], &w[j*n2], 1, d_x, 1);
          
    if (PRINT)
      printf("iter %d: ro=%e\n", iters, ro);
    
    /* test solution */
    if ( ro<=eps1 || iters >= maxits)
      out_flag = FALSE;
  } /* end of outer loop */

  /* final relative tolerance */  
  //printf("Iters = %d, residue = %e\n", iters, ro*eps/eps1);
  opts->result.niters = iters;

  /*------------------------------------------------------*
  |          Finalization:  Free Memory
   *------------------------------------------------------*/
   cuda_free(VV);
   cuda_free(w);
   free(HH);
   free(c);
   free(s);
   free(rs);
} /* end of fgmres */

