#include "gpusollib.h"

void setup_precon(matrix_t *mat, precon_t *prec, 
options_t *opts) {
  double t1,t2;
/*--------------------------------------------*/
  t1 = wall_timer();
  PREC_TYPE ptype = opts->prectype;
  switch (ptype) {
    case ILUT:
      SetupILUt(mat, prec, opts);
      prec->op = lusol;
    break;
    case ILUK:
      SetupILUK(mat, prec, opts);
      prec->op = lusol;
    break;
    case MCSOR:
      SetupMCSOR(mat, prec, opts);
      prec->op = ssor;
    break;
    case MCILU0:
      SetupMCILU0(mat, prec, opts);
      prec->op = mcilu0op;
    break;
    case BILUK:
      SetupBILUK(mat, prec, opts);
      prec->op = blusol;
    break;
    case BILUT:
      SetupBILUT(mat, prec, opts);
      prec->op = blusol;
    break;
    case IC:
      SetupIC(mat, prec, opts);
      prec->op = lltsol;
    break;
    case LSPOLY:
      SetupLSPOLY(mat, prec, opts);
      prec->op = polyapprox;
    break;
    case NOPREC:
      printf("No Prec ...\n");
      prec->op = noprecop;
    break;
  }
  t2 = wall_timer();
  opts->result.tm_prec = t2-t1;
}

/*----------------------------------------*/
void noprecop(matrix_t *mat, precon_t *prec,
options_t *opts, double *d_x, double *d_b) {
/*----------------------------------------*/
  int n;
/*----------------------------------------*/
  n = mat->n;
  memcpyd2d(d_x, d_b, n*sizeof(double));
}

