#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include "datatype.h"
#include "protos.h"

#ifdef __cplusplus
extern "C" {
#endif

int cuda_itr_solver(int n, int *Ap, int *Aj, double *Ax, double *h_x, double *h_b);
int cuda_itr_solver_(int n, int *Ap, int *Aj, double *Ax, double *h_x, double *h_b);
int cuda_itr_solver__(int n, int *Ap, int *Aj, double *Ax, double *h_x, double *h_b);

#ifdef __cplusplus
}
#endif

/*-------------------------------*/
int cuda_itr_solver(int n, int *Ap, int *Aj, double *Ax, double *h_x, double *h_b)
{
    /*-------------------------------*/
    /*   driver program of GPU-sol   */
    /*-------------------------------*/
    char finput[] = "input";
    options_t *opts;
    matrix_t *mat;
    precon_t *prec;
    double *d_b,*d_x;

    /*----------------------- init */
    if (gpusol_init()) {
        printf("GPU-Solv Init Error\n");
        return -1;
    }

    /*------------ read input file */
    Calloc(opts, 1, options_t);
    read_input(finput, opts);

    /*----------- first step: ----------*/ 
    /*-- alloc 'mat' & 'prec' structure */
    /*----------------------------------*/
    Calloc(mat,  1, matrix_t);
    Calloc(prec, 1, precon_t);

    /*-------- TODO: setup CSR */
    mat->n = n;
    mat->nnz = Ap[n] - 1;
    Calloc(mat->h_csr,  1, csr_t);
    mat->h_csr->n = n;
    mat->h_csr->nnz = Ap[n] - 1;
    mat->h_csr->ia = Ap;
    mat->h_csr->ja = Aj;
    mat->h_csr->a = Ax;

    /*------ setup mat & copy to device */
    setup_matrix(mat, opts);

    /*------ setpu precon */
    setup_precon(mat, prec, opts);

    /*------ copy rhs to device */
    d_b = (double*)cuda_malloc(n*sizeof(double));
    memcpyh2d(d_b, h_b, n*sizeof(double));

    /*------ copy init guess to device */
    d_x = (double*)cuda_malloc(n*sizeof(double));
    memcpyh2d(d_x, h_x, n*sizeof(double));

    /*------ preconed itersol */
    /*------ GMRES/CG ------- */
    pitsol(mat, prec, opts, d_x, d_b);

    /*------ copy result to host */
    memcpyd2h(h_x, d_x, n*sizeof(double));

    /*------ done, free mem */
    Free(mat->h_csr);
    mat->h_csr = NULL;

    free_opts(opts);
    free_matrix(mat);
    free_precon(prec);

    cuda_free(d_b);
    cuda_free(d_x);

    /*------finalize & check error */
    gpusol_finalize();

    return 0;
}

int cuda_itr_solver_(int n, int *Ap, int *Aj, double *Ax, double *h_x, double *h_b)
{
    return cuda_itr_solver(n, Ap, Aj, Ax, h_x, h_b);
}

int cuda_itr_solver__(int n, int *Ap, int *Aj, double *Ax, double *h_x, double *h_b)
{
    return cuda_itr_solver(n, Ap, Aj, Ax, h_x, h_b);
}
