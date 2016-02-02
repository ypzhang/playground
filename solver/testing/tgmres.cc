#include <stdlib.h>                         // System includes
#include <iostream>                       // 

#include "compcol_double.h"                 // Compressed column matrix header
#include "iohb_double.h"                    // Harwell-Boeing matrix I/O header
#include "mvblasd.h"                        // MV_Vector level 1 BLAS
#include "diagpre_double.h"                 // Diagonal preconditioner
#include "ilupre_double.h"

#include MATRIX_H                           // dense matrix header
#include "gmres.h"                          // IML++ GMRES template

using namespace std;
int
main(int argc, char * argv[])
{
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " HBfile " << std::endl;
    exit(-1);
  }

  double tol = 1.e-6;                       // Convergence tolerance
  int result, maxit = 150, restart = 32;    // Maximum, restart iterations

  CompCol_Mat_double A;                     // Create a matrix
  readHB_mat(argv[1], &A);                  // Read matrix data
  VECTOR_double b, x(A.dim(1), 0.0);        // Create rhs, solution vectors
  readHB_rhs(argv[1], &b);                  // Read rhs data

  MATRIX_double H(restart+1, restart, 0.0); // storage for upper Hessenberg H

  //  DiagPreconditioner_double D(A);           // Create diagonal preconditioner
  //  result = GMRES(A, x, b, D, H, restart, maxit, tol);  // Solve system

  CompCol_ILUPreconditioner_double ILU(A);
  result = GMRES(A, x, b, ILU, H, restart, maxit, tol);  // Solve system



  cout << "GMRES flag = " << result << endl;
  cout << "iterations performed: " << maxit << endl;
  cout << "tolerance achieved  : " << tol << endl;

  return result;
}
