static char help[] = "Checks PCAIR returns the operators it was given with their reference counts intact.\n\n";

/*
  Builds a one-dimensional upwind advection operator, hands it to PCAIR as Amat
  with a separate (duplicated) Pmat, and checks after the solver has been
  destroyed that both matrices are still alive and hold exactly the one
  reference this driver owns.

  Any extra reference taken by PCAIR that is never given back leaks the caller's
  matrix; any reference given back that PCAIR never took frees the caller's
  matrix out from under it.  Both are caught here.
*/

#include <petscksp.h>
#include "pflare.h"

// Assemble the upwinded operator into mat, in the COO format so that assembly
// happens on the device when needed
static PetscErrorCode AssembleAdvection(Mat mat, PetscBool preallocate)
{
  PetscInt    i, local_size, global_row_start, global_row_end_plus_one, start_assign;
  PetscInt    *oor, *ooc;
  PetscScalar *v;
  PetscCount  counter;

  PetscFunctionBeginUser;
  PetscCall(MatGetLocalSize(mat, &local_size, NULL));
  PetscCall(MatGetOwnershipRange(mat, &global_row_start, &global_row_end_plus_one));

  PetscCall(PetscMalloc2(2 * local_size, &oor, 2 * local_size, &ooc));
  PetscCall(PetscMalloc1(2 * local_size, &v));

  counter = 0;
  // Dirichlet condition on left boundary
  if (global_row_start == 0) {
    start_assign = 1;
    oor[counter] = 0;
    ooc[counter] = 0;
    v[counter]   = 1.0;
    counter      = counter + 1;
  } else {
    start_assign = global_row_start;
  }

  for (i = start_assign; i < global_row_end_plus_one; i++) {
    // Upwinded dimensionless uniform grid finite difference operator
    oor[counter] = i;
    ooc[counter] = i - 1;
    v[counter]   = -1.0;

    oor[counter + 1] = i;
    ooc[counter + 1] = i;
    v[counter + 1]   = 1.0;

    counter = counter + 2;
  }

  if (preallocate) PetscCall(MatSetPreallocationCOO(mat, counter, oor, ooc));
  PetscCall(PetscFree2(oor, ooc));
  PetscCall(MatSetValuesCOO(mat, v, INSERT_VALUES));
  PetscCall(PetscFree(v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Check the matrix is still alive and only this driver holds a reference to it
static PetscErrorCode CheckOwnedAlone(Mat mat, PetscInt n, const char *name)
{
  PetscInt refct, global_rows;

  PetscFunctionBeginUser;
  // If the solver has already freed this out from under us then PETSc catches
  // the dangling object here in a debug build
  PetscCall(PetscObjectGetReference((PetscObject)mat, &refct));
  PetscCheck(refct == 1, PetscObjectComm((PetscObject)mat), PETSC_ERR_LIB,
             "%s has %" PetscInt_FMT " references after the solver was destroyed, expected 1", name, refct);
  // Touch it to be sure it is still a usable matrix
  PetscCall(MatGetSize(mat, &global_rows, NULL));
  PetscCheck(global_rows == n, PetscObjectComm((PetscObject)mat), PETSC_ERR_LIB,
             "%s has %" PetscInt_FMT " rows after the solver was destroyed, expected %" PetscInt_FMT, name, global_rows, n);
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  Vec      x, b;
  Mat      A, P;
  KSP      ksp;
  PC       pc;
  // Small is fine - this checks reference counts, not convergence, but keep it
  // big enough that the default options still build a hierarchy with several levels
  PetscInt n = 50, local_size;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));

  // Register the pflare types
  PCRegister_PFLARE();

  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, n));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x, &b));
  PetscCall(VecGetLocalSize(x, &local_size));
  PetscCall(VecSet(b, 0.0));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, local_size, local_size, n, n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(AssembleAdvection(A, PETSC_TRUE));

  // A separate preconditioner matrix - PCAIR must not confuse the two
  PetscCall(MatCreate(PETSC_COMM_WORLD, &P));
  PetscCall(MatSetSizes(P, local_size, local_size, n, n));
  PetscCall(MatSetFromOptions(P));
  PetscCall(AssembleAdvection(P, PETSC_TRUE));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));
  PetscCall(KSPSetOperators(ksp, A, P));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCAIR));
  PetscCall(KSPSetFromOptions(ksp));

  PetscCall(VecSet(x, 1.0));
  PetscCall(KSPSolve(ksp, b, x));

  // Change the values in place and solve again, so the setup that resets and
  // rebuilds the hierarchy is checked as well as the first one
  PetscCall(AssembleAdvection(A, PETSC_FALSE));
  PetscCall(AssembleAdvection(P, PETSC_FALSE));
  PetscCall(VecSet(x, 1.0));
  PetscCall(KSPSolve(ksp, b, x));

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(KSPDestroy(&ksp));

  PetscCall(CheckOwnedAlone(A, n, "Amat"));
  PetscCall(CheckOwnedAlone(P, n, "Pmat"));

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&P));

  PetscCall(PetscFinalize());
  return 0;
}
