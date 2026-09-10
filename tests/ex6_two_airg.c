static char help[] = "Reads a PETSc matrix and sets up two concurrent PCAIR\n\
preconditioners on it, this checks the data structures are per PCAIR.\n\
Also checks the options prefix of each PCAIR reaches its inner PCMG: any\n\
-air1_mg_* / -air2_mg_* options given must be used, and unprefixed -mg_*\n\
options must not be.\n\
\n\
Input arguments:\n\
  -f <input_file> : matrix to load (see $PETSC_DIR/share/petsc/datafiles/matrices)\n\n";

#include <petscksp.h>
#include "pflare.h"

/* Build a single KSP wrapping PCAIR on the given operator. KSPSetUp is called
   explicitly so that the per-PCAIR multigrid hierarchy (and the per-level
   fine/coarse IS views, on the Kokkos path) is constructed up front. Pass a
   distinct strong_threshold per instance so the two PCAIRs produce different
   CF splittings (and therefore different per-level IS views) — that is what
   makes the file-scope-globals bug observable; with identical splittings the
   overwritten view happens to have the same contents and the bug is silent. */
static PetscErrorCode BuildAIRKSP(MPI_Comm comm, Mat A, const char *prefix, PetscReal strong_threshold, KSP *ksp)
{
  PC pc;
  PetscFunctionBeginUser;
  PetscCall(KSPCreate(comm, ksp));
  PetscCall(KSPSetOperators(*ksp, A, A));
  PetscCall(KSPSetTolerances(*ksp, 1e-6, PETSC_DEFAULT, PETSC_DEFAULT, 100));
  PetscCall(KSPGetPC(*ksp, &pc));
  PetscCall(PCSetType(pc, PCAIR));
  PetscCall(PCAIRSetStrongThreshold(pc, strong_threshold));
  if (prefix) PetscCall(KSPSetOptionsPrefix(*ksp, prefix));
  PetscCall(KSPSetFromOptions(*ksp));
  PetscCall(KSPSetUp(*ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* PCAIR must hand its options prefix down to its inner PCMG, whose level KSPs
   are then named <prefix>mg_coarse_ / <prefix>mg_levels_N_ and read their
   options in PCSetUp_MG. Any such prefixed option on the command line must
   therefore have been used by the time the KSPs are set up, and the
   unprefixed -mg_* options must have been ignored. Only options that are
   actually present are checked, so the driver still runs bare. */
static PetscErrorCode CheckPrefixedOption(const char *name, PetscBool expect_used, PetscBool *ok)
{
  PetscBool present, used;
  PetscFunctionBeginUser;
  /* PetscOptionsUsed matches the stored name, which has no leading dash. It
     has to be asked first, as PetscOptionsHasName itself marks the option
     as used */
  PetscCall(PetscOptionsUsed(NULL, name + 1, &used));
  PetscCall(PetscOptionsHasName(NULL, NULL, name, &present));
  if (!present) PetscFunctionReturn(PETSC_SUCCESS);
  if (used != expect_used) {
    *ok = PETSC_FALSE;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Option %s was %s\n", name, used ? "used but should not have been" : "not used"));
  }
  /* An unprefixed option is meant to go unused, drop it so PetscFinalize
     doesn't warn about it */
  if (!expect_used) PetscCall(PetscOptionsClearValue(NULL, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckOptionsPrefixes(PetscBool *ok)
{
  const char *prefixed[] = {"-air1_mg_coarse_ksp_type", "-air1_mg_coarse_ksp_max_it", "-air1_mg_coarse_pc_type",
                            "-air2_mg_coarse_ksp_type", "-air2_mg_coarse_ksp_max_it", "-air2_mg_coarse_pc_type"};
  const char *unprefixed[] = {"-mg_coarse_ksp_type", "-mg_coarse_ksp_max_it", "-mg_coarse_pc_type"};
  size_t      i;
  PetscFunctionBeginUser;
  *ok = PETSC_TRUE;
  for (i = 0; i < sizeof(prefixed) / sizeof(prefixed[0]); i++) PetscCall(CheckPrefixedOption(prefixed[i], PETSC_TRUE, ok));
  for (i = 0; i < sizeof(unprefixed) / sizeof(unprefixed[0]); i++) PetscCall(CheckPrefixedOption(unprefixed[i], PETSC_FALSE, ok));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  Mat                A, A_diff_type;
  Vec                b, x1, x2;
  PetscRandom        rnd;
  PetscViewer        fd;
  char               file[PETSC_MAX_PATH_LEN];
  PetscBool          flg;
  PetscInt           m, n, M, N, one = 1;
  MatType            mtype, mtype_input;
  KSP                ksp1, ksp2;
  KSPConvergedReason reason1, reason2;
  PetscBool          prefixes_ok;
  int                npe;

  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PCRegister_PFLARE();

  PetscCall(PetscOptionsGetString(NULL, NULL, "-f", file, sizeof(file), &flg));
  if (!flg) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Must indicate binary file with the -f option");
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &fd));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatLoad(A, fd));
  PetscCall(PetscViewerDestroy(&fd));

  /* Partition the loaded matrix when in parallel (copy from ex6.c). */
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &npe));
  if (npe != 1) {
    MatPartitioning part;
    IS              is, isrows;
    Mat             A_partitioned;
    PetscCall(MatPartitioningCreate(PETSC_COMM_WORLD, &part));
    PetscCall(MatPartitioningSetAdjacency(part, A));
    PetscCall(MatPartitioningSetNParts(part, npe));
    PetscCall(MatPartitioningSetFromOptions(part));
    PetscCall(MatPartitioningApply(part, &is));
    PetscCall(ISBuildTwoSided(is, NULL, &isrows));
    PetscCall(MatCreateSubMatrix(A, isrows, isrows, MAT_INITIAL_MATRIX, &A_partitioned));
    PetscCall(MatDestroy(&A));
    PetscCall(MatPartitioningDestroy(&part));
    PetscCall(ISDestroy(&is));
    PetscCall(ISDestroy(&isrows));
    A = A_partitioned;
  }

  /* Convert A to the user-requested matrix type (e.g. -mat_type aijkokkos). */
  PetscCall(MatGetLocalSize(A, &m, &n));
  PetscCall(MatGetSize(A, &M, &N));
  PetscCall(MatCreateFromOptions(PETSC_COMM_WORLD, NULL, one, m, n, M, N, &A_diff_type));
  PetscCall(MatAssemblyBegin(A_diff_type, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A_diff_type, MAT_FINAL_ASSEMBLY));

  PetscCall(MatGetType(A, &mtype));
  PetscCall(MatGetType(A_diff_type, &mtype_input));
  if (strcmp(mtype, mtype_input) != 0) {
    PetscCall(MatCopy(A, A_diff_type, DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatDestroy(&A));
    A = A_diff_type;
  } else {
    PetscCall(MatDestroy(&A_diff_type));
  }

  /* Build a random RHS that matches A's vec type. */
  PetscCall(MatCreateVecs(A, &b, &x1));
  PetscCall(VecDuplicate(x1, &x2));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rnd));
  PetscCall(PetscRandomSetFromOptions(rnd));
  PetscCall(VecSetRandom(b, rnd));

  /* Build BOTH KSPs (each with its own PCAIR on A) before applying either.
     Different strong thresholds give different CF splittings — without that,
     the two PCAIRs end up with identical per-level IS views and the bug is
     hidden behind matching contents. */
  PetscCall(BuildAIRKSP(PETSC_COMM_WORLD, A, "air1_", 0.3, &ksp1));
  PetscCall(BuildAIRKSP(PETSC_COMM_WORLD, A, "air2_", 0.8, &ksp2));

  /* Both hierarchies are built, so the prefixed -airN_mg_* options have been
     read by now if the prefix made it down to the inner PCMGs */
  PetscCall(CheckOptionsPrefixes(&prefixes_ok));

  PetscCall(KSPSolve(ksp1, b, x1));
  PetscCall(KSPGetConvergedReason(ksp1, &reason1));

  /* ksp2's solve, the second AIRG application. */
  PetscCall(KSPSolve(ksp2, b, x2));
  PetscCall(KSPGetConvergedReason(ksp2, &reason2));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "ksp1 reason = %d, ksp2 reason = %d, options prefixes %s\n",
                        (int)reason1, (int)reason2, prefixes_ok ? "ok" : "NOT propagated"));

  int exit_code = (reason1 >= 0 && reason2 >= 0 && prefixes_ok) ? 0 : 1;

  PetscCall(KSPDestroy(&ksp1));
  PetscCall(KSPDestroy(&ksp2));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&x1));
  PetscCall(VecDestroy(&x2));
  PetscCall(PetscRandomDestroy(&rnd));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return exit_code;
}
