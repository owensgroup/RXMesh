#pragma once
// RXMesh CUDA->HIP math-library shim.
//
// Aliases the cuBLAS / cuSPARSE / cuSOLVER spellings RXMesh's matrix and
// error-handling code uses to their hip* equivalents. Included from
// util/macros.h on HIP (which defines the CU*_ERROR wrappers over these status
// types). The sparse direct-solver surface (cusolverSp csrchol / csrqr /
// csrlsv*) has no hipSOLVER equivalent and is intentionally NOT aliased here;
// the matrix/solver sources and their tests are excluded from the HIP build.
// This header exists so the foundational headers that merely declare
// handle/status types and the generic dense/sparse API still compile under HIP.

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_complex.h>
#include <hipblas/hipblas.h>
#include <hipsparse/hipsparse.h>
#include <hipsolver/hipsolver.h>

// ---- complex scalar types (used by util.h cuda_type() and the QR solver) ----
#define cuComplex               hipFloatComplex
#define cuDoubleComplex         hipDoubleComplex
#define make_cuComplex          make_hipFloatComplex
#define make_cuDoubleComplex    make_hipDoubleComplex

// ---- handles / status types ----
#define cublasHandle_t          hipblasHandle_t
#define cublasStatus_t          hipblasStatus_t
#define cusparseHandle_t        hipsparseHandle_t
#define cusparseStatus_t        hipsparseStatus_t
#define cusolverStatus_t        hipsolverStatus_t
#define cusolverSpHandle_t      hipsolverHandle_t

// ---- status success / error-string ----
#define CUBLAS_STATUS_SUCCESS   HIPBLAS_STATUS_SUCCESS
#define CUSPARSE_STATUS_SUCCESS HIPSPARSE_STATUS_SUCCESS
#define cublasGetStatusString   hipblasStatusToString
#define cusparseGetErrorString  hipsparseGetErrorString

// ---- cuSOLVER status enums referenced by the error wrapper in macros.h ----
#define CUSOLVER_STATUS_SUCCESS                  HIPSOLVER_STATUS_SUCCESS
#define CUSOLVER_STATUS_NOT_INITIALIZED          HIPSOLVER_STATUS_NOT_INITIALIZED
#define CUSOLVER_STATUS_ALLOC_FAILED             HIPSOLVER_STATUS_ALLOC_FAILED
#define CUSOLVER_STATUS_INVALID_VALUE            HIPSOLVER_STATUS_INVALID_VALUE
#define CUSOLVER_STATUS_ARCH_MISMATCH            HIPSOLVER_STATUS_ARCH_MISMATCH
#define CUSOLVER_STATUS_EXECUTION_FAILED         HIPSOLVER_STATUS_EXECUTION_FAILED
#define CUSOLVER_STATUS_INTERNAL_ERROR           HIPSOLVER_STATUS_INTERNAL_ERROR
#define CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED \
    HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED

// ---- pointer mode ----
#define CUBLAS_POINTER_MODE_HOST   HIPBLAS_POINTER_MODE_HOST
#define CUSPARSE_POINTER_MODE_HOST HIPSPARSE_POINTER_MODE_HOST

// ---- cuBLAS handle/control + level-1 BLAS used by DenseMatrix ----
#define cublasCreate            hipblasCreate
#define cublasDestroy           hipblasDestroy
#define cublasSetStream         hipblasSetStream
#define cublasSetPointerMode    hipblasSetPointerMode
#define cublasSaxpy             hipblasSaxpy
#define cublasDaxpy             hipblasDaxpy
#define cublasCaxpy             hipblasCaxpy
#define cublasZaxpy             hipblasZaxpy
#define cublasSdot              hipblasSdot
#define cublasDdot              hipblasDdot
#define cublasCdotc             hipblasCdotc
#define cublasCdotu             hipblasCdotu
#define cublasZdotc             hipblasZdotc
#define cublasZdotu             hipblasZdotu
#define cublasSasum             hipblasSasum
#define cublasDasum             hipblasDasum
#define cublasScasum            hipblasScasum
#define cublasDzasum            hipblasDzasum
#define cublasSnrm2             hipblasSnrm2
#define cublasDnrm2             hipblasDnrm2
#define cublasScnrm2            hipblasScnrm2
#define cublasDznrm2            hipblasDznrm2
#define cublasSscal             hipblasSscal
#define cublasDscal             hipblasDscal
#define cublasCscal             hipblasCscal
#define cublasCsscal            hipblasCsscal
#define cublasZscal             hipblasZscal
#define cublasZdscal            hipblasZdscal
#define cublasSswap             hipblasSswap
#define cublasDswap             hipblasDswap
#define cublasCswap             hipblasCswap
#define cublasZswap             hipblasZswap
#define cublasIsamax            hipblasIsamax
#define cublasIdamax            hipblasIdamax
#define cublasIsamin            hipblasIsamin
#define cublasIdamin            hipblasIdamin

// ---- common generic dense/sparse enums + types ----
#define cusparseOperation_t        hipsparseOperation_t
#define cusparseMatDescr_t         hipsparseMatDescr_t
#define cusparseSpMatDescr_t       hipsparseSpMatDescr_t
#define cusparseDnMatDescr_t       hipsparseDnMatDescr_t
#define cusparseDnVecDescr_t       hipsparseDnVecDescr_t
#define cusparseSpGEMMDescr_t      hipsparseSpGEMMDescr_t
#define cusparseIndexType_t        hipsparseIndexType_t
#define CUSPARSE_OPERATION_NON_TRANSPOSE       HIPSPARSE_OPERATION_NON_TRANSPOSE
#define CUSPARSE_OPERATION_TRANSPOSE           HIPSPARSE_OPERATION_TRANSPOSE
#define CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE
#define CUSPARSE_INDEX_BASE_ZERO   HIPSPARSE_INDEX_BASE_ZERO
#define CUSPARSE_INDEX_32I         HIPSPARSE_INDEX_32I
#define CUSPARSE_ORDER_COL         HIPSPARSE_ORDER_COL
#define CUSPARSE_MATRIX_TYPE_GENERAL HIPSPARSE_MATRIX_TYPE_GENERAL
#define CUSPARSE_DIAG_TYPE_NON_UNIT  HIPSPARSE_DIAG_TYPE_NON_UNIT
#define CUSPARSE_ACTION_NUMERIC      HIPSPARSE_ACTION_NUMERIC
#define CUSPARSE_SPGEMM_DEFAULT      HIPSPARSE_SPGEMM_DEFAULT
#define CUSPARSE_SPMM_ALG_DEFAULT    HIPSPARSE_SPMM_ALG_DEFAULT
#define CUSPARSE_SPMV_ALG_DEFAULT    HIPSPARSE_MV_ALG_DEFAULT

// ---- cuSPARSE control + generic dense/sparse API used by Dense/SparseMatrix ----
#define cusparseCreate               hipsparseCreate
#define cusparseDestroy              hipsparseDestroy
#define cusparseSetStream            hipsparseSetStream
#define cusparseSetPointerMode       hipsparseSetPointerMode
#define cusparseCreateMatDescr       hipsparseCreateMatDescr
#define cusparseDestroyMatDescr      hipsparseDestroyMatDescr
#define cusparseSetMatType           hipsparseSetMatType
#define cusparseSetMatIndexBase      hipsparseSetMatIndexBase
#define cusparseSetMatDiagType       hipsparseSetMatDiagType
#define cusparseCreateCsr            hipsparseCreateCsr
#define cusparseCsrSetPointers       hipsparseCsrSetPointers
#define cusparseCreateDnMat          hipsparseCreateDnMat
#define cusparseDestroyDnMat         hipsparseDestroyDnMat
#define cusparseCreateDnVec          hipsparseCreateDnVec
#define cusparseDestroyDnVec         hipsparseDestroyDnVec
#define cusparseDestroySpMat         hipsparseDestroySpMat
#define cusparseSpMatGetSize         hipsparseSpMatGetSize
#define cusparseSpMM                 hipsparseSpMM
#define cusparseSpMM_bufferSize      hipsparseSpMM_bufferSize
#define cusparseSpMV                 hipsparseSpMV
#define cusparseSpMV_bufferSize      hipsparseSpMV_bufferSize
#define cusparseSpGEMM_createDescr   hipsparseSpGEMM_createDescr
#define cusparseSpGEMM_destroyDescr  hipsparseSpGEMM_destroyDescr
#define cusparseSpGEMM_workEstimation hipsparseSpGEMM_workEstimation
#define cusparseSpGEMM_compute       hipsparseSpGEMM_compute
#define cusparseSpGEMM_copy          hipsparseSpGEMM_copy

#endif
