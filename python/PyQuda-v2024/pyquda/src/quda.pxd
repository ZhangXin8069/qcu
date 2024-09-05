#pragma once

#
# @file  quda.h
# @brief Main header file for the QUDA library
#
# Note to QUDA developers: When adding new members to QudaGaugeParam
# and QudaInvertParam, be sure to update lib/check_params.h as well
# as the Fortran interface in lib/quda_fortran.F90.
#

from enum_quda cimport *
from libc.stdio cimport FILE
# from quda_define cimport *
from quda_constants cimport *

cdef extern from "complex.h":
    pass
ctypedef double complex double_complex

cdef extern from "quda.h":

    #
    # Parameters having to do with the gauge field or the
    # interpretation of the gauge field by various Dirac operators
    #
    ctypedef struct QudaGaugeParam:
        size_t struct_size
        QudaFieldLocation location
        int X[4]
        double anisotropy
        double tadpole_coeff
        double scale
        QudaLinkType type
        QudaGaugeFieldOrder gauge_order
        QudaTboundary t_boundary
        QudaPrecision cpu_prec
        QudaPrecision cuda_prec
        QudaReconstructType reconstruct
        QudaPrecision cuda_prec_sloppy
        QudaReconstructType reconstruct_sloppy
        QudaPrecision cuda_prec_refinement_sloppy
        QudaReconstructType reconstruct_refinement_sloppy
        QudaPrecision cuda_prec_precondition
        QudaReconstructType reconstruct_precondition
        QudaPrecision cuda_prec_eigensolver
        QudaReconstructType reconstruct_eigensolver
        QudaGaugeFixed gauge_fix
        int ga_pad
        int site_ga_pad
        int staple_pad
        int llfat_ga_pad
        int mom_ga_pad
        QudaStaggeredPhase staggered_phase_type
        int staggered_phase_applied
        double i_mu
        int overlap
        int overwrite_gauge
        int overwrite_mom
        int use_resident_gauge
        int use_resident_mom
        int make_resident_gauge
        int make_resident_mom
        int return_result_gauge
        int return_result_mom
        size_t gauge_offset
        size_t mom_offset
        size_t site_size

    #
    # Parameters relating to the solver and the choice of Dirac operator.
    #
    ctypedef struct QudaInvertParam:
        size_t struct_size
        QudaFieldLocation input_location
        QudaFieldLocation output_location
        QudaDslashType dslash_type
        QudaInverterType inv_type
        double mass
        double kappa
        double m5
        int Ls
        double_complex b_5[32]
        double_complex c_5[32]
        double eofa_shift
        int eofa_pm
        double mq1
        double mq2
        double mq3
        double mu
        double tm_rho
        double epsilon
        double evmax
        QudaTwistFlavorType twist_flavor
        int laplace3D
        int covdev_mu
        double tol
        double tol_restart
        double tol_hq
        int compute_true_res
        double true_res
        double true_res_hq
        int maxiter
        double reliable_delta
        double reliable_delta_refinement
        int use_alternative_reliable
        int use_sloppy_partial_accumulator
        int solution_accumulator_pipeline
        int max_res_increase
        int max_res_increase_total
        int max_hq_res_increase
        int max_hq_res_restart_total
        int heavy_quark_check
        int pipeline
        int num_offset
        int num_src
        int num_src_per_sub_partition
        int split_grid[6]
        int overlap
        double offset[32]
        double tol_offset[32]
        double tol_hq_offset[32]
        double true_res_offset[32]
        double iter_res_offset[32]
        double true_res_hq_offset[32]
        double residue[32]
        int compute_action
        double action[2]
        QudaSolutionType solution_type
        QudaSolveType solve_type
        QudaMatPCType matpc_type
        QudaDagType dagger
        QudaMassNormalization mass_normalization
        QudaSolverNormalization solver_normalization
        QudaPreserveSource preserve_source
        QudaPrecision cpu_prec
        QudaPrecision cuda_prec
        QudaPrecision cuda_prec_sloppy
        QudaPrecision cuda_prec_refinement_sloppy
        QudaPrecision cuda_prec_precondition
        QudaPrecision cuda_prec_eigensolver
        QudaDiracFieldOrder dirac_order
        QudaGammaBasis gamma_basis
        QudaFieldLocation clover_location
        QudaPrecision clover_cpu_prec
        QudaPrecision clover_cuda_prec
        QudaPrecision clover_cuda_prec_sloppy
        QudaPrecision clover_cuda_prec_refinement_sloppy
        QudaPrecision clover_cuda_prec_precondition
        QudaPrecision clover_cuda_prec_eigensolver
        QudaCloverFieldOrder clover_order
        QudaUseInitGuess use_init_guess
        double clover_csw
        double clover_coeff
        double clover_rho
        int compute_clover_trlog
        double trlogA[2]
        int compute_clover
        int compute_clover_inverse
        int return_clover
        int return_clover_inverse
        QudaVerbosity verbosity
        int iter
        double gflops
        double secs
        QudaTune tune
        int Nsteps
        int gcrNkrylov
        QudaInverterType inv_type_precondition
        void *preconditioner
        void *deflation_op
        void *eig_param
        QudaBoolean deflate
        QudaDslashType dslash_type_precondition
        QudaVerbosity verbosity_precondition
        double tol_precondition
        int maxiter_precondition
        double omega
        QudaCABasis ca_basis
        double ca_lambda_min
        double ca_lambda_max
        QudaCABasis ca_basis_precondition
        double ca_lambda_min_precondition
        double ca_lambda_max_precondition
        int precondition_cycle
        QudaSchwarzType schwarz_type
        QudaAcceleratorType accelerator_type_precondition
        double madwf_diagonal_suppressor
        int madwf_ls
        int madwf_null_miniter
        double madwf_null_tol
        int madwf_train_maxiter
        QudaBoolean madwf_param_load
        QudaBoolean madwf_param_save
        char madwf_param_infile[256]
        char madwf_param_outfile[256]
        QudaResidualType residual_type
        QudaPrecision cuda_prec_ritz
        int n_ev
        int max_search_dim
        int rhs_idx
        int deflation_grid
        double eigenval_tol
        int eigcg_max_restarts
        int max_restart_num
        double inc_tol
        int make_resident_solution
        int use_resident_solution
        int chrono_make_resident
        int chrono_replace_last
        int chrono_use_resident
        int chrono_max_dim
        int chrono_index
        QudaPrecision chrono_precision
        QudaExtLibType extlib_type
        QudaBoolean native_blas_lapack
        QudaBoolean use_mobius_fused_kernel
        double distance_pc_alpha0
        int distance_pc_t0

    # Parameter set for solving eigenvalue problems.
    ctypedef struct QudaEigParam:
        size_t struct_size
        QudaInvertParam *invert_param
        QudaEigType eig_type
        QudaBoolean use_poly_acc
        int poly_deg
        double a_min
        double a_max
        QudaBoolean preserve_deflation
        void *preserve_deflation_space
        QudaBoolean preserve_evals
        QudaBoolean use_dagger
        QudaBoolean use_norm_op
        QudaBoolean use_pc
        QudaBoolean use_eigen_qr
        QudaBoolean compute_svd
        QudaBoolean compute_gamma5
        QudaBoolean require_convergence
        QudaEigSpectrumType spectrum
        int n_ev
        int n_kr
        int nLockedMax
        int n_conv
        int n_ev_deflate
        double tol
        double qr_tol
        int check_interval
        int max_restarts
        int batched_rotate
        int block_size
        int max_ortho_attempts
        int ortho_block_size
        QudaBoolean arpack_check
        char arpack_logfile[512]
        char QUDA_logfile[512]
        int nk
        int np
        QudaBoolean import_vectors
        QudaPrecision cuda_prec_ritz
        QudaMemoryType mem_type_ritz
        QudaFieldLocation location
        QudaBoolean run_verify
        char vec_infile[256]
        char vec_outfile[256]
        QudaPrecision save_prec
        QudaBoolean io_parity_inflate
        QudaBoolean partfile
        double gflops
        double secs
        QudaExtLibType extlib_type

    ctypedef struct QudaMultigridParam:
        size_t struct_size
        QudaInvertParam *invert_param
        QudaEigParam *eig_param[5]
        int n_level
        int geo_block_size[5][6]
        int spin_block_size[5]
        int n_vec[5]
        QudaPrecision precision_null[5]
        int n_block_ortho[5]
        QudaBoolean block_ortho_two_pass[5]
        QudaVerbosity verbosity[5]
        QudaBoolean setup_use_mma[5]
        QudaBoolean dslash_use_mma[5]
        QudaInverterType setup_inv_type[5]
        int num_setup_iter[5]
        double setup_tol[5]
        int setup_maxiter[5]
        int setup_maxiter_refresh[5]
        QudaCABasis setup_ca_basis[5]
        int setup_ca_basis_size[5]
        double setup_ca_lambda_min[5]
        double setup_ca_lambda_max[5]
        QudaSetupType setup_type
        QudaBoolean pre_orthonormalize
        QudaBoolean post_orthonormalize
        QudaInverterType coarse_solver[5]
        double coarse_solver_tol[5]
        int coarse_solver_maxiter[5]
        QudaCABasis coarse_solver_ca_basis[5]
        int coarse_solver_ca_basis_size[5]
        double coarse_solver_ca_lambda_min[5]
        double coarse_solver_ca_lambda_max[5]
        QudaInverterType smoother[5]
        double smoother_tol[5]
        int nu_pre[5]
        int nu_post[5]
        QudaCABasis smoother_solver_ca_basis[5]
        double smoother_solver_ca_lambda_min[5]
        double smoother_solver_ca_lambda_max[5]
        double omega[5]
        QudaPrecision smoother_halo_precision[5]
        QudaSchwarzType smoother_schwarz_type[5]
        int smoother_schwarz_cycle[5]
        QudaSolutionType coarse_grid_solution_type[5]
        QudaSolveType smoother_solve_type[5]
        QudaMultigridCycleType cycle_type[5]
        QudaBoolean global_reduction[5]
        QudaFieldLocation location[5]
        QudaFieldLocation setup_location[5]
        QudaBoolean use_eig_solver[5]
        QudaBoolean setup_minimize_memory
        QudaComputeNullVector compute_null_vector
        QudaBoolean generate_all_levels
        QudaBoolean run_verify
        QudaBoolean run_low_mode_check
        QudaBoolean run_oblique_proj_check
        QudaBoolean vec_load[5]
        char vec_infile[5][256]
        QudaBoolean vec_store[5]
        char vec_outfile[5][256]
        QudaBoolean mg_vec_partfile[5]
        QudaBoolean coarse_guess
        QudaBoolean preserve_deflation
        double gflops
        double secs
        double mu_factor[5]
        QudaTransferType transfer_type[5]
        QudaBoolean allow_truncation
        QudaBoolean staggered_kd_dagger_approximation
        QudaBoolean thin_update_only

    ctypedef struct QudaGaugeObservableParam:
        size_t struct_size
        QudaBoolean su_project
        QudaBoolean compute_plaquette
        double plaquette[3]
        QudaBoolean compute_polyakov_loop
        double ploop[2]
        QudaBoolean compute_gauge_loop_trace
        double_complex *traces
        int **input_path_buff
        int *path_length
        double *loop_coeff
        int num_paths
        int max_length
        double factor
        QudaBoolean compute_qcharge
        double qcharge
        double energy[3]
        QudaBoolean compute_qcharge_density
        void *qcharge_density
        QudaBoolean remove_staggered_phase

    ctypedef struct QudaGaugeSmearParam:
        size_t struct_size
        unsigned int n_steps
        double epsilon
        double alpha
        double rho
        double alpha1
        double alpha2
        double alpha3
        unsigned int meas_interval
        QudaGaugeSmearType smear_type
        QudaBoolean restart
        double t0
        int dir_ignore

    ctypedef struct QudaBLASParam:
        size_t struct_size
        QudaBLASType blas_type
        QudaBLASOperation trans_a
        QudaBLASOperation trans_b
        int m
        int n
        int k
        int lda
        int ldb
        int ldc
        int a_offset
        int b_offset
        int c_offset
        int a_stride
        int b_stride
        int c_stride
        double_complex alpha
        double_complex beta
        int inv_mat_size
        int batch_count
        QudaBLASDataType data_type
        QudaBLASDataOrder data_order

    #
    # Interface functions, found in interface_quda.cpp
    #

    #
    # Set parameters related to status reporting.
    #
    # In typical usage, this function will be called once (or not at
    # all) just before the call to initQuda(), but it's valid to call
    # it any number of times at any point during execution.  Prior to
    # the first time it's called, the parameters take default values
    # as indicated below.
    #
    # @param verbosity  Default verbosity, ranging from QUDA_SILENT to
    #                   QUDA_DEBUG_VERBOSE.  Within a solver, this
    #                   parameter is overridden by the "verbosity"
    #                   member of QudaInvertParam.  The default value
    #                   is QUDA_SUMMARIZE.
    #
    # @param prefix     String to prepend to all messages from QUDA.  This
    #                   defaults to the empty string (""), but you may
    #                   wish to specify something like "QUDA: " to
    #                   distinguish QUDA's output from that of your
    #                   application.
    #
    # @param outfile    File pointer (such as stdout, stderr, or a handle
    #                   returned by fopen()) where messages should be
    #                   printed.  The default is stdout.
    #
    void setVerbosityQuda(QudaVerbosity verbosity, const char prefix[],
                          FILE *outfile)

    #
    # initCommsGridQuda() takes an optional "rank_from_coords" argument that
    # should be a pointer to a user-defined function with this prototype.
    #
    # @param coords  Node coordinates
    # @param fdata   Any auxiliary data needed by the function
    # @return        MPI rank or QMP node ID cooresponding to the node coordinates
    #
    # @see initCommsGridQuda
    #
    ctypedef int (*QudaCommsMap)(const int *coords, void *fdata)

    #
    # @param mycomm User provided MPI communicator in place of MPI_COMM_WORLD
    #

    void qudaSetCommHandle(void *mycomm)

    #
    # Declare the grid mapping ("logical topology" in QMP parlance)
    # used for communications in a multi-GPU grid.  This function
    # should be called prior to initQuda().  The only case in which
    # it's optional is when QMP is used for communication and the
    # logical topology has already been declared by the application.
    #
    # @param nDim   Number of grid dimensions.  "4" is the only supported
    #               value currently.
    #
    # @param dims   Array of grid dimensions.  dims[0]*dims[1]*dims[2]*dims[3]
    #               must equal the total number of MPI ranks or QMP nodes.
    #
    # @param func   Pointer to a user-supplied function that maps coordinates
    #               in the communication grid to MPI ranks (or QMP node IDs).
    #               If the pointer is NULL, the default mapping depends on
    #               whether QMP or MPI is being used for communication.  With
    #               QMP, the existing logical topology is used if it's been
    #               declared.  With MPI or as a fallback with QMP, the default
    #               ordering is lexicographical with the fourth ("t") index
    #               varying fastest.
    #
    # @param fdata  Pointer to any data required by "func" (may be NULL)
    #
    # @see QudaCommsMap
    #

    void initCommsGridQuda(int nDim, const int *dims, QudaCommsMap func, void *fdata)

    #
    # Initialize the library.  This is a low-level interface that is
    # called by initQuda.  Calling initQudaDevice requires that the
    # user also call initQudaMemory before using QUDA.
    #
    # @param device CUDA device number to use.  In a multi-GPU build,
    #               this parameter may either be set explicitly on a
    #               per-process basis or set to -1 to enable a default
    #               allocation of devices to processes.
    #
    void initQudaDevice(int device)

    #
    # Initialize the library persistant memory allocations (both host
    # and device).  This is a low-level interface that is called by
    # initQuda.  Calling initQudaMemory requires that the user has
    # previously called initQudaDevice.
    #
    void initQudaMemory()

    #
    # Initialize the library.  This function is actually a wrapper
    # around calls to initQudaDevice() and initQudaMemory().
    #
    # @param device  CUDA device number to use.  In a multi-GPU build,
    #                this parameter may either be set explicitly on a
    #                per-process basis or set to -1 to enable a default
    #                allocation of devices to processes.
    #
    void initQuda(int device)

    #
    # Finalize the library.
    #
    void endQuda()

    #
    # @brief update the radius for halos.
    # @details This should only be needed for automated testing when
    # different partitioning is applied within a single run.
    #
    void updateR()

    #
    # A new QudaGaugeParam should always be initialized immediately
    # after it's defined (and prior to explicitly setting its members)
    # using this function.  Typical usage is as follows:
    #
    #   QudaGaugeParam gauge_param = newQudaGaugeParam()
    #
    QudaGaugeParam newQudaGaugeParam()

    #
    # A new QudaInvertParam should always be initialized immediately
    # after it's defined (and prior to explicitly setting its members)
    # using this function.  Typical usage is as follows:
    #
    #   QudaInvertParam invert_param = newQudaInvertParam()
    #
    QudaInvertParam newQudaInvertParam()

    #
    # A new QudaMultigridParam should always be initialized immediately
    # after it's defined (and prior to explicitly setting its members)
    # using this function.  Typical usage is as follows:
    #
    #   QudaMultigridParam mg_param = newQudaMultigridParam()
    #
    QudaMultigridParam newQudaMultigridParam()

    #
    # A new QudaEigParam should always be initialized immediately
    # after it's defined (and prior to explicitly setting its members)
    # using this function.  Typical usage is as follows:
    #
    #   QudaEigParam eig_param = newQudaEigParam()
    #
    QudaEigParam newQudaEigParam()

    #
    # A new QudaGaugeObservableParam should always be initialized
    # immediately after it's defined (and prior to explicitly setting
    # its members) using this function.  Typical usage is as follows:
    #
    #   QudaGaugeObservalbeParam obs_param = newQudaGaugeObservableParam();
    #
    QudaGaugeObservableParam newQudaGaugeObservableParam()

    #
    # A new QudaGaugeSmearParam should always be initialized
    # immediately after it's defined (and prior to explicitly setting
    # its members) using this function.  Typical usage is as follows:
    #
    #   QudaGaugeSmearParam smear_param = newQudaGaugeSmearParam();
    #
    QudaGaugeSmearParam newQudaGaugeSmearParam()

    #
    # A new QudaBLASParam should always be initialized immediately
    # after it's defined (and prior to explicitly setting its members)
    # using this function.  Typical usage is as follows:
    #
    #   QudaBLASParam blas_param = newQudaBLASParam()
    #
    QudaBLASParam newQudaBLASParam()

    #
    # Print the members of QudaGaugeParam.
    # @param param The QudaGaugeParam whose elements we are to print.
    #
    void printQudaGaugeParam(QudaGaugeParam *param)

    #
    # Print the members of QudaInvertParam.
    # @param param The QudaInvertParam whose elements we are to print.
    #
    void printQudaInvertParam(QudaInvertParam *param)

    #
    # Print the members of QudaMultigridParam.
    # @param param The QudaMultigridParam whose elements we are to print.
    #
    void printQudaMultigridParam(QudaMultigridParam *param)

    #
    # Print the members of QudaEigParam.
    # @param param The QudaEigParam whose elements we are to print.
    #
    void printQudaEigParam(QudaEigParam *param)

    #
    # Print the members of QudaGaugeObservableParam.
    # @param param The QudaGaugeObservableParam whose elements we are to print.
    #
    void printQudaGaugeObservableParam(QudaGaugeObservableParam *param)

    #
    # Print the members of QudaBLASParam.
    # @param param The QudaBLASParam whose elements we are to print.
    #
    void printQudaBLASParam(QudaBLASParam *param)

    #
    # Load the gauge field from the host.
    # @param h_gauge Base pointer to host gauge field (regardless of dimensionality)
    # @param param   Contains all metadata regarding host and device storage
    #
    void loadGaugeQuda(void *h_gauge, QudaGaugeParam *param)

    #
    # Free QUDA's internal copy of the gauge field.
    #
    void freeGaugeQuda()

    #
    # Free a unique type (Wilson, HISQ fat, HISQ long, smeared) of internal gauge field.
    # @param link_type[in] Type of link type to free up
    #
    void freeUniqueGaugeQuda(QudaLinkType link_type)

    #
    # Free QUDA's internal smeared gauge field.
    #
    void freeGaugeSmearedQuda()

    #
    # Free QUDA's internal two-link gauge field.
    #
    void freeGaugeTwoLinkQuda()

    #
    # Save the gauge field to the host.
    # @param h_gauge Base pointer to host gauge field (regardless of dimensionality)
    # @param param   Contains all metadata regarding host and device storage
    #
    void saveGaugeQuda(void *h_gauge, QudaGaugeParam *param)

    #
    # Load the clover term and/or the clover inverse from the host.
    # Either h_clover or h_clovinv may be set to NULL.
    # @param h_clover    Base pointer to host clover field
    # @param h_cloverinv Base pointer to host clover inverse field
    # @param inv_param   Contains all metadata regarding host and device storage
    #
    void loadCloverQuda(void *h_clover, void *h_clovinv,
                        QudaInvertParam *inv_param)

    #
    # Free QUDA's internal copy of the clover term and/or clover inverse.
    #
    void freeCloverQuda()

    #
    # Perform the solve, according to the parameters set in param.  It
    # is assumed that the gauge field has already been loaded via
    # loadGaugeQuda().
    # @param h_x    Solution spinor field
    # @param h_b    Source spinor field
    # @param param  Contains all metadata regarding host and device
    #               storage and solver parameters
    #
    void lanczosQuda(int k0, int m, void *hp_Apsi, void *hp_r, void *hp_V, void *hp_alpha, void *hp_beta,
                     QudaEigParam *eig_param)

    #
    # Perform the eigensolve. The problem matrix is defined by the invert param, the
    # mode of solution is specified by the eig param. It is assumed that the gauge
    # field has already been loaded via  loadGaugeQuda().
    # @param h_evecs  Array of pointers to application eigenvectors
    # @param h_evals  Host side eigenvalues
    # @param param Contains all metadata regarding the type of solve.
    #
    void eigensolveQuda(void **h_evecs, double_complex *h_evals, QudaEigParam *param)

    #
    # Perform the solve, according to the parameters set in param.  It
    # is assumed that the gauge field has already been loaded via
    # loadGaugeQuda().
    # @param h_x    Solution spinor field
    # @param h_b    Source spinor field
    # @param param  Contains all metadata regarding host and device
    #               storage and solver parameters
    #
    void invertQuda(void *h_x, void *h_b, QudaInvertParam *param)

    #
    # @brief Perform the solve like @invertQuda but for multiple rhs by spliting the comm grid into
    # sub-partitions: each sub-partition invert one or more rhs'.
    # The QudaInvertParam object specifies how the solve should be performed on each sub-partition.
    # Unlike @invertQuda, the interface also takes the host side gauge as input. The gauge pointer and
    # gauge_param are used if for inv_param split_grid[0] * split_grid[1] * split_grid[2] * split_grid[3]
    # is larger than 1, in which case gauge field is not required to be loaded beforehand; otherwise
    # this interface would just work as @invertQuda, which requires gauge field to be loaded beforehand,
    # and the gauge field pointer and gauge_param are not used.
    # @param _hp_x       Array of solution spinor fields
    # @param _hp_b       Array of source spinor fields
    # @param param       Contains all metadata regarding host and device storage and solver parameters
    #
    void invertMultiSrcQuda(void **_hp_x, void **_hp_b, QudaInvertParam *param)

    #
    # Solve for multiple shifts (e.g., masses).
    # @param _hp_x    Array of solution spinor fields
    # @param _hp_b    Source spinor fields
    # @param param  Contains all metadata regarding host and device
    #               storage and solver parameters
    #
    void invertMultiShiftQuda(void **_hp_x, void *_hp_b, QudaInvertParam *param)

    #
    # Setup the multigrid solver, according to the parameters set in param.  It
    # is assumed that the gauge field has already been loaded via
    # loadGaugeQuda().
    # @param param  Contains all metadata regarding host and device
    #               storage and solver parameters
    #
    void* newMultigridQuda(QudaMultigridParam *param)

    #
    # @brief Free resources allocated by the multigrid solver
    # @param mg_instance Pointer to instance of multigrid_solver
    # @param param Contains all metadata regarding host and device
    # storage and solver parameters
    #
    void destroyMultigridQuda(void *mg_instance)

    #
    # @brief Updates the multigrid preconditioner for the new gauge / clover field
    # @param mg_instance Pointer to instance of multigrid_solver
    # @param param Contains all metadata regarding host and device
    # storage and solver parameters, of note contains a flag specifying whether
    # to do a full update or a thin update.
    #
    void updateMultigridQuda(void *mg_instance, QudaMultigridParam *param)

    #
    # @brief Dump the null-space vectors to disk
    # @param[in] mg_instance Pointer to the instance of multigrid_solver
    # @param[in] param Contains all metadata regarding host and device
    # storage and solver parameters (QudaMultigridParam::vec_outfile
    # sets the output filename prefix).
    #
    void dumpMultigridQuda(void *mg_instance, QudaMultigridParam *param)

    #
    # Apply the Dslash operator (D_{eo} or D_{oe}).
    # @param[out] h_out  Result spinor field
    # @param[in] h_in   Input spinor field
    # @param[in] param  Contains all metadata regarding host and device
    #               storage
    # @param[in] parity The destination parity of the field
    #
    void dslashQuda(void *h_out, void *h_in, QudaInvertParam *inv_param, QudaParity parity)

    #
    # Apply the covariant derivative.
    # @param[out] h_out  Result spinor field
    # @param[in] h_in   Input spinor field
    # @param[in] dir    Direction of application
    # @param[in] param  Metadata for host and device storage
    #
    void covDevQuda(void *h_out, void *h_in, int dir, QudaInvertParam *param)

    #
    # Apply the covariant derivative.
    # @param[out] h_out  Result spinor field
    # @param[in] h_in   Input spinor field
    # @param[in] dir    Direction of application
    # @param[in] sym    Apply forward=2, backward=2 or symmetric=3 shift
    # @param[in] param  Metadata for host and device storage
    #
    void shiftQuda(void *h_out, void *h_in, int dir, int sym, QudaInvertParam *param)

    #
    # Apply the spin-taste operator.
    # @param[out] h_out  Result spinor field
    # @param[in] h_in   Input spinor field
    # @param[in] spin   Spin gamma structure
    # @param[in] taste  Taste gamma structure
    # @param[in] param  Metadata for host and device storage
    #
    void spinTasteQuda(void *h_out, void *h_in, int spin, int taste, QudaInvertParam *param)

    #
    # @brief Perform the solve like @dslashQuda but for multiple rhs by spliting the comm grid into
    # sub-partitions: each sub-partition does one or more rhs'.
    # The QudaInvertParam object specifies how the solve should be performed on each sub-partition.
    # Unlike @invertQuda, the interface also takes the host side gauge as
    # input - gauge field is not required to be loaded beforehand.
    # @param _hp_x       Array of solution spinor fields
    # @param _hp_b       Array of source spinor fields
    # @param param       Contains all metadata regarding host and device storage and solver parameters
    # @param parity      Parity to apply dslash on
    #
    void dslashMultiSrcQuda(void **_hp_x, void **_hp_b, QudaInvertParam *param, QudaParity parity)

    #
    # Apply the clover operator or its inverse.
    # @param h_out  Result spinor field
    # @param h_in   Input spinor field
    # @param param  Contains all metadata regarding host and device
    #               storage
    # @param parity The source and destination parity of the field
    # @param inverse Whether to apply the inverse of the clover term
    #
    void cloverQuda(void *h_out, void *h_in, QudaInvertParam *inv_param, QudaParity parity, int inverse)

    #
    # Apply the full Dslash matrix, possibly even/odd preconditioned.
    # @param h_out  Result spinor field
    # @param h_in   Input spinor field
    # @param param  Contains all metadata regarding host and device
    #               storage
    #
    void MatQuda(void *h_out, void *h_in, QudaInvertParam *inv_param)

    #
    # Apply M^{\dag}M, possibly even/odd preconditioned.
    # @param h_out  Result spinor field
    # @param h_in   Input spinor field
    # @param param  Contains all metadata regarding host and device
    #               storage
    #
    void MatDagMatQuda(void *h_out, void *h_in, QudaInvertParam *inv_param)


    #
    # The following routines are temporary additions used by the HISQ
    # link-fattening code.
    #

    void set_dim(int *)
    void pack_ghost(void **cpuLink, void **cpuGhost, int nFace,
                    QudaPrecision precision)

    void computeKSLinkQuda(void* fatlink, void* longlink, void* ulink, void* inlink,
                           double *path_coeff, QudaGaugeParam *param)

    #
    # Compute two-link field
    #
    # @param[out] twolink computed two-link field
    # @param[in] inlink  the external field
    # @param[in] param  Contains all metadata regarding host and device
    #               storage
    #
    void computeTwoLinkQuda(void *twolink, void *inlink, QudaGaugeParam *param)

    #
    # Either downloads and sets the resident momentum field, or uploads
    # and returns the resident momentum field
    #
    # @param[in,out] mom The external momentum field
    # @param[in] param The parameters of the external field
    #
    void momResidentQuda(void *mom, QudaGaugeParam *param)

    #
    # Compute the gauge force and update the momentum field
    #
    # @param[in,out] mom The momentum field to be updated
    # @param[in] sitelink The gauge field from which we compute the force
    # @param[in] input_path_buf[dim][num_paths][path_length]
    # @param[in] path_length One less that the number of links in a loop (e.g., 3 for a staple)
    # @param[in] loop_coeff Coefficients of the different loops in the Symanzik action
    # @param[in] num_paths How many contributions from path_length different "staples"
    # @param[in] max_length The maximum number of non-zero of links in any path in the action
    # @param[in] dt The integration step size (for MILC this is dt*beta/3)
    # @param[in] param The parameters of the external fields and the computation settings
    #
    int computeGaugeForceQuda(void *mom, void *sitelink, int ***input_path_buf, int *path_length, double *loop_coeff,
                              int num_paths, int max_length, double dt, QudaGaugeParam *qudaGaugeParam)

    #
    # Compute the product of gauge links along a path and add to/overwrite the output field
    #
    # @param[in,out] out The output field to be updated
    # @param[in] sitelink The gauge field from which we compute the products of gauge links
    # @param[in] input_path_buf[dim][num_paths][path_length]
    # @param[in] path_length One less that the number of links in a loop (e.g., 3 for a staple)
    # @param[in] loop_coeff Coefficients of the different loops in the Symanzik action
    # @param[in] num_paths How many contributions from path_length different "staples"
    # @param[in] max_length The maximum number of non-zero of links in any path in the action
    # @param[in] dt The integration step size (for MILC this is dt*beta/3)
    # @param[in] param The parameters of the external fields and the computation settings
    #
    int computeGaugePathQuda(void *out, void *sitelink, int ***input_path_buf, int *path_length, double *loop_coeff,
                             int num_paths, int max_length, double dt, QudaGaugeParam *qudaGaugeParam)

    #
    # Compute the traces of products of gauge links along paths using the resident field
    #
    # @param[in,out] traces The computed traces
    # @param[in] sitelink The gauge field from which we compute the products of gauge links
    # @param[in] path_length The number of links in each loop
    # @param[in] loop_coeff Multiplicative coefficients for each loop
    # @param[in] num_paths Total number of loops
    # @param[in] max_length The maximum number of non-zero of links in any path in the action
    # @param[in] factor An overall normalization factor
    #
    void computeGaugeLoopTraceQuda(double_complex *traces, int **input_path_buf, int *path_length, double *loop_coeff,
                                   int num_paths, int max_length, double factor)

    #
    # Evolve the gauge field by step size dt, using the momentum field
    # I.e., Evalulate U(t+dt) = e(dt pi) U(t)
    #
    # @param gauge The gauge field to be updated
    # @param momentum The momentum field
    # @param dt The integration step size step
    # @param conj_mom Whether to conjugate the momentum matrix
    # @param exact Whether to use an exact exponential or Taylor expand
    # @param param The parameters of the external fields and the computation settings
    #
    void updateGaugeFieldQuda(void* gauge, void* momentum, double dt,
                              int conj_mom, int exact, QudaGaugeParam* param)

    #
    # Apply the staggered phase factors to the gauge field.  If the
    # imaginary chemical potential is non-zero then the phase factor
    # exp(imu/T) will be applied to the links in the temporal
    # direction.
    #
    # @param gauge_h The gauge field
    # @param param The parameters of the gauge field
    #
    void staggeredPhaseQuda(void *gauge_h, QudaGaugeParam *param)

    #
    # Project the input field on the SU(3) group.  If the target
    # tolerance is not met, this routine will give a runtime error.
    #
    # @param gauge_h The gauge field to be updated
    # @param tol The tolerance to which we iterate
    # @param param The parameters of the gauge field
    #
    void projectSU3Quda(void *gauge_h, double tol, QudaGaugeParam *param)

    #
    # Evaluate the momentum contribution to the Hybrid Monte Carlo
    # action.
    #
    # @param momentum The momentum field
    # @param param The parameters of the external fields and the computation settings
    # @return momentum action
    #
    double momActionQuda(void* momentum, QudaGaugeParam* param)

    #
    # Allocate a gauge (matrix) field on the device and optionally download a host gauge field.
    #
    # @param gauge The host gauge field (optional - if set to 0 then the gauge field zeroed)
    # @param geometry The geometry of the matrix field to create (1 - scalar, 4 - vector, 6 - tensor)
    # @param param The parameters of the external field and the field to be created
    # @return Pointer to the gauge field (cast as a void*)
    #
    void* createGaugeFieldQuda(void* gauge, int geometry, QudaGaugeParam* param)

    #
    # Copy the QUDA gauge (matrix) field on the device to the CPU
    #
    # @param outGauge Pointer to the host gauge field
    # @param inGauge Pointer to the device gauge field (QUDA device field)
    # @param param The parameters of the host and device fields
    #
    void  saveGaugeFieldQuda(void* outGauge, void* inGauge, QudaGaugeParam* param)

    #
    # Reinterpret gauge as a pointer to a GaugeField and call destructor.
    #
    # @param gauge Gauge field to be freed
    #
    void destroyGaugeFieldQuda(void* gauge)

    #
    # Compute the clover field and its inverse from the resident gauge field.
    #
    # @param param The parameters of the clover field to create
    #
    void createCloverQuda(QudaInvertParam* param)

    #
    # Compute the clover force contributions from a set of partial
    # fractions stemming from a rational approximation suitable for use
    # within MILC.
    #
    # @param mom Force matrix
    # @param dt Integrating step size
    # @param x Array of solution vectors
    # @param p Array of intermediate vectors
    # @param coeff Array of residues for each contribution (multiplied by stepsize)
    # @param kappa2 -kappa*kappa parameter
    # @param ck -clover_coefficient * kappa / 8
    # @param nvec Number of vectors
    # @param multiplicity Number fermions this bilinear reresents
    # @param gauge Gauge Field
    # @param gauge_param Gauge field meta data
    # @param inv_param Dirac and solver meta data
    #
    void computeCloverForceQuda(void *mom, double dt, void **x, void **p, double *coeff, double kappa2, double ck,
                                int nvector, double multiplicity, void *gauge,
                                QudaGaugeParam *gauge_param, QudaInvertParam *inv_param)

    #
    # Compute the force from a clover or twisted clover determinant or
    # a set of partial fractions stemming from a rational approximation
    # suitable for use from within tmLQCD.
    #
    # @param h_mom Host force matrix
    # @param h_x Array of solution vectors x_i = ( Q^2 + s_i )^{-1} b
    # @param h_x0 Array of source vector necessary to compute the force of a ratio of determinant
    # @param coeff Array of coefficients for the rational approximation or {1.0} for the determinant.
    # @param nvector Number of solution vectors and coefficients
    # @param gauge_param Gauge field meta data
    # @param inv_param Dirac and solver meta data
    # @param detratio if 0 compute the force of a determinant otherwise compute the force from a ratio of determinants
    #
    void computeTMCloverForceQuda(void *h_mom, void **h_x, void **h_x0, double *coeff, int nvector,
                                  QudaGaugeParam *gauge_param, QudaInvertParam *inv_param, int detratio)

    #
    # Compute the naive staggered force.  All fields must be in the same precision.
    #
    # @param mom Momentum field
    # @param dt Integrating step size
    # @param delta Additional scale factor when updating momentum (mom += delta * [force]_TA
    # @param gauge Gauge field (at present only supports resident gauge field)
    # @param x Array of single-parity solution vectors (at present only supports resident solutions)
    # @param gauge_param Gauge field meta data
    # @param invert_param Dirac and solver meta data
    #
    void computeStaggeredForceQuda(void *mom, double dt, double delta, void *gauge, void **x, QudaGaugeParam *gauge_param,
                                   QudaInvertParam *invert_param)

    #
    # Compute the fermion force for the HISQ quark action and integrate the momentum.
    # @param momentum        The momentum field we are integrating
    # @param dt              The stepsize used to integrate the momentum
    # @param level2_coeff    The coefficients for the second level of smearing in the quark action.
    # @param fat7_coeff      The coefficients for the first level of smearing (fat7) in the quark action.
    # @param w_link          Unitarized link variables obtained by applying fat7 smearing and unitarization to the original links.
    # @param v_link          Fat7 link variables.
    # @param u_link          SU(3) think link variables.
    # @param quark           The input fermion field.
    # @param num             The number of quark fields
    # @param num_naik        The number of naik contributions
    # @param coeff           The coefficient multiplying the fermion fields in the outer product
    # @param param.          The field parameters.
    #
    void computeHISQForceQuda(void* momentum,
                              double dt,
                              const double level2_coeff[6],
                              const double fat7_coeff[6],
                              const void* const w_link,
                              const void* const v_link,
                              const void* const u_link,
                              void** quark,
                              int num,
                              int num_naik,
                              double** coeff,
                              QudaGaugeParam* param)

    #
    # @brief Generate Gaussian distributed fields and store in the
    # resident gauge field. We create a Gaussian-distributed su(n)
    # field and exponentiate it, e.g., U = exp(sigma * H), where H is
    # the distributed su(n) field and sigma is the width of the
    # distribution (sigma = 0 results in a free field, and sigma = 1 has
    # maximum disorder).

    # @param seed The seed used for the RNG
    # @param sigma Width of Gaussian distrubution
    #
    void gaussGaugeQuda(unsigned long long seed, double sigma)

    #
    # @brief Generate Gaussian distributed fields and store in the
    # resident momentum field. We create a Gaussian-distributed su(n)
    # field, e.g., sigma * H, where H is the distributed su(n) field
    # and sigma is the width of the distribution (sigma = 0 results
    # in a free field, and sigma = 1 has maximum disorder).
    #
    # @param seed The seed used for the RNG
    # @param sigma Width of Gaussian distrubution
    #
    void gaussMomQuda(unsigned long long seed, double sigma)

    #
    # Computes the total, spatial and temporal plaquette averages of the loaded gauge configuration.
    # @param[out] Array for storing the averages (total, spatial, temporal)
    #
    void plaqQuda(double plaq[3])

    #
    # @brief Computes the trace of the Polyakov loop of the current resident field
    # in a given direction.

    # @param[out] ploop Trace of the Polyakov loop in direction dir
    # @param[in] dir Direction of Polyakov loop
    #
    void polyakovLoopQuda(double ploop[2], int dir)

    #
    # Performs a deep copy from the internal extendedGaugeResident field.
    # @param Pointer to externally allocated GaugeField
    #
    void copyExtendedResidentGaugeQuda(void *resident_gauge)

    #
    # Performs gaussian/Wuppertal smearing on a given spinor using the gauge field
    # gaugeSmeared, if it exist, or gaugePrecise if no smeared field is present.
    # @param h_in   Input spinor field
    # @param h_out  Output spinor field
    # @param param  Contains all metadata regarding host and device
    #               storage and operator which will be applied to the spinor
    # @param n_steps Number of steps to apply.
    # @param coeff  Width of the Gaussian distribution
    # @param smear_type Gaussian/Wuppertal smearing
    #
    void performFermionSmearQuda(void *h_out, void *h_in, QudaInvertParam *param, const int n_steps, const double coeff,
                                const QudaFermionSmearType smear_type)

    #
    # LEGACY
    # Performs Wuppertal smearing on a given spinor using the gauge field
    # gaugeSmeared, if it exist, or gaugePrecise if no smeared field is present.
    # @param h_out  Result spinor field
    # @param h_in   Input spinor field
    # @param param  Contains all metadata regarding host and device
    #               storage and operator which will be applied to the spinor
    # @param n_steps Number of steps to apply.
    # @param alpha  Alpha coefficient for Wuppertal smearing.
    #
    void performWuppertalnStep(void *h_out, void *h_in, QudaInvertParam *param, unsigned int n_steps, double alpha)

    #
    # LEGACY
    # Performs gaussian smearing on a given spinor using the gauge field
    # gaugeSmeared, if it exist, or gaugePrecise if no smeared field is present.
    # @param h_in   Input spinor field
    # @param h_out  Output spinor field
    # @param param  Contains all metadata regarding host and device
    #               storage and operator which will be applied to the spinor
    # @param n_steps Number of steps to apply.
    # @param omega  Width of the Gaussian distribution
    #
    void performGaussianSmearNStep(void *h_out, void *h_in, QudaInvertParam *param, const int n_steps, const double omega)

    #
    # Performs APE, Stout, or Over Imroved STOUT smearing on gaugePrecise and stores it in gaugeSmeared
    # @param[in] smear_param Parameter struct that defines the computation parameters
    # @param[in,out] obs_param Parameter struct that defines which
    # observables we are making and the resulting observables.
    #
    void performGaugeSmearQuda(QudaGaugeSmearParam *smear_param, QudaGaugeObservableParam *obs_param)

    #
    # Performs Wilson Flow on gaugePrecise and stores it in gaugeSmeared
    # @param[in] smear_param Parameter struct that defines the computation parameters
    # @param[in,out] obs_param Parameter struct that defines which
    # observables we are making and the resulting observables.
    #
    void performWFlowQuda(QudaGaugeSmearParam *smear_param, QudaGaugeObservableParam *obs_param)

    #
    # @brief Calculates a variety of gauge-field observables.  If a
    # smeared gauge field is presently loaded (in gaugeSmeared) the
    # observables are computed on this, else the resident gauge field
    # will be used.
    # @param[in,out] param Parameter struct that defines which
    # observables we are making and the resulting observables.
    #
    void gaugeObservablesQuda(QudaGaugeObservableParam *param)

    #
    # Public function to perform color contractions of the host spinors x and y.
    # @param[in] x pointer to host data
    # @param[in] y pointer to host data
    # @param[out] result pointer to the 16 spin projections per lattice site
    # @param[in] cType Which type of contraction (open, degrand-rossi, etc)
    # @param[in] param meta data for construction of ColorSpinorFields.
    # @param[in] X spacetime data for construction of ColorSpinorFields.
    #
    void contractQuda(const void *x, const void *y, void *result, const QudaContractType cType, QudaInvertParam *param,
                      const int *X)

    #
    # @param[in] x pointer to host data array
    # @param[in] y pointer to host data array
    # @param[out] result pointer to the spin*spin projections per lattice slice site
    # @param[in] cType Which type of contraction (open, degrand-rossi, etc)
    # @param[in] param meta data for construction of ColorSpinorFields.
    # @param[in] src_colors color dilution parameter
    # @param[in] X local lattice dimansions
    # @param[in] source_position source position array
    # @param[in] number of momentum modes
    # @param[in] mom_modes momentum modes
    # @param[in] fft_type Fourier phase factor type (cos, sin or exp{ikx})
    #
    void contractFTQuda(void **x, void **y, void **result, const QudaContractType cType, void *cs_param_ptr,
                        const int src_colors, const int *X, const int *const source_position, const int n_mom,
                        const int *const mom_modes, const QudaFFTSymmType *const fft_type)


    #
    # @brief Gauge fixing with overrelaxation with support for single and multi GPU.
    # @param[in,out] gauge, gauge field to be fixed
    # @param[in] gauge_dir, 3 for Coulomb gauge fixing, other for Landau gauge fixing
    # @param[in] Nsteps, maximum number of steps to perform gauge fixing
    # @param[in] verbose_interval, print gauge fixing info when iteration count is a multiple of this
    # @param[in] relax_boost, gauge fixing parameter of the overrelaxation method, most common value is 1.5 or 1.7.
    # @param[in] tolerance, torelance value to stop the method, if this value is zero then the method stops when
    # iteration reachs the maximum number of steps defined by Nsteps
    # @param[in] reunit_interval, reunitarize gauge field when iteration count is a multiple of this
    # @param[in] stopWtheta, 0 for MILC criterion and 1 to use the theta value
    # @param[in] param The parameters of the external fields and the computation settings
    #
    int computeGaugeFixingOVRQuda(void *gauge, const unsigned int gauge_dir, const unsigned int Nsteps,
                                  const unsigned int verbose_interval, const double relax_boost, const double tolerance,
                                  const unsigned int reunit_interval, const unsigned int stopWtheta, QudaGaugeParam *param)

    #
    # @brief Gauge fixing with Steepest descent method with FFTs with support for single GPU only.
    # @param[in,out] gauge, gauge field to be fixed
    # @param[in] gauge_dir, 3 for Coulomb gauge fixing, other for Landau gauge fixing
    # @param[in] Nsteps, maximum number of steps to perform gauge fixing
    # @param[in] verbose_interval, print gauge fixing info when iteration count is a multiple of this
    # @param[in] alpha, gauge fixing parameter of the method, most common value is 0.08
    # @param[in] autotune, 1 to autotune the method, i.e., if the Fg inverts its tendency we decrease the alpha value
    # @param[in] tolerance, torelance value to stop the method, if this value is zero then the method stops when
    # iteration reachs the maximum number of steps defined by Nsteps
    # @param[in] stopWtheta, 0 for MILC criterion and 1 to use the theta value
    # @param[in] param The parameters of the external fields and the computation settings
    #
    int computeGaugeFixingFFTQuda(void *gauge, const unsigned int gauge_dir, const unsigned int Nsteps,
                                  const unsigned int verbose_interval, const double alpha, const unsigned int autotune,
                                  const double tolerance, const unsigned int stopWtheta, QudaGaugeParam *param)

    #
    # @brief Strided Batched GEMM
    # @param[in] arrayA The array containing the A matrix data
    # @param[in] arrayB The array containing the B matrix data
    # @param[in] arrayC The array containing the C matrix data
    # @param[in] native boolean to use either the native or generic version
    # @param[in] param The data defining the problem execution.
    #
    void blasGEMMQuda(void *arrayA, void *arrayB, void *arrayC, QudaBoolean native, QudaBLASParam *param)

    #
    # @brief Strided Batched in-place matrix inversion via LU
    # @param[in] Ainv The array containing the A inverse matrix data
    # @param[in] A The array containing the A matrix data
    # @param[in] use_native Boolean to use either the native or generic version
    # @param[in] param The data defining the problem execution.
    #
    void blasLUInvQuda(void *Ainv, void *A, QudaBoolean use_native, QudaBLASParam *param)

    #
    # @brief Flush the chronological history for the given index
    # @param[in] index Index for which we are flushing
    #
    void flushChronoQuda(int index)


    #
    # Create deflation solver resources.
    #
    #
    void* newDeflationQuda(QudaEigParam *param)

    #
    # Free resources allocated by the deflated solver
    #
    void destroyDeflationQuda(void *df_instance)

    void setMPICommHandleQuda(void *mycomm)

    # Parameter set for quark smearing operations
    ctypedef struct QudaQuarkSmearParam:
        QudaInvertParam *inv_param
        int n_steps
        double width
        int compute_2link
        int delete_2link
        int t0
        double secs
        double gflops

    #
    # Performs two-link Gaussian smearing on a given spinor (for staggered fermions).
    # @param[in,out] h_in Input spinor field to smear
    # @param[in] smear_param   Contains all metadata the operator which will be applied to the spinor
    #
    void performTwoLinkGaussianSmearNStep(void *h_in, QudaQuarkSmearParam *smear_param)

    #
    # @brief Performs contractions between a set of quark fields and
    # eigenvectors of the 3-d Laplace operator.
    # @param[in,out] host_sinks An array representing the inner
    # products between the quark fields and the eigen-vector fields.
    # Ordered as [nQuark][nEv][Lt][nSpin][complexity].
    # @param[in] host_quark Array of quark fields we are taking the inner over
    # @param[in] n_quark Number of quark fields
    # @param[in] tile_quark Tile size for quark fields (batch size)
    # @param[in] host_evec Array of eigenvectors we are taking the inner over
    # @param[in] n_evec Number of eigenvectors
    # @param[in] tile_evec Tile size for eigenvectors (batch size)
    # @param[in] inv_param Meta-data structure
    # @param[in] X Lattice dimensions
    #
    void laphSinkProject(double_complex *host_sinks, void **host_quark, int n_quark, int tile_quark,
                        void **host_evec, int nevec, int tile_evec, QudaInvertParam *inv_param, const int X[4])



