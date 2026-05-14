module TentMPS

using Reexport
@reexport using MPSKit, MPSKitModels, TensorKit

# tensorkit core
using TensorKit, BlockTensorKit
const ⊞ = BlockTensorKit.oplus
using TensorKit: permute

# MPSKit structs
using MPSKit: AbstractFiniteMPS, GrassmannMPS, Multiline, MPODerivativeOperator
using MPSKit: Algorithm, TransferMatrix

# tangent space projections
using MPSKit: AC_hamiltonian, AC2_hamiltonian, C_hamiltonian

# basic functions and types used in algorithms
using MPSKit: DMRG, IterativeSolver, updatetol, calc_galerkin
import MPSKit: find_groundstate

# gauging, environments and state manipulation
using MPSKit: eachsite, environments, regauge!, gauge_step!, recalculate!, _transpose_front

# logging
using MPSKit: LoggingExtras, IterLog, @infov, @warnv,
    loginit!, logiter!, logcancel!, logfinish!

# linear algebra and arrays
using LinearAlgebra, FastGaussQuadrature

# solvers
using OptimKit, KrylovKit

# parallelism
using Base.Threads
using OhMyThreads: tforeach

include("utils.jl")
include("defaults.jl")
include("core/mpskit_overrides.jl")
include("algorithms/environments.jl")

include("core/hamiltonian.jl")
include("core/norm_tensor.jl")
include("core/refinement.jl")
include("core/observables.jl")

include("algorithms/common.jl")
include("algorithms/dmrg.jl")
include("algorithms/vumps.jl")
include("algorithms/gradient.jl")
include("algorithms/lobpcg.jl")
include("algorithms/dense_eig.jl")

# core functions
export build_norm_mpo, construct_hamiltonian, find_generalized_groundstate

# observables
export particle_density, get_energy_densities, single_particle_density_matrix, real_space_single_particle_density_matrix, momentum_distribution

# eigensolver
export LOBPCG, GeneralizedGradientGrassmann, DenseEig

# multigrid optimization
export refine_state

# norm environments
export FiniteLogEnvironments, log_environments

end