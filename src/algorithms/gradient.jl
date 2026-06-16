module GeneralizedGrassmannMPS

using TensorKit
using MPSKit
using MPSKit: AbstractMPSEnvironments, InfiniteEnvironments, MultilineEnvironments, AC_hamiltonian, recalculate!

import TensorKitManifolds.Grassmann
using TensorKitManifolds.Grassmann: GrassmannTangent, checkbase
using MPSKit.GrassmannMPS: rmul

using OhMyThreads
using ..TentMPS: calc_generalized_energy

function fg(
    state::FiniteMPS,
    operator::Union{O,LazySum{O}},
    norm_operator::Union{N,LazySum{N}},
    Henvs::AbstractMPSEnvironments=environments(state, operator),
    Nenvs::AbstractMPSEnvironments=environments(state, norm_operator)
) where {O<:Union{FiniteMPO,FiniteMPOHamiltonian},N<:FiniteMPO}

    f = calc_generalized_energy(state, operator, norm_operator, Henvs, Nenvs)

    # <psi|N|psi>
    n = expectation_value(state, norm_operator, Nenvs) * dot(state, state)

    isapprox(imag(f), 0; atol=eps(abs(f))^(3 / 4)) || @warn "MPO might not be Hermitian: $f"

    gs = map(1:length(state)) do i

        H_AC = AC_hamiltonian(i, state, operator, state, Henvs) * state.AC[i]
        N_AC = AC_hamiltonian(i, state, norm_operator, state, Nenvs) * state.AC[i]
        AC′ = (H_AC - f * N_AC) / n
        g = Grassmann.project(AC′, state.AL[i])

        return rmul(g, state.C[i]')
    end

    return real(f), gs
end
end


struct GeneralizedGradientGrassmann{O<:OptimKit.OptimizationAlgorithm,F} <: Algorithm
    "optimization algorithm"
    method::O
    "callback function applied after each iteration, of signature `finalize!(x, f, g, numiter) -> x, f, g`"
    finalize!::F

    function GeneralizedGradientGrassmann(;
        method=ConjugateGradient, (finalize!)=OptimKit._finalize!,
        tol=Defaults.tol, maxiter=Defaults.maxiter,
        verbosity=Defaults.verbosity - 1
    )
        if isa(method, OptimKit.OptimizationAlgorithm)
            m = method
        elseif method <: OptimKit.OptimizationAlgorithm
            linesearch = OptimKit.HagerZhangLineSearch(;
                verbosity=verbosity - 2, maxiter=100
            )
            m = method(; maxiter, verbosity, gradtol=tol, linesearch)
        else
            msg = "method should be either an instance or a subtype of `OptimKit.OptimizationAlgorithm`."
            throw(ArgumentError(msg))
        end
        return new{typeof(m),typeof(finalize!)}(m, finalize!)
    end
end

function find_generalized_groundstate(
    ψ::S, H, N, alg::GeneralizedGradientGrassmann, Henvs::P=environments(ψ, H), Nenvs::Q=environments(ψ, N)
)::Tuple{S,P,Q,Float64} where {S,P,Q}
    !isa(ψ, FiniteMPS) || dim(ψ.C[end]) == 1 ||
        @warn "This is not fully supported - split the mps up in a sum of mps's and optimize seperately"
    normalize!(ψ)

    fg(x) = GeneralizedGrassmannMPS.fg(x, H, N, Henvs, Nenvs)
    x, _, _, _, normgradhistory = optimize(
        fg, ψ, alg.method;
        GrassmannMPS.transport!,
        GrassmannMPS.retract,
        GrassmannMPS.inner,
        GrassmannMPS.scale!,
        GrassmannMPS.add!,
        GrassmannMPS.precondition,
        alg.finalize!,
        isometrictransport=true
    )
    return x, Henvs, Nenvs, normgradhistory[end]
end
