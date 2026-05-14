
# Internal state of the VUMPS algorithm for a generalized eigenvalue problem
struct GeneralizedVUMPSState{S,O,N,E,F}
    mps::S
    operator::O
    norm_operator::N
    envs::E
    norm_envs::F
    iter::Int
    ϵ::Float64
    which::Symbol
end

function find_generalized_groundstate(
    ψi::InfiniteMPS, H, N, alg::VUMPS, Henvs=environments(ψi, H), Nenvs=environments(ψi, N)
)
    return generalized_dominant_eigsolve(H, N, ψi, alg, Henvs, Nenvs; which=:SR)
end

function generalized_dominant_eigsolve(
    H, N, ψi, alg::VUMPS, Henvs=environments(ψi, H), Nenvs=environments(ψi, N);
    which
)
    ψ = deepcopy(ψi)
    log = IterLog("VUMPS")
    iter = 0
    ϵ = calc_generalized_galerkin(ψ, H, N, ψ, Henvs, Nenvs)
    alg_environments = updatetol(alg.alg_environments, iter, ϵ)
    recalculate!(Henvs, ψ, H, ψ; alg_environments.tol)
    recalculate!(Nenvs, ψ, N, ψ; alg_environments.tol)

    state = GeneralizedVUMPSState(ψ, H, N, Henvs, Nenvs, iter, ϵ, which)
    it = IterativeSolver(alg, state)

    return LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ, sum(calc_generalized_energy(ψ, H, N, Henvs, Nenvs)))

        for (ψ, Henvs, Nenvs, ϵ) in it
            if ϵ ≤ alg.tol
                @infov 2 logfinish!(log, it.iter, ϵ, calc_generalized_energy(ψ, H, N, Henvs, Nenvs))
                return ψ, Henvs, Nenvs, ϵ
            end
            if it.iter ≥ alg.maxiter
                @warnv 1 logcancel!(log, it.iter, ϵ, calc_generalized_energy(ψ, H, N, Henvs, Nenvs))
                return ψ, Henvs, Nenvs, ϵ
            end
            @infov 3 logiter!(log, it.iter, ϵ, calc_generalized_energy(ψ, H, N, Henvs, Nenvs))
        end

        # this should never be reached
        return it.state.mps, it.state.envs, it.state.ϵ
    end
end

function Base.iterate(it::IterativeSolver{<:VUMPS,<:GeneralizedVUMPSState}, state=it.state)
    ACs = localupdate_step!(it, state)
    ψ = gauge_step!(it, state, ACs)
    Henvs, Nenvs = envs_step!(it, state, ψ)

    # finalizer step
    ψ, Henvs = it.finalize(state.iter, ψ, state.operator, Henvs)::Tuple{typeof(ψ),typeof(Henvs)}

    # error criterion
    ϵ = calc_generalized_galerkin(ψ, state.operator, state.norm_operator, ψ, Henvs, Nenvs)

    # update state
    it.state = GeneralizedVUMPSState(ψ, state.operator, state.norm_operator, Henvs, Nenvs, state.iter + 1, ϵ, state.which)

    return (ψ, Henvs, Nenvs, ϵ), it.state
end

function localupdate_step!(
    it::IterativeSolver{<:VUMPS,<:GeneralizedVUMPSState}, state, scheduler=MPSKit.Defaults.scheduler[]
)
    alg_eigsolve = updatetol(it.alg_eigsolve, state.iter, state.ϵ)
    alg_orth = LAPACK_HouseholderQR(; positive=true) #Defaults.alg_qr()

    ψ = state.mps
    src_Cs = ψ isa Multiline ? eachcol(ψ.C) : ψ.C
    src_ACs = ψ isa Multiline ? eachcol(ψ.AC) : ψ.AC
    ACs = similar(ψ.AC)
    dst_ACs = ψ isa Multiline ? eachcol(ACs) : ACs

    tforeach(eachsite(ψ), src_ACs, src_Cs; scheduler) do site, AC₀, C₀
        dst_ACs[site] = _localupdate_vumps_step!(
            site, ψ, state.operator, state.norm_operator, state.envs, state.norm_envs, AC₀, C₀;
            parallel=false, alg_orth, state.which, alg_eigsolve
        )
        return nothing
    end

    return ACs
end

function _localupdate_vumps_step!(
    site, ψ, H, N, Henvs, Nenvs, AC₀, C₀;
    parallel::Bool=false, alg_orth=MPSKit.Defaults.alg_qr(),
    alg_eigsolve=MPSKit.Defaults.eigsolver, which
)
    if !parallel
        Hac = AC_hamiltonian(site, ψ, H, ψ, Henvs)
        Nac = AC_hamiltonian(site, ψ, N, ψ, Nenvs)
        _, AC = generalizedfixedpoint((Hac, Nac), AC₀, which, alg_eigsolve)

        Hc = C_hamiltonian(site, ψ, H, ψ, Henvs)
        Nc = C_hamiltonian(site, ψ, N, ψ, Nenvs)
        _, C = generalizedfixedpoint((Hc, Nc), C₀, which, alg_eigsolve)
        return regauge!(AC, C; alg=alg_orth)
    end

    local AC, C
    @sync begin
        @spawn begin
            Hac = AC_hamiltonian(site, ψ, H, ψ, Henvs)
            Nac = AC_hamiltonian(site, ψ, N, ψ, Nenvs)
            _, AC = generalizedfixedpoint((Hac, Nac), AC₀, which, alg_eigsolve)
        end
        @spawn begin
            Hc = C_hamiltonian(site, ψ, H, ψ, Henvs)
            Nc = C_hamiltonian(site, ψ, N, ψ, Nenvs)
            _, C = generalizedfixedpoint((Hc, Nc), C₀, which, alg_eigsolve)
        end
    end
    return regauge!(AC, C; alg=alg_orth)
end

function envs_step!(it::IterativeSolver{<:VUMPS,<:GeneralizedVUMPSState}, state, ψ)
    alg_environments = updatetol(it.alg_environments, state.iter, state.ϵ)
    return recalculate!(state.envs, ψ, state.operator, ψ; alg_environments.tol), recalculate!(state.norm_envs, ψ, state.norm_operator, ψ; alg_environments.tol)
end