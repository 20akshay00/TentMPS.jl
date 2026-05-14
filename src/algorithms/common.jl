function calc_generalized_galerkin(
    pos::Int, below::Union{InfiniteMPS,FiniteMPS,WindowMPS},
    operator, norm_operator, above, envs, norm_envs
)
    AC = above.AC[pos]

    HeAC = AC_hamiltonian(pos, below, operator, above, envs) * AC
    NeAC = AC_hamiltonian(pos, below, norm_operator, above, norm_envs) * AC

    λ = real(dot(AC, HeAC) / dot(AC, NeAC))
    n_H = norm(HeAC)

    AC′ = HeAC / n_H - (λ / n_H * NeAC)
    out = mul!(AC′, below.AL[pos], below.AL[pos]' * AC′, -1, +1)
    return norm(out)
end

function calc_generalized_galerkin(
    pos::Int, below::Union{InfiniteMPS,FiniteMPS,WindowMPS},
    operator, norm_operator, above, envs::FiniteLogEnvironments, norm_envs::FiniteLogEnvironments
)
    AC = above.AC[pos]

    HeAC = AC_hamiltonian(pos, below, operator, above, envs) * AC
    NeAC = AC_hamiltonian(pos, below, norm_operator, above, norm_envs) * AC

    λ = real(dot(AC, HeAC) / dot(AC, NeAC))
    n_H = norm(HeAC)

    AC′ = HeAC / n_H - (λ / n_H * NeAC)
    out = mul!(AC′, below.AL[pos], below.AL[pos]' * AC′, -1, +1)
    return norm(out)
end

function calc_generalized_galerkin(
    below::Union{InfiniteMPS,FiniteMPS,WindowMPS}, operator, norm_operator, above, envs, norm_envs
)
    return maximum(pos -> calc_generalized_galerkin(pos, below, operator, norm_operator, above, envs, norm_envs), 1:length(above))
end

function generalizedfixedpoint(fs, x₀, which::Symbol, alg)
    vals, vecs, info = geneigsolve(fs, x₀, 1, which, alg)

    if info.converged == 0
        @warnv 1 "fixedpoint not converged after $(info.numiter) iterations: normres = $(info.normres[1])"
    end

    return vals[1], vecs[1]
end

function calc_generalized_energy(ψ, H, N, Henvs, Nenvs)
    return expectation_value(ψ, H, Henvs) / expectation_value(ψ, N, Nenvs)
end

function calc_generalized_energy(ψ, H, N, Henvs::FiniteLogEnvironments, Nenvs::FiniteLogEnvironments)
    return expectation_value(ψ, H, Henvs) / expectation_value(ψ, N, Nenvs) * exp(global_log_scale(Henvs) - global_log_scale(Nenvs))
end