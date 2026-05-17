# computes ⟨O1p_i O2_p_j N O1m_i O2m_j⟩; i.e, generic expectation values for operators in computational basis
# p -> creation, m -> annihilation
function generic_expval(Hn, state; O1p, O1m, O2p, O2m)
    envs = environments(state, Hn)
    N = length(state)

    vals = map(1:N-1) do i
        tm1 = TransferMatrix(state.AL[i], O1p × Hn[i] × O1m, state.AL[i])
        tm2 = TransferMatrix(state.AC[i+1], O2p × Hn[i+1] × O2m, state.AC[i+1])
        ρL = leftenv(envs, i, state)
        ρR = rightenv(envs, i + 1, state)

        res = tm1 * (tm2 * ρR)
        @tensor scalar[] := ρL[1, 2, 3] * res[3, 2, 1]
        return tr(scalar)
    end

    # normalization
    i = 1
    ρR = rightenv(envs, i, state)
    tm = TransferMatrix(state.AC[i], Hn[i], state.AC[i])
    ρL = leftenv(envs, i, state)
    res = tm * ρR
    @tensor normalization[] := ρL[1, 2, 3] * res[3, 2, 1]

    return vals ./ tr(normalization)
end

############## single particle density matrix

function single_particle_density_matrix(Hn, state, envs::MPSKit.FiniteEnvironments=environments(state, Hn); O1, O2, band=nothing)
    N = length(state)
    isnothing(band) && (band = N)

    rho = zeros(ComplexF64, N, N)

    ρL1, ρR1 = leftenv(envs, 1, state), rightenv(envs, 1, state)
    tm1 = MPSKit.TransferMatrix(state.AC[1], Hn[1], state.AC[1])
    @tensor sc1[] := ρL1[1, 2, 3] * (tm1*ρR1)[3, 2, 1]
    z_mb = tr(sc1)

    # diagonal entries 
    for i in 1:N
        ρL, ρR = leftenv(envs, i, state), rightenv(envs, i, state)
        tm = MPSKit.TransferMatrix(state.AC[i], O1 × Hn[i] × O2, state.AC[i])
        @tensor sc[] := ρL[1, 2, 3] * (tm*ρR)[3, 2, 1]
        rho[i, i] = tr(sc) / z_mb
    end

    # upper off-diagonal entries
    for i in 1:N-1
        # initialize the environment starting at site i with O1 applied
        ρL_i = leftenv(envs, i, state)
        tm_i = MPSKit.TransferMatrix(state.AL[i], O1 × Hn[i], state.AL[i])

        # propagated environment: represents the state from site 1 to i with O1 acting at site i.
        curr_env = ρL_i * tm_i

        for j in (i+1):min(i + band, N)
            # compute the correlation at site j using O2
            ρR_j = rightenv(envs, j, state)
            tm_j = MPSKit.TransferMatrix(state.AC[j], Hn[j] × O2, state.AC[j])

            @tensor sc[] := curr_env[1, 2, 3] * (tm_j*ρR_j)[3, 2, 1]
            rho[i, j] = tr(sc) / z_mb

            # if we need to move further to the right (j+1), 
            # propagate the environment through the "background" Hn at site j
            if j < N
                tm_mid = MPSKit.TransferMatrix(state.AL[j], Hn[j], state.AL[j])
                curr_env = curr_env * tm_mid
            end
        end
    end

    # assuming Hermiticity
    for i in 1:N, j in (i+1):N
        rho[j, i] = conj(rho[i, j])
    end

    return rho
end

# assumes Hermitian; specialized for local density observables
function single_particle_density_matrix_tridiagonal(Hn, state, envs::MPSKit.FiniteEnvironments; O1, O2)
    N = length(state)

    # normalization
    ρL1, ρR1 = leftenv(envs, 1, state), rightenv(envs, 1, state)
    tm1 = TransferMatrix(state.AC[1], Hn[1], state.AC[1])
    @tensor sc1[] := ρL1[1, 2, 3] * (tm1*ρR1)[3, 2, 1]
    z_mb = tr(sc1)

    diag = map(1:N) do i
        ρL, ρR = leftenv(envs, i, state), rightenv(envs, i, state)
        tm = TransferMatrix(state.AC[i], O1 × Hn[i] × O2, state.AC[i])
        @tensor sc[] := ρL[1, 2, 3] * (tm*ρR)[3, 2, 1]
        return tr(sc) / z_mb
    end

    off_diag = map(1:N-1) do i
        ρL, ρR = leftenv(envs, i, state), rightenv(envs, i + 1, state)
        tm_i = TransferMatrix(state.AL[i], O1 × Hn[i], state.AL[i])
        tm_next = TransferMatrix(state.AC[i+1], Hn[i+1] × O2, state.AC[i+1])
        @tensor sc[] := ρL[1, 2, 3] * (tm_i*(tm_next*ρR))[3, 2, 1]
        return tr(sc) / z_mb
    end

    return Tridiagonal(conj.(off_diag), diag, off_diag)
end

# assumes Hermitian
function single_particle_density_matrix_tridiagonal(Hn, state, envs::FiniteLogEnvironments=log_environments(state, Hn); O1, O2)
    N = length(state)

    # force update
    leftenv(envs, N + 1, state)
    rightenv(envs, 0, state)
    glog = global_log_scale(envs)

    ρL1, ρR1 = leftenv(envs, 1, state), rightenv(envs, 1, state)
    tm1 = TransferMatrix(state.AC[1], Hn[1], state.AC[1])
    @tensor sc1[] := ρL1[1, 2, 3] * (tm1*ρR1)[3, 2, 1]
    z_mb = tr(sc1) * exp(local_log_scale(envs, 1) - glog)

    diag = map(1:N) do i
        ρL, ρR = leftenv(envs, i, state), rightenv(envs, i, state)
        tm = TransferMatrix(state.AC[i], O1 × Hn[i] × O2, state.AC[i])
        @tensor sc[] := ρL[1, 2, 3] * (tm*ρR)[3, 2, 1]
        (tr(sc) / z_mb) * exp(local_log_scale(envs, i) - glog)
    end

    off_diag = map(1:N-1) do i
        ρL, ρR = leftenv(envs, i, state), rightenv(envs, i + 1, state)
        tm_i = TransferMatrix(state.AL[i], O1 × Hn[i], state.AL[i])
        tm_next = TransferMatrix(state.AC[i+1], Hn[i+1] × O2, state.AC[i+1])
        @tensor sc[] := ρL[1, 2, 3] * (tm_i*(tm_next*ρR))[3, 2, 1]
        lscale = envs.log_L_scales[i] + envs.log_R_scales[i+2]
        (tr(sc) / z_mb) * exp(lscale - glog)
    end

    return Tridiagonal(conj.(off_diag), diag, off_diag)
end

########## local densities

function _make_tridiagonal_spdm(Hn, state)
    c = dim(space(first(state.AL), 2)) - 1
    return single_particle_density_matrix_tridiagonal(Hn, state,
        O1=a_plus(cutoff=c), O2=a_min(cutoff=c))
end

function _make_full_spdm(Hn, state)
    c = dim(space(first(state.AL), 2)) - 1
    return single_particle_density_matrix(Hn, state,
        O1=a_plus(cutoff=c), O2=a_min(cutoff=c))
end

function particle_density(spdm::AbstractMatrix, base_grid; compute_grid=base_grid, h=Defaults.h(base_grid))
    N = size(spdm, 1)
    return map(compute_grid) do x
        f = tent_function.(x, 1:N, h, Ref(base_grid))
        return real(dot(f, spdm, f))
    end
end

function particle_density(Hn, state, base_grid; kwargs...)
    _spdm = _make_tridiagonal_spdm(Hn, state)
    return particle_density(_spdm, base_grid; kwargs...)
end

function kinetic_energy_density(spdm::AbstractMatrix, base_grid; compute_grid=base_grid, h=Defaults.h(base_grid))
    N = size(spdm, 1)
    return map(compute_grid) do x
        df = tent_function_derivative.(x, 1:N, h, Ref(base_grid))
        return 0.5 * real(dot(df, spdm, df))
    end
end

function kinetic_energy_density(Hn, state, base_grid; kwargs...)
    _spdm = _make_tridiagonal_spdm(Hn, state)
    return kinetic_energy_density(_spdm, base_grid; kwargs...)
end

function potential_energy_density(spdm::AbstractMatrix, base_grid; compute_grid=base_grid, h=Defaults.h(base_grid), V)
    N = size(spdm, 1)
    return map(compute_grid) do x
        f = tent_function.(x, 1:N, h, Ref(base_grid))
        return V(x) * real(dot(f, spdm, f))
    end
end

function interaction_energy_density(Hn, state, base_grid; compute_grid=base_grid, h=Defaults.h(base_grid))
    N = length(state)
    cutoff = dim(space(first(state.AL), 2)) - 1

    ad_op = a_plus(cutoff=cutoff)
    a_op = a_min(cutoff=cutoff)
    id_op = one(ad_op)

    # lllm, mlll
    lllm = generic_expval(Hn, state, O1p=ad_op * ad_op, O1m=a_op, O2p=id_op, O2m=a_op)

    # mmml, lmmm
    mmml = generic_expval(Hn, state, O1p=ad_op, O1m=id_op, O2p=ad_op, O2m=a_op * a_op)

    #lmml
    lmml = generic_expval(Hn, state, O1p=ad_op, O1m=a_op, O2p=ad_op, O2m=a_op)

    #llmm, mmll
    llmm = generic_expval(Hn, state, O1p=ad_op * ad_op, O1m=id_op, O2p=id_op, O2m=a_op * a_op)

    # llll
    llll = diag(single_particle_density_matrix_tridiagonal(Hn, state, O1=ad_op * ad_op, O2=a_op * a_op))

    return map(compute_grid) do x
        f = tent_function.(x, 1:N, h, Ref(base_grid))
        fl, fm = f[1:end-1], f[2:end]

        term_llll = real(sum((f .^ 4) .* llll))
        term_lllm = 4 * real(sum(((fl .^ 3) .* fm) .* lllm))
        term_mmml = 4 * real(sum((fl .* (fm .^ 3)) .* mmml))
        term_lmml = 4 * real(sum(((fl .^ 2) .* (fm .^ 2)) .* lmml))
        term_llmm = 2 * real(sum(((fl .^ 2) .* (fm .^ 2)) .* llmm))

        return term_llll + term_lllm + term_mmml + term_lmml + term_llmm
    end
end

function get_energy_densities(Hn, state, base_grid; μ, g, V, kwargs...)
    _spdm = _make_tridiagonal_spdm(Hn, state)
    n = particle_density(_spdm, base_grid; kwargs...)
    ke = kinetic_energy_density(_spdm, base_grid; kwargs...)
    pe = potential_energy_density(_spdm, base_grid; V=V, kwargs...)
    ie = interaction_energy_density(Hn, state, base_grid; kwargs...)
    return μ * n, ke, pe, g * ie
end

### other observables

function real_space_single_particle_density_matrix(spdm::AbstractMatrix, base_grid; compute_grid=base_grid, h=Defaults.h(base_grid))
    F = [tent_function(x, k, h, base_grid) for k in 1:size(spdm, 1), x in compute_grid]
    return F' * spdm * F
end

function real_space_single_particle_density_matrix(Hn, state, base_grid; kwargs...)
    _spdm = _make_tridiagonal_spdm(Hn, state)
    return real_space_single_particle_density_matrix(_spdm, base_grid; kwargs...)
end

function momentum_distribution(spdm::AbstractMatrix, base_grid; compute_grid=base_grid, h=Defaults.h(base_grid))
    N = size(spdm, 1)
    dx = step(base_grid)
    nodes = base_grid[2:end-1]
    nk = zeros(length(compute_grid))

    for (m, k) in enumerate(compute_grid)
        if abs(k) < 1e-10
            f_k = fill(dx^2 / h, N)
        else
            phase = exp.(-im * k .* nodes)
            sinc_factor = (sin(k * dx / 2) / (k * dx / 2))^2
            f_k = (dx^2 / h) .* sinc_factor .* phase
        end

        nk[m] = real(dot(f_k, spdm, f_k))
    end

    return nk
end

function momentum_distribution(Hn, state, base_grid; kwargs...)
    _spdm = _make_full_spdm(Hn, state)
    return momentum_distribution(_spdm, base_grid; kwargs...)
end

function tan_contact(Hn, state, base_grid; g, kwargs...)
    cutoff = get_particle_cutoff(state)
    Hi = _construct_interaction_hamiltonian(Hn, base_grid, cutoff; g, kwargs...)
    return 4 * g * real(expectation_value(state, Hi) / expectation_value(state, Hn))
end