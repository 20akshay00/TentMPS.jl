# N = number of tents
# L = number of segments
# never really need number of points in domain

function kinetic_hamiltonian_coefficients(grid; h=Defaults.h(grid))
    dx, N = step(grid), length(grid) - 2
    tk_ii = fill(dx / h^2, N)
    tk_ij = fill(-dx / (2h^2), N - 1)
    return tk_ii, tk_ij
end

function _integrate_element(V, h, xL, xR, order, gl_cache)
    # compute/retrieve quadrature nodes/weights
    xi, w = get!(gl_cache, order) do
        nodes, weights = gausslegendre(order)
        (nodes .+ 1) ./ 2, weights ./ 2
    end

    dx = xR - xL
    vk = V.(xL .+ xi .* dx)
    scale = ((xR - xL) / h)^2

    # phi_L (falling) is (1-xi), phi_R (rising) is (xi)
    prefactor = dx * scale
    v_diag_L = prefactor * sum(w .* (1 .- xi) .^ 2 .* vk)
    v_diag_R = prefactor * sum(w .* xi .^ 2 .* vk)
    v_off = prefactor * sum(w .* xi .* (1 .- xi) .* vk)
    return v_diag_L, v_diag_R, v_off
end

function potential_hamiltonian_coefficients(V, grid; tol=1e-12, max_order=128, h=Defaults.h(grid))
    n = length(grid)
    Vii, Vij = zeros(n - 2), zeros(n - 3)

    # cache for quadrature nodes/weights
    gl_cache = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()

    max_reached, any_failed = 8, false

    for k in 1:(n-1)
        xL, xR = grid[k], grid[k+1]
        vL, vR, vO = 0.0, 0.0, 0.0
        order, converged = 8, false

        while order <= max_order
            nL, nR, nO = _integrate_element(V, h, xL, xR, order, gl_cache)

            if order > 8 && isapprox(nL, vL; atol=tol, rtol=tol) &&
               isapprox(nR, vR; atol=tol, rtol=tol)
                vL, vR, vO, converged = nL, nR, nO, true
                break
            end
            vL, vR, vO, max_reached = nL, nR, nO, max(max_reached, order)
            order *= 2
        end
        !converged && (any_failed = true)

        k > 1 && (Vii[k-1] += vL)
        k < n - 1 && (Vii[k] += vR)
        1 < k < n - 1 && (Vij[k-1] = vO)
    end

    any_failed ? @warn("Potential elements could not be computed upto tolerance $tol with maximal quadrature order $max_order !") :
    @info("Potential elements computed upto tolerance $tol with maximal quadrature order $(max_reached).")
    0
    return Vii, Vij
end


function chemical_potential_hamiltonian_coefficients(grid; h=Defaults.h(grid))
    dx, N = step(grid), length(grid) - 2
    tc_ii = fill(2 * dx^3 / (3 * h^2), N)
    tc_ij = fill(dx^3 / (6 * h^2), N - 1)
    return tc_ii, tc_ij
end

function interaction_hamiltonian_coefficients(grid; h=Defaults.h(grid))
    dx, N = step(grid), length(grid) - 2
    Uiiii = fill(2 * dx^5 / (5 * h^4), N)
    Uiiij = fill(dx^5 / (20 * h^4), N - 1)
    Uiijj = fill(dx^5 / (30 * h^4), N - 1)
    return Uiiii, Uiiij, Uiijj
end

function compute_hamiltonian_coefficients(grid, g, μ, V; kwargs...)
    N = length(grid) - 2

    tk_ii, tk_ij = kinetic_hamiltonian_coefficients(grid; kwargs...)
    tc_ii, tc_ij = chemical_potential_hamiltonian_coefficients(grid; kwargs...)
    Uiiii, Uiiij, Uiijj = interaction_hamiltonian_coefficients(grid; kwargs...)

    tv_ii, tv_ij = potential_hamiltonian_coefficients(V, grid; kwargs...)

    tii = tk_ii + tv_ii - μ * tc_ii
    tij = tk_ij + tv_ij - μ * tc_ij

    return tii, tij, g * Uiiii, g * Uiiij, g * Uiijj
end

function construct_hamiltonian(Hn, grid, cutoff; g, μ, V, kwargs...)
    local_mpo = Vector{BlockTensorMap}(undef, length(Hn))
    D_MPO = 9 # hard-coded
    a = a_min(cutoff=cutoff)
    ad = a_plus(cutoff=cutoff)

    N = length(grid) - 2
    tii, tij, Uiiii, Uiiij, Uiijj = compute_hamiltonian_coefficients(grid, g, μ, V; kwargs...)

    @assert length(Hn) == N

    for i in eachindex(Hn)
        Nt = Hn[i]
        codomain = (reduce(⊞, fill(space(Nt, 1), (i == 1) ? 1 : D_MPO))) ⊗ ⊞(ℂ^(cutoff + 1))
        domain = ⊞(ℂ^(cutoff + 1)) ⊗ (reduce(⊞, fill(space(Nt, 4)', (i == N) ? 1 : D_MPO)))

        local_mpo[i] = zeros(codomain ← domain)
        # first row
        if i < N
            local_mpo[i][1, 1, 1, 1] = Nt
            local_mpo[i][1, 1, 1, 2] = 2Uiiij[i] * ad × ad × Nt × a + tij[i] * ad × Nt
            local_mpo[i][1, 1, 1, 3] = 2Uiiij[i] * ad × Nt × a × a + tij[i] * Nt × a
            local_mpo[i][1, 1, 1, 4] = 2Uiiij[i] * Nt × a
            local_mpo[i][1, 1, 1, 5] = 2Uiiij[i] * ad × Nt
            local_mpo[i][1, 1, 1, 6] = Uiijj[i] * ad × ad × Nt
            local_mpo[i][1, 1, 1, 7] = 4Uiijj[i] * ad × Nt × a
            local_mpo[i][1, 1, 1, 8] = Uiijj[i] * Nt × a × a
        end

        # last column
        if i > 1
            local_mpo[i][2, 1, 1, end] = Nt × a
            local_mpo[i][3, 1, 1, end] = ad × Nt
            local_mpo[i][4, 1, 1, end] = ad × ad × Nt × a
            local_mpo[i][5, 1, 1, end] = ad × Nt × a × a
            local_mpo[i][6, 1, 1, end] = Nt × a × a
            local_mpo[i][7, 1, 1, end] = ad × Nt × a
            local_mpo[i][8, 1, 1, end] = ad × ad × Nt
            local_mpo[i][9, 1, 1, end] = Nt
        end

        # first row, last column
        local_mpo[i][1, 1, 1, end] = Uiiii[i] * ad × ad × Nt × a × a + tii[i] * ad × Nt × a
    end

    return FiniteMPO(collect(typeof(first(local_mpo)), local_mpo))
end

# to extract Tan contact
function _construct_interaction_hamiltonian(Hn, grid, cutoff; g, kwargs...)
    local_mpo = Vector{BlockTensorMap}(undef, length(Hn))
    D_MPO = 9 # hard-coded
    a = a_min(cutoff=cutoff)
    ad = a_plus(cutoff=cutoff)

    N = length(grid) - 2
    Uiiii, Uiiij, Uiijj = interaction_hamiltonian_coefficients(grid; kwargs...)

    Uiiii .*= g
    Uiiij .*= g
    Uiijj .*= g

    @assert length(Hn) == N

    for i in eachindex(Hn)
        Nt = Hn[i]
        codomain = (reduce(⊞, fill(space(Nt, 1), (i == 1) ? 1 : D_MPO))) ⊗ ⊞(ℂ^(cutoff + 1))
        domain = ⊞(ℂ^(cutoff + 1)) ⊗ (reduce(⊞, fill(space(Nt, 4)', (i == N) ? 1 : D_MPO)))

        local_mpo[i] = zeros(codomain ← domain)
        # first row
        if i < N
            local_mpo[i][1, 1, 1, 1] = Nt
            local_mpo[i][1, 1, 1, 2] = 2Uiiij[i] * ad × ad × Nt × a
            local_mpo[i][1, 1, 1, 3] = 2Uiiij[i] * ad × Nt × a × a
            local_mpo[i][1, 1, 1, 4] = 2Uiiij[i] * Nt × a
            local_mpo[i][1, 1, 1, 5] = 2Uiiij[i] * ad × Nt
            local_mpo[i][1, 1, 1, 6] = Uiijj[i] * ad × ad × Nt
            local_mpo[i][1, 1, 1, 7] = 4Uiijj[i] * ad × Nt × a
            local_mpo[i][1, 1, 1, 8] = Uiijj[i] * Nt × a × a
        end

        # last column
        if i > 1
            local_mpo[i][2, 1, 1, end] = Nt × a
            local_mpo[i][3, 1, 1, end] = ad × Nt
            local_mpo[i][4, 1, 1, end] = ad × ad × Nt × a
            local_mpo[i][5, 1, 1, end] = ad × Nt × a × a
            local_mpo[i][6, 1, 1, end] = Nt × a × a
            local_mpo[i][7, 1, 1, end] = ad × Nt × a
            local_mpo[i][8, 1, 1, end] = ad × ad × Nt
            local_mpo[i][9, 1, 1, end] = Nt
        end

        # first row, last column
        local_mpo[i][1, 1, 1, end] = Uiiii[i] * ad × ad × Nt × a × a
    end

    return FiniteMPO(collect(typeof(first(local_mpo)), local_mpo))
end