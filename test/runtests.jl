using Test
using TentMPS, QuadGK, LinearAlgebra
using Combinatorics

using TentMPS: single_particle_transformation, two_site_decomposition, construct_two_site_gates, tent_basis_overlap_matrix, potential_hamiltonian_coefficients

# utility functions
function mps_fockstate(N::Int, fock_dict::Dict{Int,Int}, cutoff::Int=1)
    # check occupations
    for occ in values(fock_dict)
        if occ < 0 || occ > cutoff
            throw(ArgumentError("Occupation $occ exceeds cutoff $cutoff or is negative."))
        end
    end

    return FiniteMPS([TensorMap(
        reshape([i == get(fock_dict, site, 0) ? 1 : 0 for i in 0:cutoff], (1, cutoff + 1, 1)),
        ℂ^1 ⊗ ℂ^(cutoff + 1) ← ℂ^1
    ) for site in 1:N])
end

# Wick's theorem for vacuum expectation values
function vev(mode_creation::Vector{Int}, mode_annihilation::Vector{Int}, N::AbstractMatrix)
    if length(mode_creation) != length(mode_annihilation)
        return 0.0
    end

    sum(permutations(1:length(mode_creation))) do perm
        prod(getindex.(Ref(N), mode_creation[perm], mode_annihilation))
    end
end

println("Starting tests")
ti = time()

@testset "Tent function integrals" begin
    L = 20 # segments
    grid = range(0, 1, L + 1)
    dx = step(grid)
    h = sqrt(2 / 3 * dx^3)

    # integrate over full grid since the functions only have support over a finite region
    g(n...) = quadgk(x -> prod(tent_function.(x, n, h, Ref(grid))), grid...)[1]

    @test g(1, 1) ≈ 2 * dx^3 / (3 * h^2)
    @test g(3, 3) ≈ 2 * dx^3 / (3 * h^2)
    @test g(3, 4) ≈ dx^3 / (6 * h^2)
    @test g(3, 5) ≈ 0.

    @test g(1, 1, 1, 1) ≈ 2 * dx^5 / (5 * h^4)
    @test g(3, 3, 3, 3) ≈ 2 * dx^5 / (5 * h^4)
    @test g(3, 3, 4, 4) ≈ dx^5 / (30 * h^4)
    @test g(3, 3, 3, 4) ≈ dx^5 / (20 * h^4)
end

@testset "Coefficient matrix" begin
    L = 20 # segments
    N = L - 1 # tents
    grid = range(-5, 5, L + 1)
    dx = step(grid)
    h = sqrt(2 / 3 * dx^3)

    overlap_matrix = tent_basis_overlap_matrix(h, grid)
    coefficient_matrix = single_particle_transformation(overlap_matrix)
    @test coefficient_matrix' * coefficient_matrix ≈ overlap_matrix
    @test coefficient_matrix isa UpperTriangular

    α = zeros(N)
    β = zeros(N - 1)

    α[1] = sqrt(overlap_matrix[1, 1])
    for i in 1:N-1
        β[i] = overlap_matrix[i, i+1] / α[i]
        α[i+1] = sqrt(overlap_matrix[i+1, i+1] - β[i]^2)
    end

    @test diag(coefficient_matrix, 0) ≈ α
    @test diag(coefficient_matrix, 1) ≈ β
end

@testset "Two-site decomposition" begin
    function embed_blocks(blocks::Vector{Matrix{T}}) where T
        L = length(blocks) + 1
        full_matrices = Vector{Matrix{T}}(undef, length(blocks))

        for i in 1:length(blocks)
            M = Matrix{T}(I, L, L)
            M[i:i+1, i:i+1] = blocks[i]
            full_matrices[i] = M
        end

        return full_matrices
    end

    L = 20 # segments
    N = L - 1 # tents
    grid = range(-5, 5, L + 1)
    dx = step(grid)
    h = sqrt(2 / 3 * dx^3)

    overlap_matrix = tent_basis_overlap_matrix(h, grid)
    coefficient_matrix = single_particle_transformation(overlap_matrix)
    two_site_matrices = two_site_decomposition(coefficient_matrix)
    @test coefficient_matrix ≈ prod(reverse(embed_blocks(two_site_matrices)))
end

@testset "Many body overlap tensor" begin
    L = 20 # segments
    N = L - 1 # tents
    grid = range(-5, 5, L + 1)
    dx = step(grid)
    h = sqrt(2 / 3 * dx^3)
    cutoff = 4

    Ho = build_norm_mpo(h, grid, cutoff)
    overlap_matrix = tent_basis_overlap_matrix(h, grid)
    modes = [
        # --- N=1 ---
        (Dict(1 => 1), Dict(1 => 1)),
        (Dict(1 => 1), Dict(2 => 1)),
        (Dict(1 => 1), Dict(5 => 1)),
        (Dict(3 => 1), Dict(4 => 1)),

        # --- N=2 ---
        (Dict(1 => 2), Dict(1 => 2)),
        (Dict(1 => 2), Dict(2 => 2)),
        (Dict(1 => 2), Dict(5 => 2)),
        (Dict(1 => 1, 2 => 1), Dict(1 => 2)),
        (Dict(1 => 1, 2 => 1), Dict(3 => 2)),
        (Dict(1 => 1, 2 => 1), Dict(1 => 1, 2 => 1)),
        (Dict(1 => 1, 3 => 1), Dict(2 => 1, 4 => 1)),
        (Dict(1 => 2), Dict(2 => 1, 3 => 1)),

        # --- N=3 ---
        (Dict(1 => 3), Dict(1 => 3)),
        (Dict(1 => 3), Dict(2 => 3)),
        (Dict(1 => 3), Dict(10 => 3)),
        (Dict(1 => 2, 2 => 1), Dict(1 => 1, 2 => 2)),
        (Dict(1 => 2, 2 => 1), Dict(3 => 3)),
        (Dict(1 => 1, 2 => 1, 3 => 1), Dict(4 => 3)),
        (Dict(1 => 1, 2 => 1, 3 => 1), Dict(1 => 1, 2 => 1, 3 => 1)),
        (Dict(1 => 1, 2 => 1, 3 => 1), Dict(1 => 2, 4 => 1)),

        # --- N=4 ---
        (Dict(1 => 4), Dict(1 => 4)),
        (Dict(1 => 4), Dict(2 => 4)),
        (Dict(1 => 2, 2 => 2), Dict(1 => 2, 2 => 2)),
        (Dict(1 => 2, 2 => 2), Dict(3 => 2, 4 => 2)),
        (Dict(1 => 2, 2 => 2), Dict(5 => 4)),
        (Dict(1 => 3, 2 => 1), Dict(1 => 1, 2 => 3)),
        (Dict(1 => 1, 2 => 1, 3 => 1, 4 => 1), Dict(5 => 4)),
        (Dict(1 => 1, 2 => 1, 3 => 1, 4 => 1), Dict(1 => 2, 2 => 2)),
        (Dict(1 => 1, 2 => 1, 3 => 1, 4 => 1), Dict(1 => 1, 3 => 1, 5 => 1, 7 => 1)),

        # --- should be 0.0 ---
        (Dict(1 => 1), Dict(1 => 2)),
        (Dict(1 => 2, 2 => 1), Dict(1 => 1, 2 => 1)),
        (Dict(1 => 4), Dict(1 => 3))
    ]

    # single particle
    for (d1, d2) in modes
        state1 = mps_fockstate(N, d1, cutoff)
        state2 = mps_fockstate(N, d2, cutoff)
        exact = vev(
            vcat([fill(k, v) for (k, v) in d1]...),
            vcat([fill(k, v) for (k, v) in d2]...),
            overlap_matrix
        ) / (prod(n -> sqrt(factorial(n)), values(d1)) * prod(n -> sqrt(factorial(n)), values(d2)))

        isapprox(dot(state1, Ho * state2), exact, atol=1e-12) || @warn "$(d1) | $(d2)"
        @test isapprox(dot(state1, Ho * state2), exact, atol=1e-12)
    end

    # hermiticity and positive-definiteness
    D = 4
    @test all(val -> (imag(val) / real(val) < 1e-12) && (real(val) > 0), expectation_value(FiniteMPS(rand, ComplexF64, N, ℂ^(cutoff + 1), ℂ^D), Ho) for _ in 1:100)
end

@testset "Exhaustive many-body overlap" begin
    L = 4 # segments
    N = L - 1 # tents
    grid = range(-1, 1, L + 1)
    dx = step(grid)
    h = sqrt(2 / 3 * dx^3)
    cutoff = 2

    Ho = build_norm_mpo(h, grid, cutoff, cutoff_buffer=cutoff, trunctol=1e-10)
    overlap_matrix = tent_basis_overlap_matrix(h, grid)

    # helpers
    to_modes(occ) = vcat([fill(i, n) for (i, n) in enumerate(occ)]...)
    norm_factor(occ) = 1.0 / prod(n -> sqrt(factorial(n)), occ)

    # generate and cache all (cutoff + 1)^N possible states
    states = collect(Iterators.product(fill(0:cutoff, N)...))
    mps_cache = [mps_fockstate(N, Dict(k => v for (k, v) in enumerate(occ) if v > 0), cutoff)
                 for occ in states]

    for (i, occ1) in enumerate(states), (j, occ2) in enumerate(states)
        exact = vev(to_modes(occ1), to_modes(occ2), overlap_matrix) * norm_factor(occ1) * norm_factor(occ2)
        overlap = dot(mps_cache[i], Ho * mps_cache[j])

        isapprox(overlap, exact, atol=1e-12) || @warn "$(occ1) | $(occ2)"
        @test isapprox(overlap, exact, atol=1e-12)
    end
end

@testset "Potential matrix elements" begin
    function harmonic_potential_hamiltonian_coefficients(h, grid)
        dx = step(grid)
        tv_ii = 0.5 * map(x -> dx^3 * (10 * x^2 + dx^2) / (15 * h^2), grid[2:end-1])
        tv_ij = 0.5 * map(x -> dx^3 * (10 * x^2 + 10 * x * dx + 3 * dx^2) / (60 * h^2), grid[2:end-2])
        return tv_ii, tv_ij
    end

    L = 200 # segments
    xmax = 7
    xs = range(-xmax, xmax, L + 1)
    h = sqrt(2 / 3 * step(xs)^3)
    V(x) = 0.5x^2

    diag, offdiag = potential_hamiltonian_coefficients(V, h, xs, tol=1e-12)
    diag1, offdiag1 = harmonic_potential_hamiltonian_coefficients(h, xs)

    @test isapprox(norm(diag - diag1), 0., atol=1e-10)
    @test isapprox(norm(offdiag - offdiag1), 0., atol=1e-10)
end

# assumes symmetric wavefunctions; very unoptimized, only for small benchmarks
function single_particle_density_matrix_naive(Hn, state, N, h, cutoff, base_grid, compute_grid=base_grid)
    chain = FiniteChain(N)
    global_a_min = [@mpoham a_min(cutoff=cutoff){chain[i]} for i in 1:N]

    op_states = [op * state for op in global_a_min]
    h_op_states = [Hn * ψ for ψ in op_states]
    h_state = Hn * state

    diagonal = map(1:N) do i
        dot(op_states[i], h_op_states[i])
    end

    off_diagonal = map(1:N-1) do i
        dot(op_states[i], h_op_states[i+1])
    end

    spdm = Tridiagonal(conj.(off_diagonal), diagonal, off_diagonal)

    normalization = dot(state, h_state)

    densities = map(compute_grid) do x
        f_vec = tent_function.(x, 1:N, h, Ref(base_grid))
        return real(f_vec' * spdm * f_vec)
    end

    return real.(densities ./ normalization)
end