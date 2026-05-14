# non unitary linear transformation to an orthonormal basis to account for non-canonical commutational relations
# convention: always upper triangular, i.e, a_j† = ∑(j ≤ i) c_i† M_ij
single_particle_transformation(commutator_matrix) = (cholesky(commutator_matrix).L)'

# returns matrices in the exponential of a gaussian operator, assumes M is bidiagonal for now
function two_site_decomposition(M)
    N = size(M, 1)
    Ms = Vector{Matrix{eltype(M)}}(undef, N - 1)

    # not really used with our convention, but nice to keep it general anyways
    if M isa LowerTriangular
        for i in 1:N-1
            Ms[i] = [
                M[i, i] 0;
                M[i+1, i] 1
            ]
        end
        Ms[end][2, 2] = M[N, N]

        return reverse(Ms)

    elseif M isa UpperTriangular
        for i in 1:N-1
            Ms[i] = [
                M[i, i] M[i, i+1];
                0 1
            ]
        end
        Ms[end][2, 2] = M[N, N]

        return Ms

    else
        throw(DomainError("Input must be UpperTriangular or LowerTriangular."))
    end
end

# constructs gaussian operators with the input of the matrices
function construct_two_site_gates(Ms, cutoff)
    n̂ = a_number(cutoff=cutoff)
    Î = one(n̂)
    âꜝâ = a_plusmin(cutoff=cutoff)
    ââꜝ = a_minplus(cutoff=cutoff)

    quadratic_operators = [n̂⊗Î âꜝâ; ââꜝ Î⊗n̂]

    gates = Vector{TensorMap}(undef, length(Ms))

    for i in eachindex(Ms)
        coefficients = log(Ms[i])
        gates[i] = exp(sum(coefficients .* quadratic_operators))
    end

    return collect(typeof(first(gates)), gates)
end

# hard-coded for lower-triangular coefficient matrix M => upper triangular M†
function _build_norm_mpo(grid, cutoff=1; tol=1e-10, getW=false, h=Defaults.h(grid))
    # hard-coded for tent functions
    N = length(grid) - 2 # number of tents
    overlap_matrix = tent_basis_overlap_matrix(grid; h=h)
    coefficient_matrix = single_particle_transformation(overlap_matrix)
    two_site_matrices = two_site_decomposition(coefficient_matrix)
    two_site_gates = construct_two_site_gates(two_site_matrices, cutoff)

    local_tensors = Vector{TensorMap}(undef, N)

    # first tensor: U*S
    U, S, V = tsvd(two_site_gates[1], ((1, 3), (2, 4)), trunc=truncerr(tol))
    US = U * S
    local_tensors[1] = insertleftunit(transpose(US, ((1,), (2, 3))), 1)
    prevV = V

    # intermediate tensors: V_prev * U * S_current
    for i in 2:N-1
        U, S, V = tsvd(two_site_gates[i], ((1, 3), (2, 4)), trunc=truncerr(tol))
        US = U * S
        @tensor local_tensors[i][-1 -2; -3 -4] := prevV[-1 1 -3] * US[-2 1 -4]
        prevV = V
    end

    # last tensor: V
    local_tensors[N] = insertrightunit(transpose(prevV, ((1, 2), (3,))), 3)

    W = FiniteMPO(collect(typeof(first(local_tensors)), local_tensors))
    return (getW) ? W : _myprod(conj(W), W)
end

## experimental/hacky

function build_cutoff_projector(m, n, N, base_type=FiniteMPO)
    local_data = zeros(n, m)
    setindex!.(Ref(local_data), 1., 1:min(m, n), 1:min(m, n))
    local_tensor = insertrightunit(insertleftunit(TensorMap(local_data, ℂ^n ← ℂ^m), 1), 3)
    return base_type(collect(typeof(local_tensor), [deepcopy(local_tensor) for _ in 1:N]))
end

function change_mpo_physical_space(mpo, m; side=:both, trunctol=nothing)
    n_out = dim(space(first(mpo), 2))
    n_in = dim(space(first(mpo), 3))
    N = length(mpo)
    base_type = (mpo isa FiniteMPO) ? FiniteMPO : InfiniteMPO
    p_out = build_cutoff_projector(n_out, m, N, base_type)
    p_in = build_cutoff_projector(m, n_in, N, base_type)

    if side === :both
        if n_out != n_in
            throw(ArgumentError("pOut ($n_out) and pIn ($n_in) dimensions differ. Specify :in or :out instead of :both."))
        end
        mpo = _myprod(p_out, _myprod(mpo, p_in, trunctol), trunctol)
    elseif side === :out
        mpo = _myprod(p_out, mpo, trunctol)
    elseif side === :in
        mpo = _myprod(mpo, p_in, trunctol)
    else
        throw(ArgumentError("Invalid side: $side. Use :in, :out, or :both."))
    end

    return mpo
end

function build_norm_mpo(grid, cutoff=1; cutoff_buffer=cutoff, trunctol=1e-10, kwargs...)
    W = change_mpo_physical_space(_build_norm_mpo(grid, cutoff + cutoff_buffer, getW=true; kwargs...), cutoff + 1, side=:in, trunctol=trunctol)
    return _myprod(conj(W), W)
end

# equivalent to exponential, implemented just for testing purposes
# function _construct_two_site_gates_alt(Ms, cutoff)
#     function construct_two_site_gate(M, cutoff)
#         d = cutoff + 1
#         # Precompute factorials to avoid overhead
#         fac = [factorial(big(i)) for i in 0:2d]
#         α, β, γ, δ = M

#         # computes <n1, n2 | W | m1, m2>
#         function weight(n1, n2, m1, m2)
#             n1 + n2 == m1 + m2 || return 0.0

#             val = 0.0
#             # k is the number of particles that "stay" in the first channel
#             for k in max(0, n1 - m2, m1 - n2):min(n1, m1)
#                 den = fac[k+1] * fac[n1-k+1] * fac[m1-k+1] * fac[n2-m1+k+1]
#                 num = (α^k) * (β^(n1 - k)) * (γ^(m1 - k)) * (δ^(n2 - m1 + k))
#                 val += num / den
#             end
#             return val * sqrt(fac[n1+1] * fac[n2+1] * fac[m1+1] * fac[m2+1])
#         end

#         return TensorMap(Float64.([weight(m1, m2, n1, n2) for n1 = 0:d-1, n2 = 0:d-1, m1 = 0:d-1, m2 = 0:d-1]), ℂ^d ⊗ ℂ^d ← ℂ^d ⊗ ℂ^d)
#     end

#     return [construct_two_site_gate(M, cutoff) for M in Ms]
# end