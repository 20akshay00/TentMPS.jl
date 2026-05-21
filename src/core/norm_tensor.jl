# non unitary linear transformation to an orthonormal basis to account for non-canonical commutational relations
# convention: always upper triangular, i.e, a_j† = ∑(j ≤ i) c_i† M_ij
single_particle_transformation(commutator_matrix) = (cholesky(commutator_matrix).L)'

# indexing scheme to go from linear to R-tuple {Q}
function index_to_Q(q::Int, dims::Vector{Int})
    R = length(dims)
    Q = zeros(Int, R)
    temp = q - 1
    for r in 1:R
        Q[r] = temp % dims[r]
        temp = div(temp, dims[r])
    end
    return Q
end

# hard-coded for tent functions, but is in principle very general
function build_norm_mpo(grid, cutoff::Int; h=Defaults.h(grid), getW=false)
    L = length(grid) - 2 # number of tents

    overlap_matrix = tent_basis_overlap_matrix(grid; h=h)
    M = single_particle_transformation(overlap_matrix)

    # extract the bandwidth R
    R = 0
    for col in 1:size(M, 2), row in 1:size(M, 1)
        if abs(M[row, col]) > 1e-12
            R = max(R, col - row)
        end
    end

    max_fac = max(cutoff, (R + 1) * cutoff)
    fac = Float64[factorial(big(x)) for x in 0:max_fac]

    # virtual bond dimension
    dims = [(R - r + 1) * cutoff + 1 for r in 1:R]
    χ = prod(dims)

    V_phys_in = ComplexSpace(cutoff + 1)
    V_virt = ComplexSpace(χ)
    V_trivial = ComplexSpace(1)

    tensors = map(1:L) do i
        VL = (i == 1) ? V_trivial : V_virt
        VR = (i == L) ? V_trivial : V_virt

        out_dim = ((R + 1) * cutoff + 1)
        V_out = ComplexSpace(out_dim)

        arr = zeros(Float64, dim(VL), out_dim, cutoff + 1, dim(VR))

        for r_idx in 1:dim(VR), d in 1:cutoff+1, dp in 1:out_dim, l_idx in 1:dim(VL)
            nj, njp = d - 1, dp - 1

            # map 1D virtual indices to Q-tuples
            Q_left = (i == 1) ? zeros(Int, R) : index_to_Q(l_idx, dims)
            Q_right = (i == L) ? zeros(Int, R) : index_to_Q(r_idx, dims)

            # solve the local transition equations for path variables k
            k = zeros(Int, R + 1)
            k[1] = njp - Q_right[1]
            for r_sub in 1:R
                q_right_next = (r_sub == R) ? 0 : Q_right[r_sub+1]
                k[r_sub+1] = Q_left[r_sub] - q_right_next
            end

            # evaluate the matrix elements if constraints are satisfied
            if all(k .>= 0) && sum(k) == nj
                multinomial = fac[nj+1] / prod(fac[val+1] for val in k)

                m_prod = 1.0
                for r_sub in 0:R
                    row = i - r_sub
                    if row < 1 || row > size(M, 1)
                        if k[r_sub+1] > 0
                            m_prod = 0.0
                            break
                        end
                    else
                        m_prod *= M[row, i]^k[r_sub+1]
                    end
                end

                arr[l_idx, dp, d, r_idx] = sqrt(fac[njp+1] / fac[nj+1]) * multinomial * m_prod
            end
        end

        return TensorMap(arr, VL * V_out, V_phys_in * VR)
    end

    W = FiniteMPO(tensors)
    (getW) && return W
    return _safe_prod(conj(W), W)
end