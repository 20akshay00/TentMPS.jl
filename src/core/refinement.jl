# make new MPS on refined grid with vacuum states inserted
function _refine_state_with_vacuum(state)
    N = length(state)
    ALs = push!(state.AL[1:end-1], state.AC[end]) # to preserve norm
    As = Vector{typeof(first(ALs))}(undef, 2 * N + 1)
    vspaces = hcat(space(ALs[1], 1), conj.(space.(ALs, 3))...)
    pspace = space(first(ALs), 2)

    for i in 1:N
        As[2*i-1] = zeros(eltype(first(ALs)), vspaces[i] ⊗ pspace ← vspaces[i])
        As[2*i] = ALs[i]

        χ = dim(vspaces[i])
        As[2*i-1][][:, 1, :] .= Array(I, χ, χ)
    end

    χ = dim(vspaces[end])
    As[2*N+1] = zeros(eltype(first(ALs)), vspaces[end] ⊗ pspace ← vspaces[end])
    As[2*N+1][][:, 1, :] .= Array(I, χ, χ)

    return FiniteMPS(As, normalize=false)
end

function build_refinement_mpo(grid, cutoff::Int; truncate_physical_dim::Bool=true)
    N = length(grid) - 2
    fac = Float64[factorial(big(x)) for x in 0:2*cutoff]

    V_phys_in = ComplexSpace(cutoff + 1)
    V_virt = ComplexSpace(cutoff + 1)
    V_trivial = ComplexSpace(1)

    tensors = map(1:N) do i
        VL = (i == 1) ? V_trivial : V_virt
        VR = (i == N) ? V_trivial : V_virt

        out_dim = (isodd(i) && !truncate_physical_dim) ? (2 * cutoff + 1) : (cutoff + 1)
        V_out = ComplexSpace(out_dim)

        arr = zeros(Float64, dim(VL), out_dim, cutoff + 1, dim(VR))

        for r in 1:dim(VR), d in 1:cutoff+1, dp in 1:out_dim, l in 1:dim(VL)
            nj, njp = d - 1, dp - 1
            q_left = (i == 1) ? 0 : (l - 1)
            q_right = (i == N) ? 0 : (r - 1)

            if isodd(i)
                if nj == 0 && njp == q_left + q_right
                    arr[l, dp, d, r] = sqrt(fac[njp+1])
                end
            else
                if nj == q_left + njp + q_right
                    arr[l, dp, d, r] = sqrt(fac[nj+1] * fac[njp+1]) /
                                       (fac[q_left+1] * fac[njp+1] * fac[q_right+1]) *
                                       2^(njp - 1.5 * nj)
                end
            end
        end
        return TensorMap(arr, VL * V_out, V_phys_in * VR)
    end

    return FiniteMPO(tensors)
end

# expects vacuum inserted state and applies refinement operator
function refine_state(state, grid; trunc_bond=true, kwargs...)
    cutoff = dim(space(state[1], 2)) - 1
    new_grid = range(first(grid), last(grid), 2 * length(grid) - 1)
    P = build_refinement_mpo(new_grid, cutoff; kwargs...)
    new_state = P * _refine_state_with_vacuum(state)
    if trunc_bond
        new_D = maximum(dim.(space.(state.AL[1:end], 1)))
        new_state = changebonds(new_state, SvdCut(trscheme=truncdim(new_D)))
    end

    return new_grid, new_state
end
