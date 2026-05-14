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

function _build_refinement_mpo(grid, cutoff; tol=1e-16)
    N = length(grid) - 2
    c = 1 / 2√2 # normalization, but it seems irrelevant.

    # blocks for the bulk
    blocks = [[1. c; 0. 1.], [2c 0.; c 1.]]
    gates = construct_two_site_gates(blocks, cutoff)

    UA, SA, VA = tsvd(gates[1], ((1, 3), (2, 4)), trunc=truncerr(tol))
    UB, SB, VB = tsvd(gates[2], ((1, 3), (2, 4)), trunc=truncerr(tol))
    USA, USB = UA * SA, UB * SB

    TL = insertleftunit(transpose(USA, ((1,), (2, 3))), 1)
    @tensor T_odd[-1 -2; -3 -4] := VA[-1 1 -3] * USB[-2 1 -4]
    @tensor T_even[-1 -2; -3 -4] := VB[-1 1 -3] * USA[-2 1 -4]

    # right boundary for odd N
    TRB = insertrightunit(transpose(VB, ((1, 2), (3,))), 3)

    # right boundary for even N
    if iseven(N)
        B_final = [1. c; 0. 2c]
        gate_final = construct_two_site_gates([B_final], cutoff)[1]
        _, _, V_final = tsvd(gate_final, ((1, 3), (2, 4)), trunc=truncerr(tol))
        TR_final = insertrightunit(transpose(V_final, ((1, 2), (3,))), 3)
    end

    tensors = map(1:N) do i
        i == 1 && return TL
        i == N && return iseven(N) ? TR_final : TRB
        return iseven(i) ? T_odd : T_even
    end

    return FiniteMPO(tensors)
end

# with possibility to add cutoff_buffer
function build_refinement_mpo(grid, cutoff; cutoff_buffer=cutoff, trunctol=nothing, kwargs...)
    mpo = _build_refinement_mpo(grid, cutoff + cutoff_buffer; kwargs...)
    return change_mpo_physical_space(mpo, cutoff + 1, trunctol=trunctol)
end