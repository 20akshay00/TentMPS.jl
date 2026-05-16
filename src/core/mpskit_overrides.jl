# some overrides to MPSKit functions that should eventually be submitted as a PR upstream

using MPSKit: MPOTensor, check_length, fuse_mul_mpo
using TensorKit: notrunc, SDD

# with TentMPS, the norm tensor is extensive in system size, so we should avoid propagating the scaling factors from the SVD through the chain as it will grow exponentially 
function _safe_changebonds!(mpo::FiniteMPO, alg::SvdCut)
    length(mpo) == 1 && return mpo

    O_left = transpose(mpo[1], ((3, 1, 2), (4,)))
    local O_right
    for i in 2:length(mpo)
        U, S, V, = tsvd!(O_left; trunc=alg.trscheme, alg=alg.alg_svd)
        f = sqrt(norm(S))
        U *= f
        S /= f
        @inbounds mpo[i-1] = transpose(U, ((2, 3), (1, 4)))
        if i < length(mpo)
            @plansor O_left[-3 -1 -2; -4] := S[-1; 1] * V[1; 2] * mpo[i][2 -2; -3 -4]
        else
            @plansor O_right[-1; -3 -4 -2] := S[-1; 1] * V[1; 2] * mpo[end][2 -2; -3 -4]
        end
    end

    for i in (length(mpo)-1):-1:1
        U, S, V, = tsvd!(O_right; trunc=alg.trscheme, alg=alg.alg_svd)
        f = sqrt(norm(S))
        V *= f
        S /= f
        @inbounds mpo[i+1] = transpose(V, ((1, 4), (2, 3)))
        if i > 1
            @plansor O_right[-1; -3 -4 -2] := mpo[i][-1 -2; -3 2] * U[2; 1] * S[1; -4]
        else
            @plansor _O[-1 -2; -3 -4] := mpo[1][-1 -2; -3 2] * U[2; 1] * S[1; -4]
            @inbounds mpo[1] = _O
        end
    end

    return mpo
end

# changed to used _safe_changebonds! instead of MPSKit.changebonds!
function _safe_prod(mpo1::FiniteMPO{<:MPOTensor}, mpo2::FiniteMPO{<:MPOTensor}, trunctol=nothing)
    N = check_length(mpo1, mpo2)
    (S = spacetype(mpo1)) == spacetype(mpo2) || throw(SectorMismatch())

    if (left_virtualspace(mpo1, 1) != oneunit(S) || left_virtualspace(mpo2, 1) != oneunit(S)) ||
       (right_virtualspace(mpo1, N) != oneunit(S) || right_virtualspace(mpo2, N) != oneunit(S))
        @warn "left/right virtual space is not trivial, fusion may not be unique"
        # this is a warning because technically any isomorphism that fuses the left/right
        # would work and for now I dont feel like figuring out if this is important
    end

    O = map(fuse_mul_mpo, parent(mpo1), parent(mpo2))
    return _safe_changebonds!(FiniteMPO(O), SvdCut(; trscheme=isnothing(trunctol) ? notrunc() : truncerr(trunctol)))
end

########################################################
# everything below is cursed and is only a temporary solution to force MPS 
# and MPOs into a vector interface for use with LOBPCG geneigsolve
########################################################

import LinearAlgebra: mul!
import KrylovKit: geneigsolve
using KrylovKit: ConvergenceInfo

# TYPE PIRACY!!
function input_space(h::MPSKit.MPODerivativeOperator)

    V_l_raw = left_virtualspace(h.leftenv)
    V_l = V_l_raw isa SumSpace ? only(V_l_raw) : V_l_raw

    V_r_raw = right_virtualspace(h.rightenv)
    V_r = V_r_raw isa SumSpace ? only(V_r_raw) : V_r_raw

    V_o = prod(physicalspace, h.operators)

    input_spaces = V_l ⊗ V_o ← V_r

    if length(codomain(input_spaces).spaces) == 3
        input_spaces = TensorKit.permute(input_spaces, ((1, 2), (4, 3)))
    end

    return input_spaces
end

function input_size(h::MPSKit.MPODerivativeOperator)
    s = input_space(h)
    Tuple([dims(codomain(s))...; dims(domain(s))...])
end

function Base.size(h::MPSKit.MPODerivativeOperator)
    V_l = left_virtualspace(h.leftenv)
    V_r = right_virtualspace(h.rightenv)
    V_o = prod(physicalspace, h.operators)

    return (dim(V_l), dim(V_o), dim(V_r))
end

Base.size(h::MPSKit.MPODerivativeOperator, i::Integer) = size(h)[i]

function LinearAlgebra.mul!(C, A::MPSKit.MPODerivativeOperator, X)
    Xt = TensorMap(reshape(X, input_size(A)), input_space(A))
    Ct = A * Xt
    C .= Ct.data
end