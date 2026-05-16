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

"""
    MPODerivativeLinearMap(op::MPODerivativeOperator)

A wrapper that transforms an `MPODerivativeOperator` (which acts on `TensorMap`s)
into an `AbstractMatrix` interface (which acts on flattened `Arrays`).
This enables use with LOBPCG.
"""
struct MPODerivativeLinearMap{T}
    inner_op::T
    in_space::Any
    in_size::Tuple
    total_dim::Int
end

function MPODerivativeLinearMap(op::MPSKit.MPODerivativeOperator)
    v_l_raw = left_virtualspace(op.leftenv)
    v_l = v_l_raw isa SumSpace ? only(v_l_raw) : v_l_raw

    v_r_raw = right_virtualspace(op.rightenv)
    v_r = v_r_raw isa SumSpace ? only(v_r_raw) : v_r_raw

    v_p = prod(physicalspace, op.operators)

    space = v_l ⊗ v_p ← v_r
    if length(codomain(space).spaces) == 3
        space = TensorKit.permute(space, ((1, 2), (4, 3)))
    end

    sz = Tuple([dims(codomain(space))...; dims(domain(space))...])
    dim = prod(sz)

    return MPODerivativeLinearMap{typeof(op)}(op, space, sz, dim)
end

Base.size(A::MPODerivativeLinearMap) = (A.total_dim, A.total_dim)
Base.size(A::MPODerivativeLinearMap, i::Int) = i <= 2 ? A.total_dim : 1
Base.eltype(::MPODerivativeLinearMap) = ComplexF64

function LinearAlgebra.mul!(C::AbstractMatrix, A::MPODerivativeLinearMap, X::AbstractMatrix)
    for i in 1:size(X, 2)
        Xt = TensorMap(reshape(view(X, :, i), A.in_size), A.in_space)
        Ct = A.inner_op * Xt
        C[:, i] .= vec(Ct.data)
    end
    return C
end

