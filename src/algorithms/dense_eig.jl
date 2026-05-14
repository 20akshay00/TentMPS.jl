Base.@kwdef struct DenseEig
end

function _dense_eigsolve((h, n), x₀, howmany)
    input_dim = prod(size(x₀[]))

    @tensor h_dense[-1 -2 -3; -4 -5 -6] := h.leftenv[-1 5; -4] * h.operators[1][5 -2; -5 3] * h.rightenv[-6 3; -3]
    @tensor n_dense[-1 -2 -3; -4 -5 -6] := n.leftenv[-1 5; -4] * n.operators[1][5 -2; -5 3] * n.rightenv[-6 3; -3]
    h_matrix = reshape(convert(Array, only(h_dense)), input_dim, input_dim)
    n_matrix = reshape(convert(Array, n_dense), input_dim, input_dim)
    vals, vecs = eigen(h_matrix, n_matrix)
    return vals[1:howmany], [TensorMap(reshape(vecs[:, idx], size(x₀[])), space(x₀)) for idx in 1:howmany]
end

# very bad!
# hard-coded for (D, d, D) TensorMaps for the lowest eigenvalue only!; also `which` is a stub
function KrylovKit.geneigsolve((h, n), x₀, howmany::Int, which, alg::DenseEig)
    vals, vecs = _dense_eigsolve((h, n), x₀, 1)
    numiter = 1
    numops = 1
    normres = 0.
    converged = 1

    return (
        vals,
        vecs,
        ConvergenceInfo(converged, nothing, normres, numiter, numops)
    )
end