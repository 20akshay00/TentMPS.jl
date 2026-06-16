function tent_function(x, n, h, grid)
    dx = step(grid)
    @assert 0 < n < length(grid) - 1 "Grid of N points can only support N-2 basis elements!"
    xc = grid[n+1]
    return max(0.0, (dx - abs(x - xc)) / h)
end

function tent_function_derivative(x, n, h, grid)
    dx = step(grid)
    @assert 0 < n < length(grid) - 1 "Grid of N points can only support N-2 basis elements!"
    xc = grid[n+1]

    cell_idx = searchsortedlast(grid, x)

    if cell_idx == n
        return 1.0 / h
    elseif cell_idx == n + 1
        return -1.0 / h
    else
        return 0.0
    end
end

function tent_basis_overlap_matrix(grid; h=Defaults.h(grid))
    N, dx = length(grid) - 2, step(grid)

    offdiag = fill(dx^3 / (6 * h^2), N - 1)
    diag = fill(2 * dx^3 / (3 * h^2), N)

    return SymTridiagonal(diag, offdiag)
end

function multiply_physical(A, B)

    dimsA = [length(space(A).domain.spaces), length(space(A).codomain.spaces)]
    dimsB = [length(space(B).domain.spaces), length(space(B).codomain.spaces)]

    ## constructs AB (read operator application from top to bottom)
    if (all(dimsA .== 2) && all(dimsB .== 1))
        #     |
        #     B
        #     |

        #     |
        #   --A--
        #     |
        @tensor C[-1 -2; -3 -4] := B[1; -3] * A[-1 -2; 1 -4]
    elseif (all(dimsA .== 1) && all(dimsB .== 2))
        #     |
        #   --B--
        #     |

        #     |
        #     A
        #     |
        @tensor C[-1 -2; -3 -4] := B[-1 1; -3 -4] * A[-2 1]
    elseif (all(dimsA .== 1) && all(dimsB .== 1))
        #     |
        #     B
        #     |

        #     |
        #     A
        #     |
        # C = insertrightunit(insertleftunit(A * B, 1), 3)
        C = A * B
    else
        throw(DomainError("Incompatible tensors!"))
    end

    return C
end

const × = multiply_physical

get_particle_cutoff(state) = dim(space(first(state.AL), 2)) - 1