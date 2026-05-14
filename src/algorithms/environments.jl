using MPSKit: AbstractMPSEnvironments, FiniteEnvironments
import MPSKit: leftenv, rightenv, poison!

struct FiniteLogEnvironments{A,B,C,D,E} <: AbstractMPSEnvironments
    above::A
    operator::B

    ldependencies::Vector{C}
    rdependencies::Vector{C}

    GLs::Vector{D}
    GRs::Vector{D}

    log_L_scales::Vector{E}
    log_R_scales::Vector{E}
end

function log_environments(below, operator, above, leftstart, rightstart)
    N = length(below)
    T = eltype(below.AL[1])
    leftenvs = [i == 0 ? leftstart : similar(leftstart) for i in 0:N]
    rightenvs = [i == N ? rightstart : similar(rightstart) for i in 0:N]

    log_L = zeros(T, N + 1)
    log_R = zeros(T, N + 1)

    s_l = norm(leftenvs[1])
    leftenvs[1] /= s_l
    log_L[1] = log(s_l)

    s_r = norm(rightenvs[end])
    rightenvs[end] /= s_r
    log_R[end] = log(s_r)

    t = similar(below.AL[1])
    return FiniteLogEnvironments(
        above, operator, fill(t, N),
        fill(t, N),
        leftenvs,
        rightenvs,
        log_L,
        log_R
    )
end

function log_environments(
    below::FiniteMPS{S}, O::Union{FiniteMPO,FiniteMPOHamiltonian}, above=nothing
) where {S}
    Vl_bot = left_virtualspace(below, 1)
    Vl_mid = left_virtualspace(O, 1)
    Vl_top = isnothing(above) ? left_virtualspace(below, 1) : left_virtualspace(above, 1)
    leftstart = isomorphism(storagetype(S), Vl_bot ⊗ Vl_mid' ← Vl_top)

    N = length(below)
    Vr_bot = right_virtualspace(below, N)
    Vr_mid = right_virtualspace(O, N)
    Vr_top = isnothing(above) ? right_virtualspace(below, N) : right_virtualspace(above, N)
    rightstart = isomorphism(storagetype(S), Vr_top ⊗ Vr_mid ← Vr_bot)

    return log_environments(below, O, above, leftstart, rightstart)
end

function poison!(ca::FiniteLogEnvironments, ind)
    ca.ldependencies[ind] = similar(ca.ldependencies[ind])
    return ca.rdependencies[ind] = similar(ca.rdependencies[ind])
end

function rightenv(ca::FiniteLogEnvironments, ind, state)
    a = findfirst(i -> !(state.AR[i] === ca.rdependencies[i]), length(state):-1:(ind+1))
    a = isnothing(a) ? nothing : length(state) - a + 1

    if !isnothing(a)
        for j in a:-1:(ind+1)
            above = isnothing(ca.above) ? state.AR[j] : ca.above.AR[j]
            ca.GRs[j] = TransferMatrix(above, ca.operator[j], state.AR[j]) * ca.GRs[j+1]

            s = norm(ca.GRs[j])
            ca.GRs[j] /= s
            ca.log_R_scales[j] = ca.log_R_scales[j+1] + log(s)

            ca.rdependencies[j] = state.AR[j]
        end
    end

    return ca.GRs[ind+1]
end

function leftenv(ca::FiniteLogEnvironments, ind, state)
    a = findfirst(i -> !(state.AL[i] === ca.ldependencies[i]), 1:(ind-1))

    if !isnothing(a)
        for j in a:(ind-1)
            above = isnothing(ca.above) ? state.AL[j] : ca.above.AL[j]
            ca.GLs[j+1] = ca.GLs[j] *
                          TransferMatrix(above, ca.operator[j], state.AL[j])

            s = norm(ca.GLs[j+1])
            ca.GLs[j+1] /= s
            ca.log_L_scales[j+1] = ca.log_L_scales[j] + log(s)

            ca.ldependencies[j] = state.AL[j]
        end
    end

    return ca.GLs[ind]
end

function local_log_scale(ca::FiniteLogEnvironments, ind)
    return ca.log_L_scales[ind] + ca.log_R_scales[ind+1]
end

# assumes all environments are computed already
function global_log_scale(ca::FiniteLogEnvironments)
    return ca.log_L_scales[end]
end