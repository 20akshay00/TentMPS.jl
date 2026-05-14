function find_generalized_groundstate(ψi::AbstractFiniteMPS, H, N, alg::DMRG, Henvs=log_environments(ψi, H), Nenvs=log_environments(ψi, N); finalize=nothing)

    ψ = deepcopy(ψi)
    ϵs = map(pos -> calc_generalized_galerkin(pos, ψ, H, N, ψ, Henvs, Nenvs), 1:length(ψ))
    ϵ = maximum(ϵs)
    log = IterLog("DMRG")

    LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ, calc_generalized_energy(ψ, H, N, Henvs, Nenvs))
        for iter in 1:(alg.maxiter)
            alg_eigsolve = updatetol(alg.alg_eigsolve, iter, ϵ)

            zerovector!(ϵs)
            for pos in [1:(length(ψ)-1); length(ψ):-1:2]
                h = AC_hamiltonian(pos, ψ, H, ψ, Henvs)
                n = AC_hamiltonian(pos, ψ, N, ψ, Nenvs)
                _, vec = generalizedfixedpoint((h, n), ψ.AC[pos], :SR, alg_eigsolve)
                ϵs[pos] = max(ϵs[pos], calc_generalized_galerkin(pos, ψ, H, N, ψ, Henvs, Nenvs))
                ψ.AC[pos] = vec
            end
            ϵ = maximum(ϵs)

            if !isnothing(finalize)
                ψ, Henvs, Nenvs = finalize(iter, ψ, H, N, Henvs, Nenvs)::Tuple{typeof(ψ),typeof(Henvs),typeof(Nenvs)}
            end

            if ϵ <= alg.tol
                @infov 2 logfinish!(log, iter, ϵ, calc_generalized_energy(ψ, H, N, Henvs, Nenvs))
                break
            end
            if iter == alg.maxiter
                @warnv 1 logcancel!(log, iter, ϵ, calc_generalized_energy(ψ, H, N, Henvs, Nenvs))
            else
                @infov 3 logiter!(log, iter, ϵ, calc_generalized_energy(ψ, H, N, Henvs, Nenvs))
            end
        end
    end
    return ψ, Henvs, Nenvs, ϵ
end

function find_generalized_groundstate(ψi::AbstractFiniteMPS, H, N, alg::DMRG2, Henvs=log_environments(ψi, H), Nenvs=log_environments(ψi, N); finalize=nothing)
    ψ = deepcopy(ψi)
    ϵs = map(pos -> calc_generalized_galerkin(pos, ψ, H, N, ψ, Henvs, Nenvs), 1:length(ψ))
    ϵ = maximum(ϵs)
    log = IterLog("DMRG2")

    LoggingExtras.withlevel(; alg.verbosity) do
        for iter in 1:(alg.maxiter)
            alg_eigsolve = updatetol(alg.alg_eigsolve, iter, ϵ)
            zerovector!(ϵs)

            # left to right sweep
            for pos in 1:(length(ψ)-1)
                @plansor ac2[-1 -2; -3 -4] := ψ.AC[pos][-1 -2; 1] * ψ.AR[pos+1][1 -4; -3]
                Hac2 = AC2_hamiltonian(pos, ψ, H, ψ, Henvs)
                Nac2 = AC2_hamiltonian(pos, ψ, N, ψ, Nenvs)
                _, newA2center = generalizedfixedpoint((Hac2, Nac2), ac2, :SR, alg_eigsolve)

                al, c, ar, = tsvd!(newA2center; trunc=alg.trscheme, alg=alg.alg_svd)
                normalize!(c)
                v = @plansor ac2[1 2; 3 4] * conj(al[1 2; 5]) * conj(c[5; 6]) * conj(ar[6; 3 4])
                ϵs[pos] = max(ϵs[pos], abs(1 - abs(v)))

                ψ.AC[pos] = (al, complex(c))
                ψ.AC[pos+1] = (complex(c), _transpose_front(ar))
            end

            # right to left sweep
            for pos in (length(ψ)-2):-1:1
                @plansor ac2[-1 -2; -3 -4] := ψ.AL[pos][-1 -2; 1] * ψ.AC[pos+1][1 -4; -3]
                Hac2 = AC2_hamiltonian(pos, ψ, H, ψ, Henvs)
                Nac2 = AC2_hamiltonian(pos, ψ, N, ψ, Nenvs)
                _, newA2center = generalizedfixedpoint((Hac2, Nac2), ac2, :SR, alg_eigsolve)

                al, c, ar, = tsvd!(newA2center; trunc=alg.trscheme, alg=alg.alg_svd)
                normalize!(c)
                v = @plansor ac2[1 2; 3 4] * conj(al[1 2; 5]) * conj(c[5; 6]) * conj(ar[6; 3 4])
                ϵs[pos] = max(ϵs[pos], abs(1 - abs(v)))

                ψ.AC[pos+1] = (complex(c), _transpose_front(ar))
                ψ.AC[pos] = (al, complex(c))
            end

            ϵ = maximum(ϵs)

            if !isnothing(finalize)
                ψ, Henvs, Nenvs = finalize(iter, ψ, H, N, Henvs, Nenvs)::Tuple{typeof(ψ),typeof(Henvs),typeof(Nenvs)}
            end

            if ϵ <= alg.tol
                @infov 2 logfinish!(log, iter, ϵ, calc_generalized_energy(ψ, H, N, Henvs, Nenvs))
                break
            end
            if iter == alg.maxiter
                @warnv 1 logcancel!(log, iter, ϵ, calc_generalized_energy(ψ, H, N, Henvs, Nenvs))
            else
                @infov 3 logiter!(log, iter, ϵ, calc_generalized_energy(ψ, H, N, Henvs, Nenvs))
            end
        end
    end
    return ψ, Henvs, Nenvs, ϵ
end