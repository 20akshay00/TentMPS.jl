# TentMPS.jl

[![Build Status](https://github.com/20akshay00/TentMPS.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/20akshay00/TentMPS.jl/actions/workflows/CI.yml?query=branch%3Amain)

**TentMPS** is a framework for extracting ground state properties of 1D continuum quantum many-body systems. It integrates first-order finite-element expansions (i.e, linear "tent" functions) with the Matrix Product State (MPS) formalism. By using a finite-element basis rather than standard finite-difference grids, TentMPS provides a **strictly variational** treatment of the continuum limit, ensuring that ground-state energies remain a rigorous upper bound. 

The current implementation supports the following single species bosonic model;

$$
\hat{H} = \int dx \ \hat{\Psi}^{\dagger}(x) \left[ -\frac{1}{2} \partial_x^2 + V(x) - \mu \right] \hat{\Psi}(x) + g\int dx \ \hat{\Psi}^{\dagger}(x)\hat{\Psi}^{\dagger}(x)\hat{\Psi}(x)\hat{\Psi}(x)
$$

> [!WARNING]
> **Type Piracy Alert:** This package currently performs type piracy on `MPSKit.jl` to extend specific solver behaviors. Use with caution.

## Installation
```julia
using Pkg
Pkg.add(url="https://github.com/20akshay00/TentMPS.jl")
```

## Usage

Following is the general workflow to find the groundstate for a specified potential $V(x)$.
```julia
cutoff, xmax, L = 2, 7., 50
xs = range(-xmax, xmax, L + 1)

g, μ = 10., 10.

# generate operator mpos
@time Hn = build_norm_mpo(xs, cutoff)
@time Ht = construct_hamiltonian(Hn, xs, cutoff; g=g, μ=μ, V=x->0.5x^2)

bond_dimension = 15
state, = find_generalized_groundstate(state, Ht, Hn, DMRG2(maxiter=1, tol=1e-5, verbosity=3, alg_eigsolve=LOBPCG(maxiter=5000), trscheme=truncdim(bond_dimension)))
state, = find_generalized_groundstate(state, Ht, Hn, DMRG(maxiter=2000, tol=1e-5, verbosity=3, alg_eigsolve=LOBPCG(maxiter=5000, ignore_warnings=true)))

```

The package also implements a refinement scheme to interpolate the state onto a finer grid.

```julia
refined_xs, refined_state = refine_state(state, xs)
```

For additional scripts demonstrating the usage of this package, take a look at [20akshay00/TentMPSAnalysis](https://github.com/20akshay00/TentMPSAnalysis). 