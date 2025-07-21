# Majoranas

[![Build Status](https://github.com/williamesamuelson/Majoranas.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/williamesamuelson/Majoranas.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/williamesamuelson/Majoranas.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/williamesamuelson/Majoranas.jl)

## Installation on Julia versions below 1.11
This package depends on FermionicHilbertSpaces and AffineRayleighOptimization, which are not registered in the general registry. You may need to manually add them directly by

```julia
using Pkg; Pkg.add(url="https://github.com/cvsvensson/FermionicHilbertSpaces.jl"); Pkg.add(url="https://github.com/williamesamuelson/AffineRayleighOptimization.jl")
```
or by adding the registry
```julia
using Pkg; Pkg.Registry.add(RegistrySpec(url = "https://github.com/williamesamuelson/PackageRegistry"))
```
and add them from there
```julia
using Pkg; Pkg.add(FermionicHilbertSpaces); Pkg.add(AffineRayleighOptimization)
```
