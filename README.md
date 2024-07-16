# Majoranas

[![Build Status](https://github.com/williamesamuelson/Majoranas.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/williamesamuelson/Majoranas.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/williamesamuelson/Majoranas.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/williamesamuelson/Majoranas.jl)

## Installation
This package depends on QuantumDots, which is not registered in the general registry. You may need to manually add it directly by 

```julia
using Pkg; Pkg.add(url="https://github.com/cvsvensson/QuantumDots.jl")
```
or by adding the registry 
```julia
using Pkg; Pkg.Registry.add(RegistrySpec(url = "https://github.com/cvsvensson/PackageRegistry"))
```
