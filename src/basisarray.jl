
using Base: @propagate_inbounds

struct BasisArray{T,N,A,B} <: AbstractArray{T,N}
    parent::A
    basis::B
    function BasisArray(parent::AbstractArray{T,N}, basis::B) where {T,N,B}
        new{T,N,typeof(parent),B}(parent, basis)
    end
end

BasisArray{T}(parent::AbstractArray{T,N}, basis) where {T,N} = BasisArray(parent, basis)

BasisArray{T,N}(parent::AbstractArray{T,N}, basis) where {T,N} = BasisArray(parent, basis)

BasisArray{T,N,P}(parent::P, basis) where {T,N,P<:AbstractArray{T,N}} = BasisArray(parent, basis)

#--------------------------------------
# aliases

const BVector{T,P} = BasisArray{T,1,P}

BVector(parent::AbstractVector, basis) = BasisArray(parent, basis)

const BMatrix{T,P} = BasisArray{T,2,P}

BMatrix(parent::AbstractMatrix, basis) = BasisArray(parent, basis)

#--------------------------------------
# https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-parentay

Base.size(x::BasisArray, args...) = size(x.parent, args...)

@propagate_inbounds function Base.getindex(x::BasisArray, args...)
    getindex(x.parent, args...)
end
Base.setindex!(x::BasisArray, args...) = setindex!(x.parent, args...)

Base.IndexStyle(::Type{<:BasisArray{T,N,P}}) where {T,N,P} = IndexStyle(P)

Base.iterate(x::BasisArray, args...) = iterate(x.parent, args...)

Base.length(x::BasisArray) = length(x.parent)

Base.similar(x::BasisArray) = BasisArray(similar(x.parent), x.basis)
Base.similar(x::BasisArray, dims::Union{Integer,AbstractUnitRange}...) = BasisArray(similar(x.parent, dims...), x.basis)
Base.similar(x::BasisArray, ::Type{T}, dims::Union{Integer,AbstractUnitRange}...) where {T} = BasisArray(similar(x.parent, T, dims...), x.basis)

Base.axes(x::BasisArray) = axes(x.parent)

function Base.IteratorSize(::Type{<:BasisArray{T,N,P}}) where {T,N,P}
    Base.IteratorSize(P)
end

function Base.IteratorEltype(::Type{<:BasisArray{T,N,P}}) where {T,N,P}
    Base.IteratorEltype(P)
end

function Base.eltype(::Type{<:BasisArray{T,N,P}}) where {T,N,P}
    eltype(P)
end

Base.firstindex(x::BasisArray) = firstindex(x.parent)

Base.lastindex(x::BasisArray) = lastindex(x.parent)

Base.strides(x::BasisArray) = strides(x.parent)

function Base.unsafe_convert(p::Type{Ptr{T}}, x::BasisArray) where {T}
    Base.unsafe_convert(p, x.parent)
end

Base.stride(x::BasisArray, i::Int) = stride(x.parent, i)

Base.parent(x::BasisArray) = x.parent

# multiplication
# Base.:*(x::BasisArray, y::AbstractArray) = BasisArray(x.parent .* y, x.basis)
# Base.:*(x::AbstractArray, y::BasisArray) = BasisArray(x .* y.parent, y.basis)
Base.:*(x::BasisArray, y::BasisArray) = x.basis == y.basis ? BasisArray(x.parent * y.parent, x.basis) : throw(ArgumentError("Basis mismatch"))
Base.:*(x::BasisArray, y::Number) = BasisArray(x.parent .* y, x.basis)
Base.:*(x::Number, y::BasisArray) = BasisArray(x .* y.parent, y.basis)

# addition
Base.:+(x::BasisArray, y::UniformScaling) = BasisArray(x.parent + y, x.basis)
Base.:+(x::UniformScaling, y::BasisArray) = BasisArray(x + y.parent, y.basis)
Base.:+(x::BasisArray, y::BasisArray) = x.basis == y.basis ? BasisArray(x.parent + y.parent, x.basis) : throw(ArgumentError("Basis mismatch"))
Base.:+(x::BasisArray, y::Number) = BasisArray(x.parent + y, x.basis)
Base.:+(x::Number, y::BasisArray) = BasisArray(x + y.parent, y.basis)

# subtraction
Base.:-(x::BasisArray, y::UniformScaling) = BasisArray(x.parent - y, x.basis)
Base.:-(x::UniformScaling, y::BasisArray) = BasisArray(x - y.parent, y.basis)
Base.:-(x::BasisArray, y::BasisArray) = x.basis == y.basis ? BasisArray(x.parent - y.parent, x.basis) : throw(ArgumentError("Basis mismatch"))
Base.:-(x::BasisArray) = BasisArray(-x.parent, x.basis)
Base.:-(x::BasisArray, y::Number) = BasisArray(x.parent - y, x.basis)
Base.:-(x::Number, y::BasisArray) = BasisArray(x - y.parent, y.basis)


Base.adjoint(b::BasisArray) = BasisArray(adjoint(b.parent), b.basis)
Base.transpose(b::BasisArray) = BasisArray(transpose(b.parent), b.basis)
Base.zero(x::BasisArray) = BasisArray(zero(x.parent), x.basis)
Base.one(x::BasisArray) = BasisArray(one(x.parent), x.basis)

QuantumDots.wedge(bs::AbstractVector{<:BasisArray}, b::FermionBasis) = BasisArray(wedge(map(b -> b.parent, bs), map(b -> b.basis, bs), b), b)
QuantumDots.blockdiagonal(b::BasisArray) = BasisArray(blockdiagonal(b.parent, b.basis), b.basis)
QuantumDots.partial_trace(m::Union{BMatrix,BVector}, bsub::QuantumDots.AbstractBasis) = BasisArray(partial_trace(m.parent, bsub, m.basis), bsub)


@testitem "BasisArray" begin
    using Majoranas: BasisArray
    using QuantumDots, LinearAlgebra
    qn = ParityConservation()
    c1 = FermionBasis(1:1; qn)
    c2 = FermionBasis(2:2; qn)
    c12 = FermionBasis(1:2; qn)
    v = BasisArray(rand(2), c1)
    f1 = BasisArray(c1[1], c1)
    @test f1 * v isa BasisArray
    @test f1 + c1[1] == 2 * f1
    @test f1 == f1
    @test iszero(f1 - f1)
    @test (f1 * I * f1' * I) * 2 + 2I + f1 - 5 * f1 isa BasisArray

    f2 = BasisArray(c2[2], c2)
    @test wedge([f1, f2], c12) isa BasisArray
    @test wedge([f1, f2], c12) == c12[2] * c12[1]

    rho = BasisArray(c12[1]' * c12[1], c12)
    @test partial_trace(rho, c1) isa BasisArray

    @test length(blockdiagonal(rho).parent.blocks) == 2

    @test diagonalize(rho).original == rho
end