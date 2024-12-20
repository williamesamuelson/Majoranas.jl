
# Based on https://github.com/JuliaArrays/ReadOnlyArrays.jl
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
Base.:*(x::BasisArray, y::BasisArray) = x.basis == y.basis ? BasisArray(x.parent * y.parent, x.basis) : throw(ArgumentError("Basis mismatch"))
Base.:*(x::BasisArray, y::Number) = BasisArray(x.parent .* y, x.basis)
Base.:*(x::Number, y::BasisArray) = BasisArray(x .* y.parent, y.basis)

# addition
# MatrixTypes = Union{<:UniformScaling,<:Matrix,<:SparseArrays.SparseMatrixCSC}
# VecTypes = Union{UniformScaling,Array,SparseMatrixCSC,SparseVector}
Base.:+(x::BMatrix, y::UniformScaling) = BasisArray(x.parent + y, x.basis)
Base.:+(x::UniformScaling, y::BMatrix) = BasisArray(x + y.parent, y.basis)
Base.:+(x::BasisArray, y::Array) = BasisArray(x.parent + y, x.basis)
Base.:+(x::Array, y::BasisArray) = BasisArray(x + y.parent, y.basis)
Base.:+(x::BMatrix, y::SparseMatrixCSC) = BasisArray(x.parent + y, x.basis)
Base.:+(x::SparseMatrixCSC, y::BMatrix) = BasisArray(x + y.parent, y.basis)
Base.:+(x::BVector, y::SparseVector) = BasisArray(x.parent + y, x.basis)
Base.:+(x::SparseVector, y::BVector) = BasisArray(x + y.parent, y.basis)
Base.:+(x::BasisArray, y::BasisArray) = x.basis == y.basis ? BasisArray(x.parent + y.parent, x.basis) : throw(ArgumentError("Basis mismatch"))

# subtraction
Base.:-(x::BMatrix, y::UniformScaling) = BasisArray(x.parent - y, x.basis)
Base.:-(x::UniformScaling, y::BMatrix) = BasisArray(x - y.parent, y.basis)
Base.:-(x::BasisArray, y::Array) = BasisArray(x.parent - y, x.basis)
Base.:-(x::Array, y::BasisArray) = BasisArray(x - y.parent, y.basis)
Base.:-(x::BMatrix, y::SparseMatrixCSC) = BasisArray(x.parent - y, x.basis)
Base.:-(x::SparseMatrixCSC, y::BMatrix) = BasisArray(x - y.parent, y.basis)
Base.:-(x::BVector, y::SparseVector) = BasisArray(x.parent - y, x.basis)
Base.:-(x::SparseVector, y::BVector) = BasisArray(x - y.parent, y.basis)
Base.:-(x::BasisArray, y::BasisArray) = x.basis == y.basis ? BasisArray(x.parent - y.parent, x.basis) : throw(ArgumentError("Basis mismatch"))
Base.:-(x::BasisArray) = BasisArray(-x.parent, x.basis)
Base.:-(x::Number, y::BasisArray) = BasisArray(x - y.parent, y.basis)

Base.adjoint(b::BasisArray) = BasisArray(adjoint(b.parent), b.basis)
Base.transpose(b::BasisArray) = BasisArray(transpose(b.parent), b.basis)
Base.zero(x::BasisArray) = BasisArray(zero(x.parent), x.basis)
Base.one(x::BasisArray) = BasisArray(one(x.parent), x.basis)

QuantumDots.wedge(bs::AbstractVector{<:BasisArray}, b::ManyBodyBasisArrayWrapper) = wedge(bs, b.basis)
QuantumDots.wedge(bs::AbstractVector{<:BasisArray}, b::FermionBasis) = BasisArray(wedge(map(b -> b.parent, bs), map(b -> b.basis, bs), b), b)
QuantumDots.blockdiagonal(b::BasisArray) = BasisArray(blockdiagonal(b.parent, b.basis), b.basis)
QuantumDots.partial_trace(m::Union{BMatrix,BVector}, bsub::QuantumDots.AbstractBasis) = BasisArray(partial_trace(m.parent, bsub, m.basis), bsub)

struct ManyBodyBasisArrayWrapper{B<:QuantumDots.AbstractManyBodyBasis} <: QuantumDots.AbstractManyBodyBasis
    basis::B
end
Base.getindex(b::ManyBodyBasisArrayWrapper, i) = BasisArray(b.basis[i], b.basis)
Base.getindex(b::ManyBodyBasisArrayWrapper, args...) = BasisArray(b.basis[args], b.basis)
Base.keys(b::ManyBodyBasisArrayWrapper) = keys(b.basis)
Base.show(io::IO, ::MIME"text/plain", b::ManyBodyBasisArrayWrapper) = show(io, b)
function Base.show(io::IO, b::ManyBodyBasisArrayWrapper)
    print(io, "ManyBodyBasisArrayWrapper{\n")
    show(b.basis)
    print(io, "}")
end
function Base.iterate(b::ManyBodyBasisArrayWrapper)
    i = iterate(b.basis)
    isnothing(i) && return i
    x, s = i
    (BasisArray(x, b.basis), s)
end
function Base.iterate(b::ManyBodyBasisArrayWrapper, state)
    i = iterate(b.basis, state)
    isnothing(i) && return i
    x, s = i
    (BasisArray(x, b.basis), s)
end
Base.length(b::ManyBodyBasisArrayWrapper) = length(b.basis)
QuantumDots.symmetry(b::ManyBodyBasisArrayWrapper) = QuantumDots.symmetry(b.basis)
QuantumDots.nbr_of_fermions(b::ManyBodyBasisArrayWrapper) = nbr_of_fermions(b.basis)
Base.eltype(b::ManyBodyBasisArrayWrapper) = eltype(b.basis)
Base.keytype(b::ManyBodyBasisArrayWrapper) = keytype(b.basis)
BasisArray(m::AbstractArray, b::ManyBodyBasisArrayWrapper) = BasisArray(m, b.basis)

# function basisarrays(b::FermionBasis{M,D,S}) where {M,D,S}
#     newdict = QuantumDots.OrderedCollections.OrderedDict([k => BasisArray(v, b) for (k, v) in pairs(b.dict)])
#     b = FermionBasis{M,typeof(newdict),S}(newdict, b.symmetry)
# end

@testitem "BasisArray" begin
    using Majoranas: BasisArray, basisarrays, ManyBodyBasisArrayWrapper
    using QuantumDots, LinearAlgebra
    qn = ParityConservation()
    c1 = FermionBasis(1:1; qn) |> ManyBodyBasisArrayWrapper
    c2 = FermionBasis(2:2; qn) |> ManyBodyBasisArrayWrapper
    c12 = FermionBasis(1:2; qn) |> ManyBodyBasisArrayWrapper
    v = BasisArray(rand(2), c1)

    @test c1[1] * v isa BasisArray
    @test c1[1] + c1[1].parent == 2 * c1[1]
    @test c1[1] == c1[1]
    @test iszero(c1[1] - c1[1])
    @test (c1[1] * I * c1[1]' * I) * 2 + 2I + c1[1].parent - 5 * c1[1] isa BasisArray

    @test wedge([c1[1], c2[2]], c12) isa BasisArray
    @test wedge([c1[1], c2[2]], c12) == c12[2] * c12[1]
    rho = c12[1]' * c12[1]
    @test partial_trace(rho, c1) isa BasisArray

    @test length(blockdiagonal(rho).parent.blocks) == 2

    @test diagonalize(rho).original == rho
end