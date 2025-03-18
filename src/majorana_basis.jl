using QuantumDots: nbr_of_fermions

abstract type AbstractMajoranaBasis <: QuantumDots.AbstractBasis end
abstract type BasisNorm end
struct HilbertNorm <: BasisNorm end
struct FrobeniusNorm <: BasisNorm end

(::HilbertNorm)(A::AbstractMatrix) = sqrt(hilbert_scalar_product(A, A))
(::FrobeniusNorm)(A::AbstractMatrix) = norm(A)

"""
    SingleParticleMajoranaBasis(fermion_basis, majorana_labels=(:+, :-))

Create a single particle Majorana basis from a fermion basis. Labels for the Majoranas can be provided.
"""
struct SingleParticleMajoranaBasis{D, B, N} <: AbstractMajoranaBasis
    dict::D
    fermion_basis::B
    basis_norm::N
    function SingleParticleMajoranaBasis(fermion_basis::B, basis_labels=standard_bas_labels(fermion_basis, (:+,:-)), basis_norm=HilbertNorm()) where B <: FermionBasis
        N = length(fermion_basis)
        length(basis_labels) == 2 * N || throw(ErrorException("Number of majoranas is not twice the fermion number"))
        majs = reduce(vcat, [[f + f', 1im * (f - f')] for f in fermion_basis]) # to get correct ordering
        majs = _normalize_basis(majs, basis_norm)
        d = OrderedDict(zip(basis_labels, values(majs)))
        new{typeof(d),B, typeof(basis_norm)}(d, fermion_basis, basis_norm)
    end
end

_get_basis_norm(γ::SingleParticleMajoranaBasis) = γ.basis_norm
_normalize_basis(majs, basis_norm::BasisNorm) = map(maj -> maj / basis_norm(maj), majs)

function SingleParticleMajoranaBasis(fermion_basis::FermionBasis, maj_flavors::NTuple{2}, basis_norm=HilbertNorm())
    basis_labels = standard_bas_labels(fermion_basis, maj_flavors)
    return SingleParticleMajoranaBasis(fermion_basis, basis_labels, basis_norm)
end

function SingleParticleMajoranaBasis(nbr_of_majoranas::Int, basis_labels=1:nbr_of_majoranas, basis_norm=HilbertNorm())
    iseven(nbr_of_majoranas) || throw(ErrorException("Number of majoranas must be even"))
    length(basis_labels) == nbr_of_majoranas || throw(ErrorException("Length of labels must match nbr_of_majoranas"))
    c = FermionBasis(1:div(nbr_of_majoranas, 2); qn=QuantumDots.parity)
    SingleParticleMajoranaBasis(c, basis_labels, basis_norm)
end

function standard_bas_labels(fermion_basis, maj_flavors)
    reduce(vcat, [[(label..., maj_flavors[1]), (label..., maj_flavors[2])] for label in keys(fermion_basis)])
end

@testitem "SingleParticleMajoranaBasis without fermions" begin
    using QuantumDots, LinearAlgebra
    nbr_of_majoranas = 6
    basis_labels = 0:5
    γ = SingleParticleMajoranaBasis(nbr_of_majoranas, basis_labels)
    i = rand(basis_labels)
    for j in basis_labels
        @test γ[j]' == γ[j]
        anti_comm = γ[i]*γ[j] + γ[j]*γ[i]
        if i == j
            @test anti_comm == 2I
        else
            @test anti_comm == 0I
        end
    end
end

"""
    full_many_body_majorana_labels(single_particle_labels, max_combinations)

All combinations of many body Majorana labels with products of lengths in combination_lengths.
"""
function full_many_body_majorana_labels(single_particle_labels, combination_lengths::AbstractVector)
    mapreduce(length -> collect(combinations(single_particle_labels, length)), vcat, combination_lengths)
end

@testitem "Many Body Majorana labels" begin
    using QuantumDots
    c = FermionBasis(1:2)
    γ = SingleParticleMajoranaBasis(c)
    labels = Majoranas.full_many_body_majorana_labels(collect(keys(γ)), [2])
    @test length(labels) == binomial(4, 2) # Number of pairs from 4 Majoranas
    empty_label = Majoranas.full_many_body_majorana_labels(collect(keys(γ)), [0])
    @test isempty(only(empty_label))
end

"""
To ensure that individual basis elememts square to one and are Hermitian.
Gives phases (0,1), (1,1), (2,i), (3, -i), (4, -1)...
"""
manybody_majorana_phase(nbr_of_factors) = 1im^(nbr_of_factors*(nbr_of_factors - 1) / 2)

"""
Takes a list of single particle Majoranas and a single particle basis and constructs the corresponding many body Majorana.
"""
function labels_to_manybody_majorana(labels, γ)
    if isempty(labels)
        id_size = size(first(values(γ.dict)))
        return I(first(id_size))
    end
    phase = manybody_majorana_phase(length(labels))
    return phase * mapreduce(label -> γ[label], *, labels)
end

"""
Constructs a many body Majorana basis from a single particle basis.
"""
struct ManyBodyMajoranaBasis{M, N}<:AbstractMajoranaBasis
    dict::M
    basis_norm::N
    function ManyBodyMajoranaBasis(γ::SingleParticleMajoranaBasis, combination_lengths::AbstractVector)
        many_body_labels = full_many_body_majorana_labels(collect(keys(γ)), combination_lengths)
        γmb = map(labels -> labels_to_manybody_majorana(labels, γ), many_body_labels)
        γmb = _normalize_basis(γmb, γ.basis_norm)
        d = OrderedDict(zip(Tuple.(many_body_labels), γmb))
        new{typeof(d), typeof(γ.basis_norm)}(d, γ.basis_norm)
    end
end

_get_basis_norm(γmb::ManyBodyMajoranaBasis) = γmb.basis_norm

function ManyBodyMajoranaBasis(γ::SingleParticleMajoranaBasis, max_length::Int=length(γ))
    return ManyBodyMajoranaBasis(γ, 1:2:max_length)
end

@testitem "ManyBodyMajoranaBasis constructor" begin
    using QuantumDots, LinearAlgebra
    c = FermionBasis(1:3)
    γ_sp = SingleParticleMajoranaBasis(c)
    γ_mb = ManyBodyMajoranaBasis(γ_sp, 1)
    @test all(values(γ_sp.dict) .== values(γ_mb.dict))
    @test all(values(γ_sp.dict) .== values(ManyBodyMajoranaBasis(γ_sp, 1).dict))
    @test γ_sp[1,:+] == γ_sp[(1,:+)]
    γ3 = ManyBodyMajoranaBasis(γ_sp, 3:3)
    nbr_of_sp_majoranas(M, i) = length(collect(keys(M))[i])
    @test all([nbr_of_sp_majoranas(γ3, i) == 3 for i in 1:length(γ3)])
    γ_sp_new_label = SingleParticleMajoranaBasis(c, (:x, :y))
    γ_new_label = ManyBodyMajoranaBasis(γ_sp_new_label, 3)
    sp_labels = [label for labelvec in keys(γ_new_label) for label in labelvec]
    @test all([any((:x, :y) .== label[end]) for label in sp_labels])

    γ_mb = ManyBodyMajoranaBasis(γ_sp, 0:6)
    @test γ_mb[()] == I

    γ_sp = SingleParticleMajoranaBasis(6)
    γ_mb = ManyBodyMajoranaBasis(γ_sp)
    @test length(γ_mb) == 32
end

@testitem "Majorana basis norm" begin
    using QuantumDots, LinearAlgebra
    import Majoranas: _get_basis_norm, HilbertNorm, FrobeniusNorm, hilbert_scalar_product
    c = FermionBasis(1:2)
    for basis_norm in (HilbertNorm(), FrobeniusNorm())
        γ = SingleParticleMajoranaBasis(c, (:x, :y), basis_norm)
        γmb = ManyBodyMajoranaBasis(γ, 0:4)
        @test _get_basis_norm(γ) == basis_norm
        @test _get_basis_norm(γmb) == basis_norm
        @test all([basis_norm(γb) == 1 for γb in γmb])
    end
    γ = SingleParticleMajoranaBasis(c, (:x, :y), FrobeniusNorm())
    γmb = ManyBodyMajoranaBasis(γ, 0:4)
    @test norm([tr(γ1' * γ2) for γ1 in γmb, γ2 in γmb] - I) < 1e-10
end

Base.getindex(M::AbstractMajoranaBasis, label)= M.dict[label]
Base.getindex(M::AbstractMajoranaBasis, labels...)= M.dict[labels]

Base.iterate(M::AbstractMajoranaBasis) = iterate(values(M.dict))
Base.iterate(M::AbstractMajoranaBasis, state) = iterate(values(M.dict), state)
Base.keys(M::AbstractMajoranaBasis) = keys(M.dict)
Base.length(M::AbstractMajoranaBasis) = length(M.dict)
