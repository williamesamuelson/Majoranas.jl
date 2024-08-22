abstract type AbstractMajoranaBasis end
using QuantumDots: nbr_of_fermions

"""
    SingleParticleMajoranaBasis(fermion_basis, majorana_labels=(:+, :-))

Create a single particle Majorana basis from a fermion basis. Labels for the Majoranas can be provided.
"""
struct SingleParticleMajoranaBasis{D} <: AbstractMajoranaBasis
    dict::D
    fermion_basis::FermionBasis
    function SingleParticleMajoranaBasis(fermion_basis::FermionBasis, maj_basis_labels=standard_maj_basis_labels(fermion_basis, (:+,:-)))
        N = length(fermion_basis)
        length(maj_basis_labels) == 2 * N || throw(ErrorException("Number of majoranas is not twice the fermion number"))
        majs = reduce(vcat, [[f + f', 1im * (f - f')] for f in fermion_basis]) # to get correct ordering
        d = QuantumDots.dictionary(zip(maj_basis_labels, values(majs)))
        new{typeof(d)}(d, fermion_basis)
    end
end

function SingleParticleMajoranaBasis(fermion_basis::FermionBasis, majorana_labels::NTuple{2})
    maj_basis_labels = standard_maj_basis_labels(fermion_basis, majorana_labels)
    return SingleParticleMajoranaBasis(fermion_basis, maj_basis_labels)
end

function SingleParticleMajoranaBasis(nbr_of_majoranas::Int, maj_basis_labels=1:nbr_of_majoranas)
    iseven(nbr_of_majoranas) || throw(ErrorException("Number of majoranas must be even"))
    length(maj_basis_labels) == nbr_of_majoranas || throw(ErrorException("Length of labels must match nbr_of_majoranas"))
    c = FermionBasis(1:div(nbr_of_majoranas, 2); qn=QuantumDots.parity)
    SingleParticleMajoranaBasis(c, maj_basis_labels)
end

function standard_maj_basis_labels(fermion_basis, maj_labels)
    reduce(vcat, [[(label..., maj_labels[1]), (label..., maj_labels[2])] for label in keys(fermion_basis)])
end

@testitem "SingleParticleMajoranaBasis without fermions" begin
    using QuantumDots, LinearAlgebra
    nbr_of_majoranas = 6
    labs = 0:5
    γ = SingleParticleMajoranaBasis(nbr_of_majoranas, labs)
    i = rand(labs)
    for j in labs
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
    mapreduce(length->collect(combinations(single_particle_labels, length)), vcat, combination_lengths)
end

"""
To ensure that individual basis elememts square to one and are Hermitian.
"""
manybody_majorana_phase(nbr_of_factors) = 1im^(nbr_of_factors*(nbr_of_factors - 1) / 2) # gives phases (0,1), (1,1), (2,i), (3, -i), (4, -1)...

"""
Takes a list of single particle Majoranas and a single particle basis and constructs the corresponding many body Majorana.
"""
labels_to_manybody_majorana(labels, γ) = manybody_majorana_phase(length(labels))*mapreduce(label->γ[label], *, labels)

"""
Constructs a many body Majorana basis from a single particle basis.
"""
struct ManyBodyMajoranaBasis{M}<:AbstractMajoranaBasis
    dict::M
    function ManyBodyMajoranaBasis(γ::SingleParticleMajoranaBasis, combination_lengths::AbstractVector)
        many_body_labels = full_many_body_majorana_labels(labels(γ), combination_lengths)
        γmb = map(labels->labels_to_manybody_majorana(labels, γ), many_body_labels)
        d = QuantumDots.dictionary(zip(many_body_labels, γmb))
        new{typeof(d)}(d)
    end
end

function ManyBodyMajoranaBasis(γ::SingleParticleMajoranaBasis, max_combination_length::Int=length(γ))
    return ManyBodyMajoranaBasis(γ, 1:2:max_combination_length)
end

"""
Constructor that takes a FermionBasis instead.
"""
function ManyBodyMajoranaBasis(fermion_basis::FermionBasis, combination_lengths::AbstractVector, majorana_labels=(:+,:-))
    γ = SingleParticleMajoranaBasis(fermion_basis, majorana_labels)
    return ManyBodyMajoranaBasis(γ, combination_lengths)
end

function ManyBodyMajoranaBasis(fermion_basis::FermionBasis, max_combination_length::Int=2*nbr_of_fermions(fermion_basis), majorana_labels=(:+,:-))
    return ManyBodyMajoranaBasis(fermion_basis, 1:2:max_combination_length, majorana_labels)
end

@testitem "ManyBodyMajoranaBasis constructor" begin
    using QuantumDots
    c = FermionBasis(1:3)
    γ_sp = SingleParticleMajoranaBasis(c)
    γ_mb = ManyBodyMajoranaBasis(c, 1)
    @test γ_sp.dict.values == γ_mb.dict.values
    @test γ_sp.dict.values == ManyBodyMajoranaBasis(γ_sp, 1).dict.values
    γ3 = ManyBodyMajoranaBasis(c, 3:3)
    nbr_of_sp_majoranas(M, i) = length(labels(M)[i])
    @test all([nbr_of_sp_majoranas(γ3, i) == 3 for i in 1:length(γ3)])
    γ_new_label = ManyBodyMajoranaBasis(c, 3, (:x, :y))
    sp_labels = [label for labelvec in labels(γ_new_label) for label in labelvec]
    @test all([any((:x, :y) .== label[end]) for label in sp_labels])

    γ_sp = SingleParticleMajoranaBasis(6)
    γ_mb = ManyBodyMajoranaBasis(γ_sp)
    @test length(γ_mb) == 32
end

### with SPMajoranas, use several arguments to index (or just the int for fermion free majoranas)
#=Base.getindex(M::SingleParticleMajoranaBasis, i::Int) = M.dict[i] # for fermion free majoranas=#
#=Base.getindex(M::SingleParticleMajoranaBasis, args...) = M.dict[args]=#
#=Base.getindex(M::SingleParticleMajoranaBasis, t::Tuple)= M.dict[t]=#
Base.getindex(M::AbstractMajoranaBasis, label)= M.dict[label]

### with MBMajoranas, use vector with labels or Int to index
#=Base.getindex(M::ManyBodyMajoranaBasis, t::AbstractVector)= M.dict[t]=#
Base.getindex(M::ManyBodyMajoranaBasis, i::Int)= M.dict[labels(M)[i]] # do we need this?

Base.iterate(M::AbstractMajoranaBasis) = iterate(M.dict)
Base.iterate(M::AbstractMajoranaBasis, state) = iterate(M.dict, state)
Base.keys(M::AbstractMajoranaBasis) = keys(M.dict)
Base.length(M::AbstractMajoranaBasis) = length(M.dict)
labels(M::AbstractMajoranaBasis) = keys(M).values
#=nbr_of_majoranas(c::FermionBasis) = 2*nbr_of_fermions(c)=#
