abstract type AbstractMajoranaBasis end
using QuantumDots: nbr_of_fermions

"""
    SingleParticleMajoranaBasis(fermion_basis, majorana_labels=(:+, :-))

Create a single particle Majorana basis from a fermion basis. Labels for the Majoranas can be provided.
"""
struct SingleParticleMajoranaBasis{M} <: AbstractMajoranaBasis
    dict::M
    function SingleParticleMajoranaBasis(fermion_basis, majorana_labels=(:+,:-))
        # make sure that labels are in correct order
        labels_plus = map(label->(label..., majorana_labels[1]), QuantumDots.labels(fermion_basis))
        labels_minus = map(label->(label..., majorana_labels[2]), QuantumDots.labels(fermion_basis))
        γ_plus = map(op -> op' + op, fermion_basis)
        γ_minus = map(op -> 1im*(op' - op), fermion_basis)
        d_plus = QuantumDots.dictionary(zip(labels_plus, values(γ_plus)))
        d_minus = QuantumDots.dictionary(zip(labels_minus, values(γ_minus)))
        d = merge(d_plus, d_minus)
        new{typeof(d)}(d)
    end
end

"""
    full_many_body_majorana_labels(single_particle_labels, max_combinations)

All combinations of many body Majorana labels up to products of max_combinations from single particle Majorana labels.
"""
function full_many_body_majorana_labels(single_particle_labels, max_combinations)
    lengths = 1:2:max_combinations
    mapreduce(length->collect(combinations(single_particle_labels, length)), vcat, lengths)
end

"""
To ensure that individual basis elememts square to one and are Hermitian.
"""
manybody_majorana_phase(nbr_of_factors) = 1im^((nbr_of_factors-1)/2) # gives phases 1, i, -1, -i...

"""
Takes a list of single particle Majoranas and a single particle basis and constructs the corresponding many body Majorana.
"""
labels_to_manybody_majorana(labels, γ) = manybody_majorana_phase(length(labels))*mapreduce(label->γ[label...], *, labels)

"""
Constructs a many body Majorana basis from a single particle basis.
"""
struct ManyBodyMajoranaBasis{M}<:AbstractMajoranaBasis
    dict::M
    function ManyBodyMajoranaBasis(γ::SingleParticleMajoranaBasis, max_combinations=length(γ))
        many_body_labels = full_many_body_majorana_labels(labels(γ), max_combinations)
        γmb = map(labels->labels_to_manybody_majorana(labels, γ), many_body_labels)
        d = QuantumDots.dictionary(zip(many_body_labels, γmb))
        new{typeof(d)}(d)
    end
end

"""
Constructor that takes a FermionBasis instead.
"""
function ManyBodyMajoranaBasis(fermion_basis::FermionBasis, max_combinations=2*nbr_of_fermions(fermion_basis), majorana_labels=(:+,:-))
    γ = SingleParticleMajoranaBasis(fermion_basis, majorana_labels)
    return ManyBodyMajoranaBasis(γ, max_combinations)
end

### with SPMajoranas, use either tuple with the label or several arguments to index
Base.getindex(M::SingleParticleMajoranaBasis, i...) = M.dict[i]
Base.getindex(M::SingleParticleMajoranaBasis, t::Tuple)= M[t...]

### with MBMajoranas, use vector with labels or Int to index
Base.getindex(M::ManyBodyMajoranaBasis, t::AbstractVector)= M.dict[t]
Base.getindex(M::ManyBodyMajoranaBasis, i::Int)= M.dict[labels(M)[i]]

Base.iterate(M::AbstractMajoranaBasis) = iterate(M.dict)
Base.iterate(M::AbstractMajoranaBasis, state) = iterate(M.dict, state)
Base.keys(M::AbstractMajoranaBasis) = keys(M.dict)
Base.length(M::AbstractMajoranaBasis) = length(M.dict)
labels(M::AbstractMajoranaBasis) = keys(M).values
nbr_of_majoranas(c::FermionBasis) = 2*nbr_of_fermions(c)
nbr_of_sp_majoranas(M::ManyBodyMajoranaBasis, i::Int) = length(labels(M)[i])