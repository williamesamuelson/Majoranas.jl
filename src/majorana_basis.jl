abstract type BasisNorm end
struct HilbertNorm <: BasisNorm end
struct FrobeniusNorm <: BasisNorm end

(::HilbertNorm)(A::AbstractMatrix) = sqrt(hilbert_scalar_product(A, A))
(::FrobeniusNorm)(A::AbstractMatrix) = norm(A)

scalar_product(A, B, ::HilbertNorm) = hilbert_scalar_product(A, B)
scalar_product(A, B, ::FrobeniusNorm) = tr(A'*B)
hilbert_scalar_product(A, B) = tr(A'*B)/size(B,2)

"""
    full_many_body_majorana_labels(single_particle_labels, max_combinations)

All combinations of many body Majorana labels with products of lengths in combination_lengths.
"""
function full_many_body_majorana_labels(single_particle_labels, combination_lengths::AbstractVector)
    mapreduce(length -> collect(combinations(single_particle_labels, length)), vcat, combination_lengths)
end
function symbolic_many_body_majorana_basis(labels; combination_lengths=1:length(labels), filter=ls -> true)
    @majoranas γ
    labels = full_many_body_majorana_labels(labels, combination_lengths)
    bs = [manybody_majorana_phase(length(ls)) * prod(l -> γ[l], ls) for ls in labels if filter(ls)]
    if filter(()) # Check if the identity matrix should be included
        bs = Vector{Any}(bs)
        pushfirst!(bs, 1)
    end
    bs
end
function many_body_majorana_basis(H::MajoranaHilbertSpace, norm::BasisNorm=FrobeniusNorm(); kwargs...)
    labels = collect(keys(H.majoranaindices))
    basissops = symbolic_many_body_majorana_basis(labels; kwargs...)
    map(Hermitian ∘ (x -> x / norm(x)) ∘ Base.Fix2(matrix_representation, H), basissops)
end

even_many_body_majorana_basis(H::MajoranaHilbertSpace, args...; kwargs...) =
    many_body_majorana_basis(H, args...; kwargs..., filter=iseven ∘ length)
odd_many_body_majorana_basis(H::MajoranaHilbertSpace, args...; kwargs...) =
    many_body_majorana_basis(H, args...; kwargs..., filter=isodd ∘ length)


@testitem "Many-body majorana basis" begin
    import Majoranas: symbolic_many_body_majorana_basis, many_body_majorana_basis
    using FermionicHilbertSpaces, LinearAlgebra
    labels = vec(collect(Base.product(1:4, (:+, :-))))
    H = FermionicHilbertSpaces.majorana_hilbert_space(labels)

    # Symbolic
    basissops = symbolic_many_body_majorana_basis(labels)
    @test length(basissops) == 2^length(labels)
    @test all(iszero ∘ (x -> x' - x), basissops)

    # Numeric
    matrices = many_body_majorana_basis(H)
    @test all(iszero ∘ (x -> x' - x), matrices)
    @test all(ishermitian, matrices)
    @test all(x -> tr(x'x) ≈ 1, matrices)

    # Different norm
    matrices = many_body_majorana_basis(H, Majoranas.HilbertNorm())
    @test all(iszero ∘ (x -> x' - x), matrices)
    @test all(ishermitian, matrices)
    @test all(x -> tr(x'x) ≈ 2^4, matrices)
end


"""
To ensure that individual basis elements square to one and are Hermitian.
Gives phases (0,1), (1,1), (2,i), (3, -i), (4, -1)...
"""
manybody_majorana_phase(nbr_of_factors) = 1im^(nbr_of_factors * (nbr_of_factors - 1) / 2)
