function majorana_polarization(maj_basis::Majoranas.AbstractMajoranaBasis, γ1::T1, γ2::T2, region) where {T1,T2<:AbstractMatrix}
    γs = map(γ->majorana_coefficients(maj_basis, γ), (γ1, γ2))
    return majorana_polarization(maj_basis, γs..., region)
end

function majorana_polarization(maj_basis::Majoranas.AbstractMajoranaBasis, γ1::T1, γ2::T2, region) where {T1,T2<:AbstractVector}
    regions_indices = find_region_indices(maj_basis, region)
    γ1sq, γ2sq = map(γ->γ[regions_indices].^2, (γ1, γ2))
    N = abs(sum(γ1sq .- γ2sq))
    D = sum(abs.(γ1sq) .+ abs.(γ2sq))
    return (; mp=N/D, mpu=N)
end

function find_region_indices(maj_basis::Majoranas.AbstractMajoranaBasis, region)
    maj_labels = find_sp_labels(maj_basis)
    return mapreduce(region_label->find_label_indices(maj_labels, region_label), vcat, region)
end

function find_label_indices(maj_labels, region_label)
    indices = findall(all(region_label .∈ (maj_label,)) for maj_label in maj_labels)
    return indices
end

function find_sp_labels(maj_basis::ManyBodyMajoranaBasis)
    sp_indices = findall(labelvec -> length(labelvec) == 1, labels(maj_basis))
    return only.(labels(maj_basis)[sp_indices])
end
find_sp_labels(maj_basis::SingleParticleMajoranaBasis) = labels(maj_basis)

@testitem "find_region_indices" begin
    using QuantumDots
    import Majoranas: find_sp_labels, find_label_indices, find_region_indices
    c = FermionBasis(1:3, (:a, :b))
    γ_sp = SingleParticleMajoranaBasis(c)
    γ_mb = ManyBodyMajoranaBasis(c)
    sp_labels = find_sp_labels(γ_sp)
    @test all(sp_labels .== find_sp_labels(γ_mb))
    @test find_label_indices(sp_labels, (1,:a)) == [1, 7]
    @test isempty(find_label_indices(sp_labels, (:c)))
    @test Set(find_region_indices(γ_sp, [1, 2])) == Set([1, 2, 4, 5, 7, 8, 10, 11])
    @test Set(find_region_indices(γ_mb, [1, 2])) == Set([1, 2, 4, 5, 7, 8, 10, 11])
end

@testitem "Majorana polarization" begin
    using QuantumDots, LinearAlgebra
    import QuantumDots: kitaev_hamiltonian
    c = FermionBasis(1:3; qn=QuantumDots.parity)
    pmmham = blockdiagonal(Hermitian(kitaev_hamiltonian(c; μ=0.0, t=1.0, Δ=1.0)), c)
    eig = diagonalize(pmmham)
    fullsectors = QuantumDots.blocks(eig; full=true)
    oddvecs = fullsectors[1].vectors
    evenvecs = fullsectors[2].vectors
    γ_sp = SingleParticleMajoranaBasis(c)
    γ_mb = ManyBodyMajoranaBasis(c)
    γx, γy = Majoranas.single_particle_majoranas(γ_sp, oddvecs[:, 1], evenvecs[:, 1])
    function test_mp(maj_basis, labels, known_mp)
        MP = Majoranas.majorana_polarization(maj_basis, γx, γy, labels)
        @test all(values(MP) .≈ known_mp)
    end
    test_mp(γ_mb, [1, 2], 1)
    test_mp(γ_sp, [1, 2], 1)
    test_mp(γ_mb, [1, 2, 3], 0)
    test_mp(γ_sp, [1, 2, 3], 0)
end
