function majorana_polarization(maj_basis::Majoranas.AbstractMajoranaBasis, γ1::T1, γ2::T2, region::AbstractVector) where {T1,T2<:AbstractMatrix}
    γs = map(γ->majorana_coefficients(maj_basis, γ), (γ1, γ2))
    return majorana_polarization(maj_basis, γs..., region)
end

function majorana_polarization(maj_basis::Majoranas.AbstractMajoranaBasis, γ1::T1, γ2::T2, region::AbstractVector) where {T1,T2<:AbstractVector}
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
    sp_indices = findall(labelvec -> length(labelvec) == 1, collect(keys(maj_basis))) # why do I have to collect here?
    return only.(collect(keys(maj_basis))[sp_indices])
end
find_sp_labels(maj_basis::SingleParticleMajoranaBasis) = collect(keys(maj_basis))

@testitem "find_region_indices" begin
    using QuantumDots
    import Majoranas: find_sp_labels, find_label_indices, find_region_indices
    c = FermionBasis(1:3, (:a, :b))
    γ_sp = SingleParticleMajoranaBasis(c)
    γ_mb = ManyBodyMajoranaBasis(γ_sp)
    sp_labels = find_sp_labels(γ_sp)
    @test all(sp_labels .== find_sp_labels(γ_mb))
    @test find_label_indices(sp_labels, (1,:a)) == [1, 2]
    @test isempty(find_label_indices(sp_labels, (:c)))
    @test Set(find_region_indices(γ_sp, [1, 2])) == Set([1, 2, 3, 4, 7, 8, 9, 10])
    @test Set(find_region_indices(γ_mb, [1, 2])) == Set([1, 2, 3, 4, 7, 8, 9, 10])
end

@testitem "Majorana polarization" begin
    using QuantumDots, LinearAlgebra
    import QuantumDots: kitaev_hamiltonian, BD1_hamiltonian
    cpmm = FermionBasis(1:3; qn=QuantumDots.parity)
    pmmham = blockdiagonal(Hermitian(kitaev_hamiltonian(cpmm; μ=0.0, t=1.0, Δ=1.0)), cpmm)
    function get_majoranas(sp_basis, ham)
        eig = diagonalize(ham)
        fullsectors = QuantumDots.blocks(eig; full=true)
        oddvecs = fullsectors[1].vectors
        evenvecs = fullsectors[2].vectors
        return Majoranas.single_particle_majoranas(sp_basis, oddvecs[:, 1], evenvecs[:, 1])
    end
    γ_sp = SingleParticleMajoranaBasis(cpmm)
    γ_mb = ManyBodyMajoranaBasis(γ_sp)
    γs = get_majoranas(γ_sp, pmmham)
    function test_mp(γs, maj_basis, labels, known_mp)
        MP = Majoranas.majorana_polarization(maj_basis, γs..., labels)
        @test all(map(isapprox(known_mp; atol=1e-5), values(MP)))
    end
    test_mp(γs, γ_mb, [1, 2], 1)
    test_mp(γs, γ_sp, [1, 2], 1)
    test_mp(γs, γ_mb, [1, 2, 3], 0)
    test_mp(γs, γ_sp, [1, 2, 3], 0)

    cbd1 = FermionBasis((1:2), (:↑, :↓); qn=QuantumDots.parity)
    bd1ham = blockdiagonal(Hermitian(BD1_hamiltonian(cbd1; h=1e4, μ=1e4, t=1, Δ=0, Δ1=1, θ=[0, pi/2], U=0, V=0, ϕ=0)), cbd1)
    γ_sp = SingleParticleMajoranaBasis(cbd1)
    γ_mb = ManyBodyMajoranaBasis(γ_sp)
    γs = get_majoranas(γ_sp, bd1ham)
    test_mp(γs, γ_mb, [(1,:↓)], 1)
    test_mp(γs, γ_sp, [(1,:↓)], 1)
    test_mp(γs, γ_mb, [(1,:↓), (2,:↓)], 0)
    test_mp(γs, γ_sp, [(1,:↓), (2,:↓)], 0)

    γ1 = γ_mb[((1,:↑,:+),)] + γ_mb[((2,:↑,:+),)]
    γ2 = γ_mb[((1,:↓,:-),)] + γ_mb[((2,:↓,:-),)]
    map(lab->test_mp((γ1, γ2), γ_mb, [lab], 1), collect(keys(cbd1)))
end
