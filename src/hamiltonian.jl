struct HamiltonianBasis{D} <: AbstractMajoranaBasis
    dict::D
    fermion_basis::FermionBasis
end

function HamiltonianBasis(γ::SingleParticleMajoranaBasis)
    dict = ManyBodyMajoranaBasis(γ, 2:2:length(γ)).dict # 0 doesn't have to be in right?
    insert!(dict, empty(first(dict.indices)), copyto!(similar(first(dict), size(first(dict))), I)) # identity matrix
    return HamiltonianBasis(dict, γ.fermion_basis)
end

function ProjectedHamiltonianBasis(γ::SingleParticleMajoranaBasis, parity::Int)
    H = HamiltonianBasis(γ)
    inds = γ.fermion_basis.symmetry.qntoinds[parity]
    projected_mats = map(mat->mat[inds, inds], H.dict.values)
    independent_inds = [(1:div(length(H), 2)-1)..., length(H)] # is this really correct?
    independent_labels = labels(H)[independent_inds]
    return HamiltonianBasis(QuantumDots.Dictionary(independent_labels, projected_mats[independent_inds]), H.fermion_basis)
end

@testitem "HamiltonianBasis" begin
    using QuantumDots
    import QuantumDots: kitaev_hamiltonian
    c = FermionBasis(1:3; qn=QuantumDots.parity)
    γ = SingleParticleMajoranaBasis(c)
    ham_basis = HamiltonianBasis(γ)
    μ, t, Δ, V = rand(4)
    pmmham = kitaev_hamiltonian(c; μ, t, Δ, V)
    dict = Majoranas.matrix_to_dict(ham_basis, pmmham)
    @test dict[[(1, :+), (1, :-)]] ≈ μ/2 - V/4
    @test dict[[(2, :+), (2, :-)]] ≈ μ/2 - V/2 # different value inside bulk
    @test dict[[(3, :+), (3, :-)]] ≈ μ/2 - V/4
    @test dict[[(1, :+), (2, :-)]] ≈ 1/2*(Δ + t)
    @test dict[[(1, :-), (2, :+)]] ≈ 1/2*(Δ - t)
    @test dict[[(1, :+), (1, :-), (2, :+), (2, :-)]] ≈ V/4 # sign is -1 on 4 length basis
end

@testitem "Projected HamiltonianBasis" begin
    γ = SingleParticleMajoranaBasis(6)
    H = HamiltonianBasis(γ)
    Hp = ProjectedHamiltonianBasis(γ, -1)
    # check setdiff(fullmajorana, Hp_labels) and see that none come up in Hp_labels
end
