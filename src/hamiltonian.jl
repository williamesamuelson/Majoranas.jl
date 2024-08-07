struct HamiltonianBasis{D} <: AbstractMajoranaBasis
    dict::D
    fermion_basis::FermionBasis
    function HamiltonianBasis(γ::SingleParticleMajoranaBasis)
        dict = ManyBodyMajoranaBasis(γ, 2:2:length(γ)).dict # 0 doesn't have to be in right?
        new{typeof(dict)}(dict, γ.fermion_basis)
    end
end

@testitem "HamiltonianBasis" begin
    using QuantumDots
    import QuantumDots: kitaev_hamiltonian
    c = FermionBasis(1:2; qn=QuantumDots.parity)
    γ = SingleParticleMajoranaBasis(c)
    ham_basis = HamiltonianBasis(γ)
    μ, t, Δ, V = rand(4)
    pmmham = kitaev_hamiltonian(c; μ, t, Δ, V)
    dict = Majoranas.matrix_to_dict(ham_basis, pmmham)
    @test dict[[(1, :+), (1, :-)]] ≈ μ/2 - V/4
    @test dict[[(1, :+), (2, :-)]] ≈ 1/2*(Δ + t)
    @test dict[[(1, :-), (2, :+)]] ≈ 1/2*(Δ - t)
    @test dict[[(1, :+), (1, :-), (2, :+), (2, :-)]] ≈ V/4 # sign is -1 on 4 length basis
end
