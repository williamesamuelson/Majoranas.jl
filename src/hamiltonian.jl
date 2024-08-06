struct HamiltonianBasis{D} <: AbstractMajoranaBasis
    dict::D
    fermion_basis::FermionBasis
    function HamiltonianBasis(γ::SingleParticleMajoranaBasis)
        dict = ManyBodyMajoranaBasis(γ, 2:2:length(γ)).dict
        new{typeof(dict)}(dict, γ.fermion_basis)
    end
end
