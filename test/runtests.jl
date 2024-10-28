using TestItemRunner

@run_package_tests

@testitem "Majoranas" begin
    using QuantumDots, LinearAlgebra, Random
    Random.seed!(1234)
    c = FermionBasis(1:2, (:a, :b))
    γ = SingleParticleMajoranaBasis(c)
    @test length(γ) == 2 * QuantumDots.nbr_of_fermions(c)
    test_label = (1, :a, :+)
    for γ_op in γ
        @test γ_op == γ_op'
        @test γ_op^2 == I
        γ_test = γ[test_label]
        if γ_op ≠ γ_test
            @test γ_op * γ_test + γ_test * γ_op == 0 * I
        end
    end
    for fermion_label in QuantumDots.labels(c)
        @test 1 / 2 * (γ[(fermion_label..., :+)] - 1im * γ[(fermion_label..., :-)]) == c[fermion_label...]
    end
    γ_mb = ManyBodyMajoranaBasis(γ)
    # nbr of many body basis operators should be binomial(8,1) + binomial(8,3) + ... + binomial(8,7)
    @test length(γ_mb) == mapreduce(l -> binomial(length(γ), l), +, 1:2:length(γ))
    random_index = ceil(Int, length(γ_mb) * rand())
    test_label = labels(γ_mb)[random_index]
    γ_mb_test = γ_mb[random_index]
    @test γ_mb_test == 1im^(length(test_label) * (length(test_label) - 1) / 2) * mapreduce(label -> γ[label], *, test_label)
    @test γ_mb_test' == γ_mb_test
    @test γ_mb_test^2 == I

    coeffs = Majoranas.majorana_coefficients(γ_mb, γ_mb_test)
    @test coeffs == Majoranas.majorana_coefficients(c, γ_mb_test)

    dict_coeffs = coeffs_to_dict(γ_mb, coeffs)
    @test dict_coeffs[test_label] == 1
    @test norm(collect(values(dict_coeffs))) == 1
end

@testitem "Scalar product and utils" begin
    using LinearAlgebra
    paulibasis = Majoranas.pauli_basis()
    off_diag_basis = Majoranas.off_diagonal_basis(14)
    @test isapprox([Majoranas.hilbert_schmidt_scalar_product(σi, σj) for σi in paulibasis, σj in paulibasis], I, atol=1e-6)
    @test isapprox([Majoranas.hilbert_schmidt_scalar_product(Ei, Ej) for Ei in off_diag_basis, Ej in off_diag_basis], I)
end

@testitem "Basic weak Majoranas" begin
    using QuantumDots, LinearAlgebra
    import QuantumDots: kitaev_hamiltonian
    c = FermionBasis(1:2; qn=QuantumDots.parity)
    pmmham = blockdiagonal(Hermitian(kitaev_hamiltonian(c; μ=0.0, t=1.0, Δ=1.0)), c)
    eig = diagonalize(pmmham)
    fullsectors = QuantumDots.blocks(eig; full=true)
    oddvecs = fullsectors[1].vectors
    evenvecs = fullsectors[2].vectors
    γ_sp = SingleParticleMajoranaBasis(c)
    γx, γy = Majoranas.single_particle_majoranas(γ_sp, oddvecs[:, 1], evenvecs[:, 1])
    P, Q = Majoranas.projection_ops(oddvecs, evenvecs)
    σx, σy = [0 1; 1 0], [0 -1im; 1im 0]
    @test isapprox(P' * γx * P, σx)
    @test isapprox(P' * γy * P, σy)
    @test norm(P' * γx * Q) < 1e-10
    # test matrix construction
    γ_mb = ManyBodyMajoranaBasis(c)
    @test all(values(γ_mb) .== values(ManyBodyMajoranaBasis(c, 3)))
    nbr_of_majoranas = 2 * QuantumDots.nbr_of_fermions(c)
    @test length(γ_mb) == mapreduce(l -> binomial(nbr_of_majoranas, l), +, 1:2:nbr_of_majoranas)
    BP, BPQ = Majoranas.construct_complex_matrices(γ_mb, P, Q, Majoranas._def_pauli_comps(γ_mb)[1])
    @test size(BP) == (2, length(γ_mb))
    @test size(BPQ) == (2 * size(Q, 2), length(γ_mb))
    Bcomplex = [BP; BPQ]
    B = [real.(Bcomplex); imag.(Bcomplex)]
    @test B == Majoranas.weak_majorana_constraint_matrix(γ_mb, P, Q, Majoranas._def_pauli_comps(γ_mb)[1])
    rhsx, rhsy = [Majoranas.right_hand_side(σvec, Q) for σvec in ([nothing, 1.0, 0.0, nothing], [nothing, 0.0, 1.0, nothing])]
    a_vec_x = B \ rhsx
    a_vec_y = B \ rhsy
    basic_prob = WeakMajoranaProblem(γ_mb, oddvecs, evenvecs, nothing)
    basic_sols = solve(basic_prob, Majoranas.WM_BACKSLASH())
    @test isapprox(a_vec_x, basic_sols[1])
    @test isapprox(a_vec_y, basic_sols[2])
    γx, γy = map(a_vec -> Majoranas.coeffs_to_matrix(γ_mb, a_vec), (a_vec_x, a_vec_y))
    @test isapprox(P'γx * P, σx)
    @test isapprox(P'γy * P, σy)
    @test norm(P' * γx * Q) < 1e-10
    @test norm(P' * γy * Q) < 1e-10
    @test isapprox(P' * γx^2 * P, I)
    @test isapprox(P' * γy^2 * P, I)
end

@testitem "WeakMajoranaProblem" begin
    using QuantumDots, LinearAlgebra, AffineRayleighOptimization
    import QuantumDots: kitaev_hamiltonian
    import AffineRayleighOptimization: RAYLEIGH_GENEIG, RAYLEIGH_CHOL, RAYLEIGH_EIG, RAYLEIGH_SPARSE
    c = FermionBasis(1:2; qn=QuantumDots.parity)
    pmmham = blockdiagonal(Hermitian(kitaev_hamiltonian(c; μ=0.0, t=1.0, Δ=1.0)), c)
    eig = diagonalize(pmmham)
    fullsectors = QuantumDots.blocks(eig; full=true)
    oddvecs = fullsectors[1].vectors
    evenvecs = fullsectors[2].vectors
    γ_mb = ManyBodyMajoranaBasis(c, 3)
    Q = Majoranas.many_body_content_matrix(γ_mb)

    prob = WeakMajoranaProblem(γ_mb, oddvecs, evenvecs, Majoranas.RayleighQuotient(Q))
    sol1 = solve(prob, RAYLEIGH_CHOL())
    @test all(Majoranas.test_weak_majorana_solution(prob, sol1)[1] .< 1e-10)
    @test all(Majoranas.test_weak_majorana_solution(prob, sol1)[2] .< 1e-10)
    sol2 = solve(prob, RAYLEIGH_GENEIG())
    sol3 = solve(prob, RAYLEIGH_EIG())
    sol4 = solve(prob, RAYLEIGH_SPARSE())
    @test sol1 ≈ sol2
    @test sol1 ≈ sol3
    @test sol1 ≈ sol4


    prob = WeakMajoranaProblem(γ_mb, oddvecs, evenvecs, Majoranas.QuadraticForm(Q))
    sol3 = solve(prob)
    rq(sol, Q) = sol' * Q * sol / (sol' * sol)
    @test rq(sol1[1], Q) ≈ 1
    @test rq(sol1[1], Q) < rq(sol3[1], Q)
    @test (sol1[1]'*Q*sol1[1]) > sol3[1]'*Q*sol3[1]
end
