
hopping(t, f1, f2) = t * f1'f2 + hc
pairing(Δ, f1, f2) = Δ * f2 * f1 + hc
numberop(f) = f'f
coulomb(f1, f2) = f1' * f1 * f2' * f2
_kitaev_2site(f1, f2; t, Δ, V) = hopping(-t, f1, f2) + V * coulomb(f1, f2) + pairing(Δ, f1, f2)
_kitaev_1site(f; μ) = -μ * numberop(f)

getvalue(v::Union{<:AbstractVector,<:Tuple,<:StepRange}, i, N; size=1) = v[i]
getvalue(x::Number, i, N; size=1) = 1 <= i <= N + 1 - size ? x : zero(x)

function kitaev_hamiltonian(c; μ, t, Δ, V)
    N = length(c)
    indices = collect(keys(c))
    h1s = (_kitaev_1site(c[k]; μ=getvalue(μ, j, N)) for (j, k) in enumerate(indices))
    h2s = (_kitaev_2site(c[k], c[indices[mod1(j + 1, N)]]; t=getvalue(t, j, N; size=2), Δ=getvalue(Δ, j, N; size=2), V=getvalue(V, j, N; size=2)) for (j, k) in enumerate(indices))
    sum(h1s) + sum(h2s)
end
