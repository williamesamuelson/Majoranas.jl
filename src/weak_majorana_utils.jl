function gs_projection_op(oddvec, evenvec)
    return hcat(oddvec, evenvec)
end

function projection_ops(oddvecs, evenvecs)
    P = gs_projection_op(oddvecs[:,1], evenvecs[:,1])
    Q = hcat(oddvecs[:, 2:end], evenvecs[:, 2:end]) # does the order matter?
    return P, Q
end

function projection_ops(states)
    P = gs_projection_op(states[:, 1], states[:, 2])
    Q = states[:, 3:end]
    return P, Q
end

function many_body_content(γbasis::ManyBodyMajoranaBasis, coeffs::AbstractVector)
    M = many_body_content_matrix(γbasis)
    return coeffs'*M*coeffs/norm(coeffs)^2
end

function many_body_content_matrix(γbasis::ManyBodyMajoranaBasis)
    return Diagonal([length(label) for label in eachindex(γbasis)])
end

