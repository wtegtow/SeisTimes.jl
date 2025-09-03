# ========================================
# HELPER
# ========================================

function get_segment_lengths(xk)
    return [norm(xk[i+1] - xk[i]) for i in 1:length(xk)-1]
end;

function get_unit_vector(a, b)
    dir = a - b
    return dir / norm(dir)
end;

function get_grad_t(xk1, xk2, d, g, grad_g)
    ∇x_tk = 1/g .* ( (xk2 - xk1) / d - ( d / g * grad_g) )
    return ∇x_tk
end;

# ========================================
# 2D
# ========================================

function straight_ray_intersections_2D(src, rcv, interface_depths)

    src = Float64.(src)
    rcv = Float64.(rcv)
    src_x, src_z = src
    rcv_x, rcv_z = rcv

    dir = rcv .- src
    dx = dir[1]
    dz = dir[2]

    indices = 1:length(interface_depths)
    if dz < 0
        indices = reverse(indices)
    end

    interface_ids = Int[]
    points = ([[src_x, src_z]])
  
    for idx in indices
        z_int = interface_depths[idx]
        if dz ≈ 0
            continue
        end
        t = (z_int - src_z) / dz
        if 0.0 < t < 1.0
            x = src_x + t * dx 
            push!(points, [x, z_int])
            push!(interface_ids, idx)
        end
    end

    push!(points, [rcv_x,  rcv_z])

    layer_ids = Int[]
    if length(points) > 2
        for k in 1:(length(points) - 1)
            z_mid = 0.5 * (points[k][2] + points[k+1][2])
            lid = sum(z_mid .> interface_depths) + 1
            push!(layer_ids, lid)
        end
    end

    return points, interface_ids, layer_ids
end

function C2D(vp, vs; eps=0., del=0.)
    μ = @. vs^2
    λ = @. vp^2 - 2*μ
    # density normalized stiffness
    c11 = @. ((λ + 2μ) * (2*eps .+ 1))
    c13 = @. (λ + del * (λ + 2μ)) 
    c33 = @. (λ + 2μ) 
    c55 = @. (μ) 
    return (c11 = c11,
            c13 = c13,
            c33 = c33,
            c55 = c55)
end;

function ray_angles2D(r)
    θ = atan(r[2], r[1])
    return θ
end;

function dr_dphi2D(θ)
    dr_dθ = @SVector [-sin(θ), cos(θ)]
    return dr_dθ
end

function Γn2D(c,n)
    Γ = @SMatrix[
         c.c11 * n[1]^2 + c.c55 * n[2]^2  (c.c13 + c.c55) * n[1] * n[2]    ;
        (c.c13 + c.c55) * n[1] * n[2]      c.c55 * n[1]^2 + c.c33 * n[2]^2 ]
    return Γ
end;

function solve_christoffel2D!(VpVs::MVector{2,Float64}, UpUs::MMatrix{2,2,Float64}, c, n)
    Γ = Γn2D(c,n)
    F = eigen(Γ)           
    V = F.values
    U = F.vectors
    # P -> 1, S1 -> 2
    VpVs[1] = sqrt(real(V[2]))  
    VpVs[2] = sqrt(real(V[1]))  
    UpUs[:,1] .= U[:,2]         
    UpUs[:,2] .= U[:,1]       
end;

function scalar_group_velocity2D(phase,c,n)

    phase_index = phase == :P  ? 1 :
                  phase == :S  ? 2 :
                  error("Phase $(phase) not in ['P', 'S1', 'S2']") 

    g = MVector{2,Float64}(undef)
    VpVs = MVector{2,Float64}(undef)
    UpUs = MMatrix{2,2,Float64}(undef)

    solve_christoffel2D!(VpVs,UpUs,c,n)
    P = n ./ VpVs[phase_index]
    ΓU = Γn2D(c, UpUs[:,phase_index])
    @einsum g[i] = ΓU[i,j] * P[j]
    return norm(g)
end;


function grad_g2D(g, r, n, d, dr_dθ)

    M = SMatrix{2,2}(
        d*dr_dθ[1], r[1],
        d*dr_dθ[2], r[2]
    )
    dg_dθ = -g * (dot(dr_dθ, n) / dot(r, n))
    S = SVector{2}(dg_dθ, 0.0)
    ∇g = inv(M) * S
    return ∇g
end;

function assemble_bending_equations_2D(phase, xk, ck)
    K = length(xk) - 2

    # compute for all layers 
    dk = get_segment_lengths(xk)
    rk = [get_unit_vector(xk[i+1], xk[i]) for i in 1:length(xk)-1]
    nk = rk # assuming homogeneous layers 
    Θk = [ray_angles2D(r) for r in rk]
    ∂r_∂Θk = [dr_dphi2D(θ) for θ in Θk ]
    gk = [scalar_group_velocity2D(phase, c, n) for (c,n) in zip(ck, nk)]
    Δgk = [grad_g2D(g,r,n,d,dr_dθ) for (g,r,n,d,dr_dθ) in zip(gk, rk, nk, dk, ∂r_∂Θk)]
    Δtk  = [get_grad_t(xk[k+1], xk[k], dk[k], gk[k], Δgk[k]) for k in 1:K]
    Δtk1 = [get_grad_t(xk[k+2], xk[k+1], dk[k+1], gk[k+1], Δgk[k+1]) for k in 1:K]
    Δt = [(tk[1] - tk1[1]) for (tk, tk1) in zip(Δtk, Δtk1)]

    # assemble 
    F = zeros(K)
    for k in 1:K
        F[k] = Δt[k]
    end
    return F 
end

function ray_residual_2D!(F, p, phase, xk, ck)
    xk_opt = copy(xk)
    for (i, k) in enumerate(2:length(xk)-1)
        xk_opt[k][1] = p[i]  
    end
    F .= assemble_bending_equations_2D(phase, xk_opt, ck)
end;

function traveltime_2D(phase, xk, ck)
    dk = get_segment_lengths(xk)
    rk = [get_unit_vector(xk[i+1], xk[i]) for i in 1:length(xk)-1]
    nk = rk 
    gk = [scalar_group_velocity2D(phase, c, n) for (c,n) in zip(ck, nk)]
    traveltime = sum(d/g for (d,g) in zip(dk, gk))
    return traveltime
end;

function ray_bending_2D(phase, M, interface_depths, src, rcv; verbose=false)

    vp_layers  = @view M[:,1]
    vs_layers  = @view M[:,2]
    eps_layers = @view M[:,3]
    del_layers = @view M[:,4]
    c_layers = [C2D(vp, vs; eps=eps, del=del) for (vp,vs,eps,del) in zip(vp_layers, vs_layers, eps_layers, del_layers)]

    xk, interface_ids, layer_ids = straight_ray_intersections_2D(src, rcv, interface_depths)
    ck = c_layers[layer_ids]

    xk0 = deepcopy(xk)
    if length(xk) == 2
        ; # src & rcv in the same layer
    else
        p0 = [xk[k][1] for k in 2:length(xk)-1]
        res = nlsolve((F, p) -> ray_residual_2D!(F, p, phase, xk, ck), p0)
        if verbose println(res) end
    end 

    t  = traveltime_2D(phase, xk, c_layers)
    t0 = traveltime_2D(phase, xk0, c_layers)

    return (xk0=xk0, xk=xk, t0=t0, t=t)
end;


# ========================================
# 3D
# ========================================

function straight_ray_intersections_3D(src, rcv, interface_depths)
    src = Float64.(src)
    rcv = Float64.(rcv)
    src_x, src_y, src_z = src
    rcv_x, rcv_y, rcv_z = rcv

    dir = rcv .- src
    dx, dy, dz = dir

    indices = 1:length(interface_depths)
    if dz < 0
        indices = reverse(indices)
    end

    points = [[src_x, src_y, src_z]]
    interface_ids = Int[]

    for idx in indices
        z_int = interface_depths[idx]
        if dz ≈ 0
            continue
        end
        t = (z_int - src_z) / dz
        if 0.0 < t < 1.0
            x = src_x + t * dx
            y = src_y + t * dy
            push!(points, [x, y, z_int])
            push!(interface_ids, idx)
        end
    end

    push!(points, [rcv_x, rcv_y, rcv_z])

    layer_ids = Int[]

    if length(points) > 2
        for k in 1:(length(points) - 1)
            z_mid = 0.5 * (points[k][3] + points[k+1][3])
            lid = sum(z_mid .> interface_depths) + 1
            push!(layer_ids, lid)
        end
    end

    return points, interface_ids, layer_ids
end


function C3D(vp, vs ;eps1=0., eps2=0., gam1=0., gam2=0., del1=0., del2=0., del3=0.)
    c33 = @. vp^2
    c55 = @. vs^2
    c11 = @. (2*eps2 + 1) * c33
    c22 = @. (2*eps1 + 1) * c33
    c66 = @. (2*gam1 + 1) * c55
    c44 = @. c66 / (1 + gam2)
    c13 = @. sqrt(2 * c33 * (c33 - c55) * del2 + (c33 - c55)^2) - c55
    c23 = @. sqrt(2 * c33 * (c33 - c44) * del1 + (c33 - c44)^2) - c44
    c12 = @. sqrt(2 * c11 * (c11 - c66) * del3 + (c11 - c66)^2) - c66

    return (
        c11 = c11,
        c22 = c22,
        c33 = c33,
        c44 = c44,
        c55 = c55,
        c66 = c66,
        c12 = c12,
        c13 = c13,
        c23 = c23)
end;

function ray_angles3D(r)
    θ1 = acos(r[3])
    θ2 = atan(r[2], r[1])
    return θ1, θ2
end;

function dr_dphi3D(θ1, θ2)
    dr_dPθ1 = @SVector [cos(θ1) * cos(θ2),
                        cos(θ1) * sin(θ2),
                        -sin(θ1)]

    dr_dPθ2 = @SVector [-sin(θ1) * sin(θ2),
                        sin(θ1) * cos(θ2),
                        0.0]
    return dr_dPθ1, dr_dPθ2
end;

function Γn3D(c,n)
    c11, c12, c13 = c.c11, c.c12, c.c13
    c22, c23      = c.c22, c.c23
    c33           = c.c33
    c44, c55, c66 = c.c44, c.c55, c.c66
    n1, n2, n3 = n[1], n[2], n[3]

    Γ = @SMatrix[
         c11 * n1^2 + c66 * n2^2 + c55 * n3^2  (c12 + c66) * n1 * n2                   (c13 + c55) * n1 * n3;
        (c12 + c66) * n1 * n2                   c66 * n1^2 + c22 * n2^2 + c44 * n3^2   (c23 + c44) * n2 * n3;
        (c13 + c55) * n1 * n3                  (c23 + c44) * n2 * n3                    c55 * n1^2 + c44 * n2^2 + c33 * n3^2
    ]
    return Γ
end;

function solve_christoffel3D!(VpVs::MVector{3,Float64}, UpUs::MMatrix{3,3,Float64}, c, n)
    Γ = Γn3D(c,n)
    F = eigen(Γ)           
    V = F.values
    U = F.vectors
    # P -> 1, S1 -> 2, S2 -> 3
    VpVs[1] = sqrt(real(V[3]))  
    VpVs[2] = sqrt(real(V[2]))  
    VpVs[3] = sqrt(real(V[1])) 
    UpUs[:,1] .= U[:,3]         
    UpUs[:,2] .= U[:,2]       
    UpUs[:,3] .= U[:,1] 
end;

function scalar_group_velocity3D(phase,c,n)

    phase_index = phase == :P  ? 1 :
                  phase == :S1 ? 2 :
                  phase == :S2 ? 3 :
                  error("Phase $(phase) not in ['P', 'S1', 'S2']") 

    g = MVector{3,Float64}(undef)
    VpVs = MVector{3,Float64}(undef)
    UpUs = MMatrix{3,3,Float64}(undef)

    solve_christoffel3D!(VpVs,UpUs,c,n)
    P = n ./ VpVs[phase_index]
    ΓU  = Γn3D(c, UpUs[:,phase_index])
    @einsum g[i] = ΓU[i,j] * P[j]

    return norm(g)
end;

function grad_g3D(g, r, n, d, dr_dθ1, dr_dθ2)

    M = SMatrix{3,3}(
            d*dr_dθ1[1], d*dr_dθ2[1], r[1],
            d*dr_dθ1[2], d*dr_dθ2[2], r[2],
            d*dr_dθ1[3], d*dr_dθ2[3], r[3])

    dg_dθ1 = -g * (dot(dr_dθ1,n) / dot(r,n))
    dg_dθ2 = -g * (dot(dr_dθ2,n) / dot(r,n))
    S = SVector{3}(dg_dθ1, dg_dθ2, 0.)

    ∇g = inv(M) * S
    return ∇g
end;


function assemble_bending_equations_3D(phase, xk, ck)
    K = length(xk) - 2

    # compute for all layers 
    dk = get_segment_lengths(xk)
    rk = [get_unit_vector(xk[i+1], xk[i]) for i in 1:length(xk)-1]
    nk = rk # assuming homogeneous layers 
    Θk = [ray_angles3D(r) for r in rk]
    ∂r_∂Θk = [dr_dphi3D(θ1, θ2) for (θ1, θ2) in Θk]
    gk = [scalar_group_velocity3D(phase, c, n) for (c,n) in zip(ck, nk)]
    Δgk = [grad_g3D(g,r,n,d,dr_dθ[1], dr_dθ[2]) for (g,r,n,d,dr_dθ) in zip(gk, rk, nk, dk, ∂r_∂Θk)]
    Δtk  = [get_grad_t(xk[k+1], xk[k], dk[k], gk[k], Δgk[k]) for k in 1:K]
    Δtk1 = [get_grad_t(xk[k+2], xk[k+1], dk[k+1], gk[k+1], Δgk[k+1]) for k in 1:K]

    # assemble 
    F = zeros(K*2)
    idx = 1
    for k in 1:K
        F[idx:idx+1] .= [1 0 0; 0 1 0] * (Δtk[k] - Δtk1[k])
        idx +=2 
    end 
    return F 
end

function traveltime_3D(phase, xk, ck)
    dk = get_segment_lengths(xk)
    rk = [get_unit_vector(xk[i+1], xk[i]) for i in 1:length(xk)-1]
    nk = rk 
    gk = [scalar_group_velocity3D(phase, c, n) for (c,n) in zip(ck, nk)]
    traveltime = sum(d/g for (d,g) in zip(dk, gk))
    return traveltime
end;


function ray_residual_3D!(F, p, phase, xk, ck)
    xk_opt = copy(xk)
    # insert parameter vektor into xk: x1(1),x2(1), ...x1(n),x2(n) -> [(x1(1),x2(1)),..., (x1(n), x2(n))]
    for (i, k) in enumerate(2:length(xk)-1)
        xk_opt[k][1:2] .= p[(2i-1):(2i)]
    end
    F .= assemble_bending_equations_3D(phase, xk_opt, ck)
end


function ray_bending_3D(phase, M, interface_depths, src, rcv; verbose=false)

    vp_layers  = @view M[:,1]
    vs_layers  = @view M[:,2]
    eps1_layers = @view M[:,3]
    eps2_layers = @view M[:,4]
    gam1_layers = @view M[:,5]
    gam2_layers = @view M[:,6]
    del1_layers = @view M[:,7]
    del2_layers = @view M[:,8]
    del3_layers = @view M[:,9]

    c_layers = [C3D(vp, vs; eps1=eps1, eps2=eps2, gam1=gam1, gam2=gam2, del1=del1, del2=del2, del3=del3) for 
               (vp,vs,eps1,eps2,gam1,gam2,del1,del2,del3) in zip(
                vp_layers, vs_layers, 
                eps1_layers, eps2_layers, 
                gam1_layers, gam2_layers,
                del1_layers, del2_layers, del3_layers)
    ];

    xk, interface_ids, layer_ids = straight_ray_intersections_3D(src, rcv, interface_depths)
    ck = c_layers[layer_ids]
    xk0 = deepcopy(xk)

    if length(xk) == 2
        ; # src & rcv in the same layer
    else
        p0 = reduce(vcat, [xk[k][1:2] for k in 2:length(xk)-1])
        res = nlsolve((F, p) -> ray_residual_3D!(F, p, phase, xk, ck), p0)
        if verbose println(res) end
    end 

    t  = traveltime_3D(phase, xk, c_layers)
    t0 = traveltime_3D(phase, xk0, c_layers)

    return (xk0=xk0, xk=xk, t0=t0, t=t)
end;

# Wrapper
function ray_bending(phase, M, interface_depths, src, rcv; verbose=false)

    @assert size(M, 2) in [4, 9] error("""
    Invalid input matrix format.
    Expected:
      • 2D: nlayer × 4 matrix  → columns: vp, vs, epsilon, delta
      • 3D: nlayer × 9 matrix  → columns: vp, vs, epsilon1, epsilon2, gamma1, gamma2, delta1, delta2, delta3
    """)

    nlayer = size(M, 1)
    @assert nlayer == length(interface_depths) + 1 error("""
    Mismatch between layers and interface depths.
    Expected:
    • Number of layers (rows in M): length(interface_depths) + 1
    Found:
    • Number of layers: $(nlayer)
    • Number of interface depths: $(length(interface_depths))
    """)

    dim = size(M, 2) == 4 ? 2 : 3
    if dim == 2
        @assert length(src) == length(rcv) == 2 "Length of src and rcv must have length of 2"
        ray = ray_bending_2D(phase, M, interface_depths, src, rcv; verbose=verbose)
    else
        @assert length(src) == length(rcv) == 3 "Length of src and rcv must have length of 3"
        ray = ray_bending_3D(phase, M, interface_depths, src, rcv; verbose=verbose)
    end 
    return ray
end;