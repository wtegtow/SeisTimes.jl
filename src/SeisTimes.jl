module SeisTimes

using StaticArrays, LinearAlgebra, Einsum
using Printf, Test

export fast_sweep, Solid2D, Solid3D

# ========================================
# Helper
# ========================================

function L2!(buf, x, y)
    @. buf = x - y
    return norm(buf)
end

function L∞!(buf, x, y)
    @. buf = abs(x - y)
    return maximum(buf)
end

function ΦWENO(a,b,c,d)
    ϵ_tol = 1e-8

    IS0 = 13*(a - b)^2 + 3*(a - 3b)^2
    IS1 = 13*(b - c)^2 + 3*(b + c)^2
    IS2 = 13*(c - d)^2 + 3*(3c - d)^2

    α0 = 1/(ϵ_tol + IS0)^2
    α1 = 6/(ϵ_tol + IS1)^2
    α2 = 3/(ϵ_tol + IS2)^2

    w0 = α0 / (α0 + α1 + α2)
    w2 = α2 / (α0 + α1 + α2)

    ϕweno = (1/3) * w0 * (a - 2b + c) + (1/6) * (w2 - 0.5) * (b - 2c + d)
    return ϕweno
end

# ========================================
# 2D 
# ========================================

struct Solid2D{V, I}
    x_coords::V
    z_coords::V
    c11::I 
    c13::I 
    c33::I 
    c55::I
   
    function Solid2D(x_coords, z_coords, vp, vs; eps=0.0, del=0.0)
        μ = @. vs^2
        λ = @. vp^2 - 2*μ
        # density normalized stiffness
        c11 = @. ((λ + 2μ) * (2*eps .+ 1))
        c13 = @. (λ + del * (λ + 2μ)) 
        c33 = @. (λ + 2μ) 
        c55 = @. (μ) 

        new{typeof(x_coords), typeof(c11)}(x_coords, z_coords, c11, c13, c33, c55)
    end
end

function Γn(solid::Solid2D, n, i, k)
    # Christoffel matrix Γ(n) = n C n 
    Γ = @SMatrix[
         solid.c11[i,k] * n[1]^2 + solid.c55[i,k] * n[2]^2  (solid.c13[i,k] + solid.c55[i,k]) * n[1] * n[2]    ;
        (solid.c13[i,k] + solid.c55[i,k]) * n[1] * n[2]      solid.c55[i,k] * n[1]^2 + solid.c33[i,k] * n[2]^2
        ]
    return Γ
end

function solve_christoffel!(VpVs::MVector{2,Float64}, UpUs::MMatrix{2,2,Float64}, solid::Solid2D, n::SVector{2,Float64}, i::Int, k::Int)
    Γ = Γn(solid, n, i, k)  
    F = eigen(Γ)           
    V = F.values
    U = F.vectors
    # P -> 1, S -> 2
    VpVs[1] = sqrt(real(V[2]))  
    VpVs[2] = sqrt(real(V[1]))  
    UpUs[:,1] .= U[:,2]         
    UpUs[:,2] .= U[:,1]         
end


function LxFS1(T, i, k, dx, dz)
    return SVector(T[i+1,k], T[i-1,k],  T[i,k+1],  T[i,k-1])
end

function LxFS3(T, i, k, dx, dz)

    ϵ_tol = 1e-8

    denom_x = (ϵ_tol + (T[i+1,k] - 2*T[i,k] + T[i-1,k])^2)
    denom_z = (ϵ_tol + (T[i,k+1] - 2*T[i,k] + T[i,k-1])^2)

    γ_p_x =   (ϵ_tol + (T[i,k] - 2*T[i+1,k] + T[i+2,k])^2) / denom_x
    γ_m_x =   (ϵ_tol + (T[i,k] - 2*T[i-1,k] + T[i-2,k])^2) / denom_x
    γ_p_z =   (ϵ_tol + (T[i,k] - 2*T[i,k+1] + T[i,k+2])^2) / denom_z
    γ_m_z =   (ϵ_tol + (T[i,k] - 2*T[i,k-1] + T[i,k-2])^2) / denom_z

    ω_p_x = 1 / (1 + 2 * γ_p_x^2)
    ω_m_x = 1 / (1 + 2 * γ_m_x^2)
    ω_p_z = 1 / (1 + 2 * γ_p_z^2)
    ω_m_z = 1 / (1 + 2 * γ_m_z^2)

    tp_x_ = (1 - ω_p_x) / (2 * dx) * (T[i+1,k] - T[i-1,k]) +  ω_p_x  / (2 * dx) * (-3*T[i,k] + 4*T[i+1,k] - T[i+2,k])
    tm_x_ = (1 - ω_m_x) / (2 * dx) * (T[i+1,k] - T[i-1,k]) +  ω_m_x  / (2 * dx) * ( 3*T[i,k] - 4*T[i-1,k] + T[i-2,k])
    tp_z_ = (1 - ω_p_z) / (2 * dz) * (T[i,k+1] - T[i,k-1]) +  ω_p_z  / (2 * dz) * (-3*T[i,k] + 4*T[i,k+1] - T[i,k+2])
    tm_z_ = (1 - ω_m_z) / (2 * dz) * (T[i,k+1] - T[i,k-1]) +  ω_m_z  / (2 * dz) * ( 3*T[i,k] - 4*T[i,k-1] + T[i,k-2])

    return SVector(T[i,k] + dx * tp_x_, 
                   T[i,k] - dx * tm_x_,  
                   T[i,k] + dz * tp_z_,  
                   T[i,k] - dz * tm_z_)
end


function LxFS5(T, i, k, dx, dz)
    #   Operator
    #   Δ⁺ₓ φᵢⱼ = φᵢ₊₁ⱼ − φᵢⱼ       # forward difference in x
    #   Δ⁻ₓ φᵢⱼ = φᵢⱼ   − φᵢ₋₁ⱼ     # backward difference in x
    #   Δ⁺𝓏 φᵢⱼ = φᵢⱼ₊₁ − φᵢⱼ       # forward difference in z
    #   Δ⁻𝓏 φᵢⱼ = φᵢⱼ   − φᵢⱼ₋₁     # backward difference in z

    Δpx_m2 = (T[i-1, k] - T[i-2, k]) / dx   # Δ⁺φ_{i−2,j}
    Δpx_m1 = (T[i,   k] - T[i-1, k]) / dx   # Δ⁺φ_{i−1,j}
    Δpx_0  = (T[i+1, k] - T[i,   k]) / dx   # Δ⁺φ_{i,j}
    Δpx_p1 = (T[i+2, k] - T[i+1, k]) / dx   # Δ⁺φ_{i+1,j}

    Δpz_m2 = (T[i, k-1] - T[i, k-2]) / dz   # Δ⁺φ_{i,j−2}
    Δpz_m1 = (T[i,   k] - T[i, k-1]) / dz   # Δ⁺φ_{i,j−1}
    Δpz_0  = (T[i, k+1] - T[i,   k]) / dz   # Δ⁺φ_{i,j}
    Δpz_p1 = (T[i, k+2] - T[i, k+1]) / dz   # Δ⁺φ_{i,j+1}

    # WENO parameter a,b,c,d
    axp = (T[i+3, k] - 2*T[i+2,k] + T[i+1, k]) / dx
    axm = (T[i-3, k] - 2*T[i-2,k] + T[i-1, k]) / dx
    azp = (T[i, k+3] - 2*T[i,k+2] + T[i, k+1]) / dz
    azm = (T[i, k-3] - 2*T[i,k-2] + T[i, k-1]) / dz

    bxp = (T[i+2, k] - 2*T[i+1,k] + T[i, k]) / dx
    bxm = (T[i-2, k] - 2*T[i-1,k] + T[i, k]) / dx
    bzp = (T[i, k+2] - 2*T[i,k+1] + T[i, k]) / dz
    bzm = (T[i, k-2] - 2*T[i,k-1] + T[i, k]) / dz

    cxp = (T[i+1, k] - 2*T[i,k] + T[i-1, k]) / dx
    cxm = (T[i+1, k] - 2*T[i,k] + T[i-1, k]) / dx
    czp = (T[i, k+1] - 2*T[i,k] + T[i, k-1]) / dz
    czm = (T[i, k+1] - 2*T[i,k] + T[i, k-1]) / dz

    dxp = (T[i-2, k] - 2*T[i-1,k] + T[i, k]) / dx
    dxm = (T[i+2, k] - 2*T[i+1,k] + T[i, k]) / dx
    dzp = (T[i, k-2] - 2*T[i,k-1] + T[i, k]) / dz
    dzm = (T[i, k+2] - 2*T[i,k+1] + T[i, k]) / dz

    ϕweno_xp = ΦWENO(axp,bxp,cxp,dxp)
    ϕweno_xm = ΦWENO(axm,bxm,cxm,dxm)
    ϕweno_zp = ΦWENO(azp,bzp,czp,dzp)
    ϕweno_zm = ΦWENO(azm,bzm,czm,dzm)

    tp_x_ = 1/12 * (-Δpx_m2 + 7*Δpx_m1 + 7*Δpx_0 - Δpx_p1) + ϕweno_xp
    tm_x_ = 1/12 * (-Δpx_m2 + 7*Δpx_m1 + 7*Δpx_0 - Δpx_p1) - ϕweno_xm

    tp_z_ = 1/12 * (-Δpz_m2 + 7*Δpz_m1 + 7*Δpz_0 - Δpz_p1) + ϕweno_zp
    tm_z_ = 1/12 * (-Δpz_m2 + 7*Δpz_m1 + 7*Δpz_0 - Δpz_p1) - ϕweno_zm

    return SVector(T[i,k] + dx * tp_x_, 
                   T[i,k] - dx * tm_x_,  
                   T[i,k] + dz * tp_z_,  
                   T[i,k] - dz * tm_z_)
end

function compute_viscosities(solid::Solid2D; deg_increment=3, buffer_factor=2)

    C = ([ (solid.c11[i,k], 
            solid.c13[i,k], 
            solid.c33[i,k], 
            solid.c55[i,k]) 
            for i in axes(solid.c11,1), k in axes(solid.c11,2)
        ])
                        
    unique_C = unique(C)
    indices_C = [findfirst(==(val), C) for val in unique_C]
    n_unique_C = length(unique_C)

    angles_theta = deg2rad.(0:deg_increment:360)
    n_theta = length(angles_theta)

    visc_p = zeros(n_unique_C, n_theta, 2);
    visc_s = zeros(n_unique_C, n_theta, 2);

    VpVs = MVector{2,Float64}(undef)
    UpUs = MMatrix{2,2,Float64}(undef)

    for c in 1:n_unique_C
        c_idx = indices_C[c]
        
        for (theta_idx, theta) in enumerate(angles_theta)

            n = SVector(cos(theta), sin(theta))
            n = n/norm(n)

            solve_christoffel!(VpVs, UpUs, solid, n, c_idx[1], c_idx[2])

            Pp = n ./ VpVs[1]
            Ps = n ./ VpVs[2]

            ΓUp = Γn(solid, UpUs[:,1], c_idx[1], c_idx[2])
            ΓUs = Γn(solid, UpUs[:,2], c_idx[1], c_idx[2])

            @einsum gp[i] := ΓUp[i,j] * Pp[j]
            @einsum gs[i] := ΓUs[i,j] * Ps[j]
            @test dot(gp,Pp) ≈ 1 rtol=1e-3
            @test dot(gs,Ps) ≈ 1 rtol=1e-3

            ∂H∂P = -gp ./ VpVs[1]
            ∂H∂S = -gs ./ VpVs[2]

            visc_p[c,theta_idx,:] .= abs.(∂H∂P) .* buffer_factor
            visc_s[c,theta_idx,:] .= abs.(∂H∂S) .* buffer_factor

        end
    end

    visc_p = Float64[maximum(visc_p[:,:,1]), maximum(visc_p[:,:,2])]
    visc_s = Float64[maximum(visc_s[:,:,1]), maximum(visc_s[:,:,2])]
    return visc_p, visc_s;
end;

function inject_sources!(solid::Solid2D, T, source_mask, sources_phys, scheme, wavemode) 

    source_center_ids = Tuple{Int, Int}[(argmin(abs.(solid.x_coords .- x)), 
                                         argmin(abs.(solid.z_coords .- z))) 
                                         for (x, z) in sources_phys]
                    
    cell_size = scheme == :LxFS1 ? 0 :
                scheme == :LxFS3 ? 1 :
                scheme == :LxFS5 ? 2 :
                error("Scheme $(scheme) not in ['LxFS1', 'LxFS3', 'LxFS5']")

    VpVs1 = MVector{2,Float64}(undef)
    UpUs1 = MMatrix{2,2,Float64}(undef)
    VpVs2 = MVector{2,Float64}(undef)
    UpUs2 = MMatrix{2,2,Float64}(undef)
    
    nx, nz = length(solid.x_coords), length(solid.z_coords)
    for (is,(ix, iz)) in enumerate(source_center_ids)
        for i in clamp(ix-cell_size, 1, nx):clamp(ix+cell_size, 1, nx)
            for k in clamp(iz-cell_size, 1, nz):clamp(iz+cell_size, 1, nz)

                source_mask[i, k] = true 

                # physical location at grid 
                xg = solid.x_coords[i]
                zg = solid.z_coords[k]

                # physical location of source 
                sx = sources_phys[is][1]
                sz = sources_phys[is][2]

                # distance
                r = sqrt((sx - xg)^2 + (sz - zg)^2)

                # direction vector n 
                dxi = xg - sx
                dzi = zg - sz
                n = SVector(dxi, dzi) / (1e-10 + sqrt(dxi^2 +dzi^2))

                solve_christoffel!(VpVs1, UpUs1, solid, n, i, k)
                solve_christoffel!(VpVs2, UpUs2, solid, n, ix, iz)

                # This might be not very accurate for strong heterogenities around the source
                if wavemode == :P 
                    T[i,k] = r / ((1e-10 + VpVs1[1] + VpVs2[1])/2)
                elseif wavemode == :S 
                    T[i,k] = r / ((1e-10 + VpVs1[2] + VpVs2[2])/2)
                end
            end
        end 
    end
end



function calc_time!(solid::Solid2D, T, VpVs, UpUs, lxfs, viscosities, velocity_index, i, k, dx, dz)

    tp_x, tm_x, tp_z, tm_z = lxfs(T, i, k, dx, dz)
    
    p = SVector((tp_x - tm_x) / (2dx), (tp_z - tm_z) / (2dz))
    p_norm = norm(p)
    if p_norm == 0 return end 
    p /= p_norm

    solve_christoffel!(VpVs, UpUs, solid, p, i, k)

    H = 1/VpVs[velocity_index] - p_norm 
    A = 1/(viscosities[1]/dx + viscosities[2]/dz)
    C = viscosities[1] * ((tp_x + tm_x)/(2*dx)) + viscosities[2] * ((tp_z + tm_z)/(2*dz))
    T_new = A * (H + C)
    T[i, k] = min(T_new, T[i, k])

end

function calc_bcs!(T, nx, nz, N)
    # outflow bc's
    for i in 1:nx, n in 1:N
        T[i,n] = min(max(2*T[i,n+1] - T[i,n+2], T[i,n+2]), T[i,n])
        k = nz - n + 1
        T[i,k] = min(max(2*T[i,k-1] - T[i,k-2], T[i,k-2]), T[i,k])
    end
    for k in 1:nz, n in 1:N
        T[n,k] = min(max(2*T[n+1,k] - T[n+2,k], T[n+2,k]), T[n,k])
        i = nx - n + 1
        T[i,k] = min(max(2*T[i-1,k] - T[i-2,k], T[i-2,k]), T[i,k])
    end
end

function fast_sweep(solid::Solid2D, 
                    sources_phys, 
                    wavemode, 
                    scheme; 
                    verbose=false,
                    max_iter=200,
                    max_error_tol = 1e-5,
                    viscosity_buffer = 1.5)

    # params 
    INF = 1e10 # initial value

    # allocations
    x_coords = solid.x_coords
    dx = x_coords[2] - x_coords[1]
    nx = length(x_coords)

    z_coords = solid.z_coords
    dz = z_coords[2] - z_coords[1]
    nz = length(z_coords)

    T     = fill(INF, nx, nz)
    T_old = fill(INF, nx, nz)
    E = similar(T)

    VpVs = MVector{2,Float64}(undef)
    UpUs = MMatrix{2,2,Float64}(undef)

    # calculate artificial viscosities
    visc_p, visc_s = compute_viscosities(solid; buffer_factor=viscosity_buffer);

    # select wavemode
    if wavemode == :P 
        velocity_index = 1
        viscosities = visc_p
    elseif wavemode == :S
        velocity_index = 2
        viscosities = visc_s
    else 
        error("wavemode $(wavemode) not in ['P', 'S']")
    end 

    # select scheme 
    if scheme == :LxFS1 
        lxfs = LxFS1 
        N = 1
    elseif scheme == :LxFS3 
        lxfs = LxFS3
        N = 2
    elseif scheme == :LxFS5
        lxfs = LxFS5
        N = 3
    else 
        error("Scheme $(scheme) not in ['LxFS1', 'LxFS3', 'LxFS5']")
    end

    # inject sources
    source_mask = falses(nx, nz)
    inject_sources!(solid, T, source_mask, sources_phys, scheme, wavemode) 

    # fast sweep main loop 
    inner_sweeps = (
        (1+N:1:nx-N , 1+N:1:nz-N), 
        (nx-N:-1:1+N, 1+N:1:nz-N), 
        (1+N:1:nx-N, nz-N:-1:1+N),
        (nx-N:-1:1+N, nz-N:-1:1+N)
    )

    converged = false
    diverged  = false
    L2_error = INF
    L∞_error = INF

    if verbose 
        println("===============================================")
        println("Compute Traveltimes: $(wavemode)")
        @printf("Grid Size: %d, %d\n", nx, nz)
        @printf("Viscosities: %.2e | %.2e \n", viscosities[1], viscosities[2])
        println("===============================================")
    end 

    for iter in 1:max_iter

        if verbose
        @printf("Iter: %5d | L2: %.5e | L∞: %.5e\n", iter, L2_error, L∞_error)
        end

        @inbounds begin 
        for (x_order, z_order) in inner_sweeps
            for i in x_order, k in z_order
                if !source_mask[i, k]
                    calc_time!(solid, T, VpVs, UpUs, lxfs, viscosities, velocity_index, i, k, dx, dz)
                end
            end
            calc_bcs!(T, nx, nz, N)
        end
        end

        L2_error = L2!(E, T, T_old)
        L∞_error = L∞!(E, T, T_old)
        # check convergence
        if L∞_error  < max_error_tol
            if verbose println("Solution converged after $(iter) iterations.") end 
            converged = true
            break
        end

        # check divergence 
        if any(T .< 0) || any(T .> INF*2) 
            println("Solution diverged. Try larger viscosity_buffer than $(viscosity_buffer).")
            diverged = true
            break
        end

        T_old .= T

    end

    if !converged && !diverged
        println("Solution not converged. Try larger max_iter than $(max_iter).")
    end

    return T

end

# ========================================
# 3D
# ========================================

struct Solid3D{V, I}
    x_coords::V
    y_coords::V
    z_coords::V
    c33::I
    c55::I
    c11::I
    c22::I
    c66::I
    c44::I
    c13::I
    c23::I
    c12::I
   
    function Solid3D(x_coords, y_coords, z_coords, vp, vs; 
                    eps1=0., eps2=0., gam1=0., gam2=0., 
                    del1=0., del2=0., del3=0.)


        # density normalized stiffness
        c33 = @. vp^2
        c55 = @. vs^2
        c11 = @. (2*eps2 + 1) * c33
        c22 = @. (2*eps1 + 1) * c33
        c66 = @. (2*gam1 + 1) * c55
        c44 = @. c66 / (1 + gam2)
        c13 = @. sqrt(2 * c33 * (c33 - c55) * del2 + (c33 - c55)^2) - c55
        c23 = @. sqrt(2 * c33 * (c33 - c44) * del1 + (c33 - c44)^2) - c44
        c12 = @. sqrt(2 * c11 * (c11 - c66) * del3 + (c11 - c66)^2) - c66

        new{typeof(x_coords), typeof(c11)}(
            x_coords, y_coords, z_coords,
            c33, c55, c11, c22, c66, c44, c13, c23, c12)
    end
end


function Γn(solid::Solid3D, n, i::Int, j::Int, k::Int)

    # Christoffel matrix Γ(n) = n C n 
    c11, c12, c13 = solid.c11[i,j,k], solid.c12[i,j,k], solid.c13[i,j,k]
    c22, c23      = solid.c22[i,j,k], solid.c23[i,j,k]
    c33           = solid.c33[i,j,k]
    c44, c55, c66 = solid.c44[i,j,k], solid.c55[i,j,k], solid.c66[i,j,k]
    n1, n2, n3 = n[1], n[2], n[3]

    Γ = @SMatrix[
         c11 * n1^2 + c66 * n2^2 + c55 * n3^2  (c12 + c66) * n1 * n2                   (c13 + c55) * n1 * n3;
        (c12 + c66) * n1 * n2                   c66 * n1^2 + c22 * n2^2 + c44 * n3^2   (c23 + c44) * n2 * n3;
        (c13 + c55) * n1 * n3                  (c23 + c44) * n2 * n3                    c55 * n1^2 + c44 * n2^2 + c33 * n3^2
    ]
    return Γ
end

function solve_christoffel!(VpVs::MVector{3,Float64}, UpUs::MMatrix{3,3,Float64}, solid::Solid3D, n, i::Int, j::Int, k::Int) 

    Γ = Γn(solid, n, i, j, k)  
    F = eigen(Γ)           
    V = F.values
    U = F.vectors
    # qP -> 1, S1 -> 2, S2 -> 3
    VpVs[1] = sqrt(real(V[3]))  
    VpVs[2] = sqrt(real(V[2]))  
    VpVs[3] = sqrt(real(V[1])) 
    UpUs[:,1] .= U[:,3]         
    UpUs[:,2] .= U[:,2]       
    UpUs[:,3] .= U[:,1]   
end


function LxFS1(T, i, j, k, dx, dy, dz)
    return SVector(T[i+1,j,k], T[i-1,j,k],  
                   T[i,j+1,k], T[i,j-1,k],
                   T[i,j,k+1], T[i,j,k-1])
end

function LxFS3(T, i, j, k, dx, dy, dz)

    ϵ_tol = 1e-8

    denom_x = (ϵ_tol + (T[i+1,j,k] - 2*T[i,j,k] + T[i-1,j,k])^2)
    denom_y = (ϵ_tol + (T[i,j+1,k] - 2*T[i,j,k] + T[i,j-1,k])^2)
    denom_z = (ϵ_tol + (T[i,j,k+1] - 2*T[i,j,k] + T[i,j,k-1])^2)

    γ_p_x =   (ϵ_tol + (T[i,j,k] - 2*T[i+1,j,k] + T[i+2,j,k])^2) / denom_x
    γ_m_x =   (ϵ_tol + (T[i,j,k] - 2*T[i-1,j,k] + T[i-2,j,k])^2) / denom_x
    γ_p_y =   (ϵ_tol + (T[i,j,k] - 2*T[i,j+1,k] + T[i,j+2,k])^2) / denom_y
    γ_m_y =   (ϵ_tol + (T[i,j,k] - 2*T[i,j-1,k] + T[i,j-2,k])^2) / denom_y
    γ_p_z =   (ϵ_tol + (T[i,j,k] - 2*T[i,j,k+1] + T[i,j,k+2])^2) / denom_z
    γ_m_z =   (ϵ_tol + (T[i,j,k] - 2*T[i,j,k-1] + T[i,j,k-2])^2) / denom_z

    ω_p_x = 1 / (1 + 2 * γ_p_x^2)
    ω_m_x = 1 / (1 + 2 * γ_m_x^2)
    ω_p_y = 1 / (1 + 2 * γ_p_y^2)
    ω_m_y = 1 / (1 + 2 * γ_m_y^2)
    ω_p_z = 1 / (1 + 2 * γ_p_z^2)
    ω_m_z = 1 / (1 + 2 * γ_m_z^2)

    tp_x_ = (1 - ω_p_x) / (2 * dx) * (T[i+1,j,k] - T[i-1,j,k]) +  ω_p_x  / (2 * dx) * (-3*T[i,j,k] + 4*T[i+1,j,k] - T[i+2,j,k])
    tm_x_ = (1 - ω_m_x) / (2 * dx) * (T[i+1,j,k] - T[i-1,j,k]) +  ω_m_x  / (2 * dx) * ( 3*T[i,j,k] - 4*T[i-1,j,k] + T[i-2,j,k])

    tp_y_ = (1 - ω_p_y) / (2 * dy) * (T[i,j+1,k] - T[i,j-1,k]) +  ω_p_y  / (2 * dy) * (-3*T[i,j,k] + 4*T[i,j+1,k] - T[i,j+2,k])
    tm_y_ = (1 - ω_m_y) / (2 * dy) * (T[i,j+1,k] - T[i,j-1,k]) +  ω_m_y  / (2 * dy) * ( 3*T[i,j,k] - 4*T[i,j-1,k] + T[i,j-2,k])

    tp_z_ = (1 - ω_p_z) / (2 * dz) * (T[i,j,k+1] - T[i,j,k-1]) +  ω_p_z  / (2 * dz) * (-3*T[i,j,k] + 4*T[i,j,k+1] - T[i,j,k+2])
    tm_z_ = (1 - ω_m_z) / (2 * dz) * (T[i,j,k+1] - T[i,j,k-1]) +  ω_m_z  / (2 * dz) * ( 3*T[i,j,k] - 4*T[i,j,k-1] + T[i,j,k-2])

    return SVector(T[i,j,k] + dx * tp_x_, 
                   T[i,j,k] - dx * tm_x_,  
                   T[i,j,k] + dy * tp_y_, 
                   T[i,j,k] - dy * tm_y_,
                   T[i,j,k] + dz * tp_z_,  
                   T[i,j,k] - dz * tm_z_)
end


function LxFS5(T, i, j, k, dx, dy, dz)
    #   Operator
    #   Δ⁺ₓ φᵢⱼ = φᵢ₊₁ⱼ − φᵢⱼ       # forward difference in x
    #   Δ⁻ₓ φᵢⱼ = φᵢⱼ   − φᵢ₋₁ⱼ     # backward difference in x

    Δpx_m2 = (T[i-1, j, k] - T[i-2, j, k]) / dx   
    Δpx_m1 = (T[i,   j, k] - T[i-1, j, k]) / dx   
    Δpx_0  = (T[i+1, j, k] - T[i,   j, k]) / dx  
    Δpx_p1 = (T[i+2, j, k] - T[i+1, j, k]) / dx   

    Δpy_m2 = (T[i, j-1, k] - T[i, j-2, k]) / dy
    Δpy_m1 = (T[i, j,   k] - T[i, j-1, k]) / dy   
    Δpy_0  = (T[i, j+1, k] - T[i, j,   k]) / dy
    Δpy_p1 = (T[i, j+2, k] - T[i, j+1, k]) / dy   

    Δpz_m2 = (T[i, j, k-1] - T[i, j, k-2]) / dz   
    Δpz_m1 = (T[i, j,   k] - T[i, j, k-1]) / dz 
    Δpz_0  = (T[i, j, k+1] - T[i, j,   k]) / dz   
    Δpz_p1 = (T[i, j, k+2] - T[i, j, k+1]) / dz  

    # WENO parameter a,b,c,d
    axp = (T[i+3,j,k] - 2*T[i+2,j,k] + T[i+1,j,k]) / dx
    axm = (T[i-3,j,k] - 2*T[i-2,j,k] + T[i-1,j,k]) / dx

    ayp = (T[i,j+3,k] - 2*T[i,j+2,k] + T[i,j+1,k]) / dy
    aym = (T[i,j-3,k] - 2*T[i,j-2,k] + T[i,j-1,k]) / dy

    azp = (T[i,j,k+3] - 2*T[i,j,k+2] + T[i,j,k+1]) / dz
    azm = (T[i,j,k-3] - 2*T[i,j,k-2] + T[i,j,k-1]) / dz

    bxp = (T[i+2,j,k] - 2*T[i+1,j,k] + T[i,j,k]) / dx
    bxm = (T[i-2,j,k] - 2*T[i-1,j,k] + T[i,j,k]) / dx

    byp = (T[i,j+2,k] - 2*T[i,j+1,k] + T[i,j,k]) / dy
    bym = (T[i,j-2,k] - 2*T[i,j-1,k] + T[i,j,k]) / dy

    bzp = (T[i,j,k+2] - 2*T[i,j,k+1] + T[i,j,k]) / dz
    bzm = (T[i,j,k-2] - 2*T[i,j,k-1] + T[i,j,k]) / dz

    cxp = (T[i+1,j,k] - 2*T[i,j,k] + T[i-1,j,k]) / dx
    cxm = (T[i+1,j,k] - 2*T[i,j,k] + T[i-1,j,k]) / dx

    cyp = (T[i,j+1,k] - 2*T[i,j,k] + T[i,j-1,k]) / dy
    cym = (T[i,j+1,k] - 2*T[i,j,k] + T[i,j-1,k]) / dy

    czp = (T[i,j,k+1] - 2*T[i,j,k] + T[i,j,k-1]) / dz
    czm = (T[i,j,k+1] - 2*T[i,j,k] + T[i,j,k-1]) / dz

    dxp = (T[i-2,j,k] - 2*T[i-1,j,k] + T[i,j,k]) / dx
    dxm = (T[i+2,j,k] - 2*T[i+1,j,k] + T[i,j,k]) / dx

    dyp = (T[i,j-2,k] - 2*T[i,j-1,k] + T[i,j,k]) / dy
    dym = (T[i,j+2,k] - 2*T[i,j+1,k] + T[i,j,k]) / dy

    dzp = (T[i,j,k-2] - 2*T[i,j,k-1] + T[i,j,k]) / dz
    dzm = (T[i,j,k+2] - 2*T[i,j,k+1] + T[i,j,k]) / dz

    ϕweno_xp = ΦWENO(axp,bxp,cxp,dxp)
    ϕweno_xm = ΦWENO(axm,bxm,cxm,dxm)

    ϕweno_yp = ΦWENO(ayp,byp,cyp,dyp)
    ϕweno_ym = ΦWENO(aym,bym,cym,dym)

    ϕweno_zp = ΦWENO(azp,bzp,czp,dzp)
    ϕweno_zm = ΦWENO(azm,bzm,czm,dzm)

    tp_x_ = 1/12 * (-Δpx_m2 + 7*Δpx_m1 + 7*Δpx_0 - Δpx_p1) + ϕweno_xp
    tm_x_ = 1/12 * (-Δpx_m2 + 7*Δpx_m1 + 7*Δpx_0 - Δpx_p1) - ϕweno_xm

    tp_y_ = 1/12 * (-Δpy_m2 + 7*Δpy_m1 + 7*Δpy_0 - Δpy_p1) + ϕweno_yp
    tm_y_ = 1/12 * (-Δpy_m2 + 7*Δpy_m1 + 7*Δpy_0 - Δpy_p1) - ϕweno_ym

    tp_z_ = 1/12 * (-Δpz_m2 + 7*Δpz_m1 + 7*Δpz_0 - Δpz_p1) + ϕweno_zp
    tm_z_ = 1/12 * (-Δpz_m2 + 7*Δpz_m1 + 7*Δpz_0 - Δpz_p1) - ϕweno_zm

    return SVector(T[i,j,k] + dx * tp_x_, 
                   T[i,j,k] - dx * tm_x_,  
                   T[i,j,k] + dy * tp_y_, 
                   T[i,j,k] - dy * tm_y_,
                   T[i,j,k] + dz * tp_z_,  
                   T[i,j,k] - dz * tm_z_)
end

function compute_viscosities(solid::Solid3D; deg_increment=3, buffer_factor=2)

    # assembling C for every grid point might be too expensive?
    C = ([ (solid.c11[i,j,k], solid.c12[i,j,k], solid.c13[i,j,k],
                              solid.c22[i,j,k], solid.c23[i,j,k],
            solid.c44[i,j,k], solid.c55[i,j,k], solid.c66[i,j,k]) 
            for i in axes(solid.c11,1), j in axes(solid.c11,2), k in axes(solid.c11,3)
        ])
                        
    unique_C = unique(C)
    indices_C = [findfirst(==(val), C) for val in unique_C]
    n_unique_C = length(unique_C)
    
    angles_theta = deg2rad.(0:deg_increment:360)
    n_theta = length(angles_theta)

    angles_phi = deg2rad.(0:deg_increment:90)
    n_phi =length(angles_phi)

    visc_p  = zeros(n_unique_C, n_theta, n_phi, 3);
    visc_s1 = zeros(n_unique_C, n_theta, n_phi, 3);
    visc_s2 = zeros(n_unique_C, n_theta, n_phi, 3);

    VpVs = MVector{3,Float64}(undef)
    UpUs = MMatrix{3,3,Float64}(undef)

    for c in 1:n_unique_C
        c_idx = indices_C[c]
        
        for (theta_idx, theta) in enumerate(angles_theta)
            for (phi_idx, phi) in enumerate(angles_phi)

                n = SVector(sin(phi)*cos(theta), sin(phi)*sin(theta), cos(phi))
                n = n/norm(n)

                solve_christoffel!(VpVs, UpUs, solid, n, c_idx[1], c_idx[2], c_idx[3])

                Pp  = n ./ VpVs[1]
                Ps1 = n ./ VpVs[2]
                Ps2 = n ./ VpVs[3]

                ΓUp  = Γn(solid, UpUs[:,1], c_idx[1], c_idx[2], c_idx[3])
                ΓUs1 = Γn(solid, UpUs[:,2], c_idx[1], c_idx[2], c_idx[3])
                ΓUs2 = Γn(solid, UpUs[:,3], c_idx[1], c_idx[2], c_idx[3])

                @einsum  gp[i] := ΓUp[i,j] * Pp[j]
                @einsum gs1[i] := ΓUs1[i,j] * Ps1[j]
                @einsum gs2[i] := ΓUs2[i,j] * Ps2[j]

                @test dot(gp,Pp) ≈ 1 rtol=1e-3
                @test dot(gs1,Ps1) ≈ 1 rtol=1e-3
                @test dot(gs2,Ps2) ≈ 1 rtol=1e-3

                ∂H∂P  = -gp  ./ VpVs[1]
                ∂H∂S1 = -gs1 ./ VpVs[2]
                ∂H∂S2 = -gs2 ./ VpVs[3]

                visc_p[c,theta_idx,phi_idx,:]  .= abs.(∂H∂P)  .* buffer_factor
                visc_s1[c,theta_idx,phi_idx,:] .= abs.(∂H∂S1) .* buffer_factor
                visc_s2[c,theta_idx,phi_idx,:] .= abs.(∂H∂S2) .* buffer_factor
            end
        end
    end

    visc_p  = Float64[maximum(visc_p[:,:,:,1]),  maximum(visc_p[:,:,:,2]),  maximum(visc_p[:,:,:,3])]
    visc_s1 = Float64[maximum(visc_s1[:,:,:,1]), maximum(visc_s1[:,:,:,2]), maximum(visc_s1[:,:,:,3])]
    visc_s2 = Float64[maximum(visc_s2[:,:,:,1]), maximum(visc_s2[:,:,:,2]), maximum(visc_s2[:,:,:,3])]
    return visc_p, visc_s1, visc_s2
end


function inject_sources!(solid::Solid3D, T, source_mask, sources_phys, scheme, wavemode) 

    source_center_ids = Tuple{Int, Int, Int}[(argmin(abs.(solid.x_coords .- x)), 
                                            argmin(abs.(solid.y_coords .- y)),
                                            argmin(abs.(solid.z_coords .- z))
                                            ) 
                                            for (x, y, z) in sources_phys]
                    
    cell_size = scheme == :LxFS1 ? 0 :
                scheme == :LxFS3 ? 1 :
                scheme == :LxFS5 ? 2 :
                error("Scheme $(scheme) not in ['LxFS1', 'LxFS3', 'LxFS5']")

    VpVs1 = MVector{3,Float64}(undef)
    UpUs1 = MMatrix{3,3,Float64}(undef)
    VpVs2 = MVector{3,Float64}(undef)
    UpUs2 = MMatrix{3,3,Float64}(undef)
    
    nx, ny, nz = length(solid.x_coords), length(solid.y_coords), length(solid.z_coords)
    for (is,(ix, iy, iz)) in enumerate(source_center_ids)
        for i in clamp(ix-cell_size, 1, nx):clamp(ix+cell_size, 1, nx)
            for j in clamp(iy-cell_size, 1, ny):clamp(iy+cell_size, 1, ny)
                for k in clamp(iz-cell_size, 1, nz):clamp(iz+cell_size, 1, nz)

                    source_mask[i, j, k] = true 

                    # physical location at grid 
                    xg = solid.x_coords[i]
                    yg = solid.y_coords[j]
                    zg = solid.z_coords[k]

                    # physical location of source 
                    sx = sources_phys[is][1]
                    sy = sources_phys[is][2]
                    sz = sources_phys[is][3]

                    # distance
                    r = sqrt((sx - xg)^2 + (sy - yg)^2 + (sz - zg)^2)

                    # direction vector n 
                    dxi = xg - sx
                    dyi = yg - sy
                    dzi = zg - sz
                    n = SVector(dxi, dyi, dzi) / (1e-10 + sqrt(dxi^2 + dyi^2 + dzi^2))

                    solve_christoffel!(VpVs1, UpUs1, solid, n, i, j, k)
                    solve_christoffel!(VpVs2, UpUs2, solid, n, ix, iy, iz)

                    # This might be not very accurate for strong heterogenities around the source
                    if wavemode == :P 
                        T[i,j,k] = r / ((1e-10 + VpVs1[1] + VpVs2[1])/2)
                    elseif wavemode == :S1 
                        T[i,j,k] = r / ((1e-10 + VpVs1[2] + VpVs2[2])/2)
                    elseif wavemode == :S2
                        T[i,j,k] = r / ((1e-10 + VpVs1[3] + VpVs2[3])/2)
                    end
                end
            end
        end 
    end
end

function calc_time!(solid::Solid3D, T, VpVs, UpUs, lxfs, viscosities, velocity_index, i, j, k, dx, dy, dz)

    tp_x, tm_x, tp_y, tm_y, tp_z, tm_z = lxfs(T, i, j, k, dx, dy, dz)
    
    p = SVector((tp_x - tm_x) / (2dx), 
                (tp_y - tm_y) / (2dy),
                (tp_z - tm_z) / (2dz))
    p_norm = sqrt(p[1]^2 + p[2]^2 + p[3]^2)
    if p_norm == 0 return end 
    p = p / p_norm
    solve_christoffel!(VpVs, UpUs, solid, p, i, j, k)

    H = 1/VpVs[velocity_index] - p_norm 
    A = 1/(viscosities[1]/dx + viscosities[2]/dy + viscosities[3]/dz)
    C = viscosities[1] * ((tp_x + tm_x)/(2*dx)) + 
        viscosities[2] * ((tp_y + tm_y)/(2*dy)) + 
        viscosities[3] * ((tp_z + tm_z)/(2*dz))
    T_new = A * (H + C)
    T[i,j,k] = min(T_new, T[i,j,k])

end

function calc_bcs!(T, nx, ny, nz, N)
    # outflow bc's
    for j in 1:ny, i in 1:nx, n in 1:N
        T[i,j,n] = min(max(2*T[i,j,n+1] - T[i,j,n+2], T[i,j,n+2]), T[i,j,n])
        k = nz - n + 1
        T[i,j,k] = min(max(2*T[i,j,k-1] - T[i,j,k-2], T[i,j,k-2]), T[i,j,k])
    end
    for k in 1:nz, j in 1:ny, n in 1:N
        T[n,j,k] = min(max(2*T[n+1,j,k] - T[n+2,j,k], T[n+2,j,k]), T[n,j,k])
        i = nx - n + 1
        T[i,j,k] = min(max(2*T[i-1,j,k] - T[i-2,j,k], T[i-2,j,k]), T[i,j,k])
    end
    for k in 1:nz, i in 1:nx, n in 1:N
        T[i,n,k] = min(max(2*T[i,n+1,k] - T[i,n+2,k], T[i,n+2,k]), T[i,n,k])
        j = ny - n + 1
        T[i,j,k] = min(max(2*T[i,j-1,k] - T[i,j-2,k], T[i,j-2,k]), T[i,j,k])
    end
end


function fast_sweep(solid::Solid3D, 
                    sources_phys, 
                    wavemode, 
                    scheme; 
                    verbose=false,
                    max_iter=200,
                    max_error_tol = 1e-5,
                    viscosity_buffer = 1.5)

    # params 
    INF = 1e10 # initial value

    # allocations
    x_coords = solid.x_coords
    dx = x_coords[2] - x_coords[1]
    nx = length(x_coords)

    y_coords = solid.y_coords
    dy = y_coords[2] - y_coords[1]
    ny = length(y_coords)

    z_coords = solid.z_coords
    dz = z_coords[2] - z_coords[1]
    nz = length(z_coords)

    T     = fill(INF, nx, ny, nz)
    T_old = fill(INF, nx, ny, nz)
    E =     fill(INF, nx, ny, nz)

    VpVs = MVector{3,Float64}(undef)
    UpUs = MMatrix{3,3,Float64}(undef)

    # calculate artificial viscosities
    visc_p, visc_s1, visc_s2 = compute_viscosities(solid; buffer_factor=viscosity_buffer);

    # select wavemode
    if wavemode == :P 
        velocity_index = 1
        viscosities = visc_p
    elseif wavemode == :S1
        velocity_index = 2
        viscosities = visc_s1 
    elseif wavemode == :S2
        velocity_index = 3
        viscosities = visc_s2
    else 
        error("wavemode $(wavemode) not in ['P', 'S1', 'S2']")
    end 

    # select scheme 
    if scheme == :LxFS1 
        lxfs = LxFS1 
        N = 1
    elseif scheme == :LxFS3 
        lxfs = LxFS3
        N = 2
    elseif scheme == :LxFS5
        lxfs = LxFS5
        N = 3
    else 
        error("Scheme $(scheme) not in ['LxFS1', 'LxFS3', 'LxFS5']")
    end

    # inject sources
    source_mask = falses(nx, ny, nz)
    inject_sources!(solid, T, source_mask, sources_phys, scheme, wavemode) 

    # fast sweep main loop 
    inner_sweeps = (
        (1+N:1:nx-N,  1+N:1:ny-N,  1+N:1:nz-N),    
        (1+N:1:nx-N,  1+N:1:ny-N,  nz-N:-1:1+N),   
        (1+N:1:nx-N,  ny-N:-1:1+N, 1+N:1:nz-N),   
        (1+N:1:nx-N,  ny-N:-1:1+N, nz-N:-1:1+N),  
        (nx-N:-1:1+N, 1+N:1:ny-N,  1+N:1:nz-N),   
        (nx-N:-1:1+N, 1+N:1:ny-N,  nz-N:-1:1+N),  
        (nx-N:-1:1+N, ny-N:-1:1+N, 1+N:1:nz-N),  
        (nx-N:-1:1+N, ny-N:-1:1+N, nz-N:-1:1+N) 
    )

    converged = false
    diverged  = false
    L2_error = INF
    L∞_error = INF

    if verbose 
        println("===============================================")
        println("Compute Traveltimes: $(wavemode)")
        @printf("Grid Size: %d, %d, %d\n", nx, ny, nz)
        @printf("Viscosities: %.2e | %.2e | %.2e\n", viscosities[1], viscosities[2], viscosities[3])
        println("===============================================")
    end 

    for iter in 1:max_iter

        if verbose
        @printf("Iter: %5d | L2: %.5e | L∞: %.5e\n", iter, L2_error, L∞_error)
        end

        @inbounds begin 
        for (x_order, y_order, z_order) in inner_sweeps
            for i in x_order, j in y_order, k in z_order
                if !source_mask[i, j, k]
                    calc_time!(solid, T, VpVs, UpUs, lxfs, viscosities, velocity_index, i, j, k, dx, dy, dz)
                end
            end
            calc_bcs!(T, nx, ny, nz, N)
        end
        end

        L2_error = L2!(E, T, T_old)
        L∞_error = L∞!(E, T, T_old)
        # check convergence
        if L∞_error  < max_error_tol
            if verbose println("Solution converged after $(iter) iterations.") end 
            converged = true
            break
        end

        # check divergence 
        if any(T .< 0) || any(T .> INF*2) 
            println("Solution diverged. Try larger viscosity_buffer than $(viscosity_buffer).")
            diverged = true
            break
        end

        T_old .= T

    end

    if !converged && !diverged
        println("Solution not converged. Try larger max_iter than $(max_iter).")
    end

    return T

end


end