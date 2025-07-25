module SeisTimes

using StaticArrays, LinearAlgebra, Einsum
using Printf, Test

export fast_sweep, Solid2D

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
   
    function Solid2D(x_coords, z_coords, rho, vp, vs; eps=0.0, del=0.0)
        μ = @. rho * vs^2
        λ = @. rho * vp^2 - 2*μ
        # density normalized stiffness
        c11 = @. ((λ + 2μ) * (2*eps .+ 1)) / rho
        c13 = @. (λ + del * (λ + 2μ)) / rho
        c33 = @. (λ + 2μ) / rho
        c55 = @. (μ) / rho

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

function inject_sources!(solid::Solid2D, T, source_mask, sources, scheme, wavemode, nx, nz, dx, dz) 

    if scheme == :LxFS1 
        for (sx, sz) in sources
            T[sx, sz] = 0.
            source_mask[sx, sz] = true
        end
    
    elseif scheme == :LxFS3 || scheme == :LxFS5

        cell_size = scheme == :LxFS3 ? 1 : 2
        VpVs1 = MVector{2,Float64}(undef)
        UpUs1 = MMatrix{2,2,Float64}(undef)
        VpVs2 = MVector{2,Float64}(undef)
        UpUs2 = MMatrix{2,2,Float64}(undef)

        for (sx, sz) in sources
            for i in clamp(sx-cell_size, 1, nx):clamp(sx+cell_size, 1, nx)
                for k in clamp(sz-cell_size, 1, nz):clamp(sz+cell_size, 1, nz)

                    source_mask[i, k] = true 
                    if (i,k) == (sx, sz)
                        T[i, k] = 0.
                    else 
                        
                        dxi = i - sx
                        dzi = k - sz

                        n = SVector(dxi, dzi) / sqrt(dxi^2 +dzi^2)
                        r = sqrt((dxi * dx)^2 + (dzi * dz)^2)
                        solve_christoffel!(VpVs1, UpUs1, solid, n, i, k)
                        solve_christoffel!(VpVs2, UpUs2, solid, n, sx, sz)

                        # This might be not very accurate for strong heterogenities around the source
                        if wavemode == :P 
                            T[i,k] = r / ((VpVs1[1] + VpVs2[1])/2)
                        elseif wavemode == :S 
                            T[i,k] = r / ((VpVs1[2] + VpVs2[2])/2)
                        end
                       
                    end
                end 
            end
        end
    else
        error("Scheme $(scheme) not in ['LxFS1', 'LxFS3', 'LxFS5']")
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
                    source_phys, 
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
    sources = Tuple{Int, Int}[(argmin(abs.(x_coords .- x)), 
                               argmin(abs.(z_coords .- z))) 
                               for (x, z) in source_phys]
    source_mask = falses(nx, nz)
    inject_sources!(solid, T, source_mask, sources, scheme, wavemode, nx, nz, dx, dz) 

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



end