# ============================================================
# Velocity model parser
# ============================================================

const _VELMOD_NFIELDS_2D = 7
const _VELMOD_NFIELDS_3D = 13

function _load_velmod_file(path::String)::AbstractArray
    @assert ispath(path) "Velocity model file not found: $path"

    if endswith(path, ".npz")
        file = npzread(path)
        @assert haskey(file, "velmod") "NPZ file does not contain key \"velmod\""
        return file["velmod"]

    elseif endswith(path, ".npy")
        return npzread(path)

    elseif endswith(path, ".jld2")
        file = jldopen(path)
        @assert haskey(file, "velmod") "JLD2 file does not contain key \"velmod\""
        arr = file["velmod"]
        close(file)
        return arr

    else
        error("Unsupported velocity model format. Supported: .npy, .npz, .jld2")
    end
end

function _check_velmod(arr::AbstractArray)
    nd = ndims(arr)
    if nd == 3
        size(arr, 1) == _VELMOD_NFIELDS_2D || error("3-D velmod must have $_VELMOD_NFIELDS_2D fields in dim 1, got $(size(arr,1))")
    elseif nd == 4
        size(arr, 1) == _VELMOD_NFIELDS_3D || error("4-D velmod must have $_VELMOD_NFIELDS_3D fields in dim 1, got $(size(arr,1))")
    else
        error("Expected 3-D (2D sim) or 4-D (3D sim) array, got $(nd)-D")
    end
    all(isfinite, arr) || error("Velmod contains NaN or Inf")
end

function parse_velmod(input::Union{String, AbstractArray})
    arr = input isa String ? _load_velmod_file(input) : input
    _check_velmod(arr)
    return arr
end

# ============================================================
# Stiffness & solid 
# ============================================================

struct Stiffness2D
    c11::Float64; c13::Float64; c33::Float64; c55::Float64
end

struct Stiffness3D
    c33::Float64; c55::Float64; c11::Float64; c22::Float64
    c66::Float64; c44::Float64; c13::Float64; c23::Float64; c12::Float64
end

struct Solid{D,S}
    coords::NTuple{D,AbstractVector{Float64}}
    unique_stiffnesses::Vector{S}
    stiffness_ids::Array{Int,D}
end

function build_stiffness_map(stiffness_array)
    unique_c  = Vector{eltype(stiffness_array)}()
    val_to_idx = Dict{eltype(stiffness_array),Int}()
    ids = similar(stiffness_array, Int)
    for I in CartesianIndices(stiffness_array)
        c = stiffness_array[I]
        idx = get!(val_to_idx, c) do
            push!(unique_c, c)
            length(unique_c)
        end
        ids[I] = idx
    end
    return unique_c, ids
end

@inline get_stiffness(solid::Solid{2}, I::CartesianIndex{2}) = solid.unique_stiffnesses[solid.stiffness_ids[I]]
@inline get_stiffness(solid::Solid{3}, I::CartesianIndex{3}) = solid.unique_stiffnesses[solid.stiffness_ids[I]]

function Solid(velmod::AbstractArray{T,3}) where T
    x_coords = velmod[1, :, 1]
    z_coords = velmod[2, 1, :]
    vp  = velmod[3, :, :]
    vs  = velmod[4, :, :]
    eps = velmod[6, :, :]
    del = velmod[7, :, :]
    c33 = @. vp^2
    c55 = @. vs^2
    c11 = @. c33 * (2*eps + 1)
    c13 = @. (c33 - 2*c55) + del * c33
    stiffness_array = Stiffness2D.(c11, c13, c33, c55)
    unique_c, ids = build_stiffness_map(stiffness_array)
    Solid{2, Stiffness2D}((x_coords, z_coords), unique_c, ids)
end

function Solid(velmod::AbstractArray{T,4}) where T
    x_coords = velmod[1, :, 1, 1]
    y_coords = velmod[2, 1, :, 1]
    z_coords = velmod[3, 1, 1, :]
    vp   = velmod[4, :, :, :]
    vs   = velmod[5, :, :, :]
    eps1 = velmod[7, :, :, :]
    eps2 = velmod[8, :, :, :]
    gam1 = velmod[9, :, :, :]
    gam2 = velmod[10, :, :, :]
    del1 = velmod[11, :, :, :]
    del2 = velmod[12, :, :, :]
    del3 = velmod[13, :, :, :]
    c33 = @. vp^2
    c55 = @. vs^2
    c11 = @. (2*eps2 + 1) * c33
    c22 = @. (2*eps1 + 1) * c33
    c66 = @. (2*gam1 + 1) * c55
    c44 = @. c66 / (1 + 2*gam2)
    c13 = @. sqrt(2*c33*(c33-c55)*del2 + (c33-c55)^2) - c55
    c23 = @. sqrt(2*c33*(c33-c44)*del1 + (c33-c44)^2) - c44
    c12 = @. sqrt(2*c11*(c11-c66)*del3 + (c11-c66)^2) - c66
    stiffness_array = Stiffness3D.(c33, c55, c11, c22, c66, c44, c13, c23, c12)
    unique_c, ids = build_stiffness_map(stiffness_array)
    Solid{3, Stiffness3D}((x_coords, y_coords, z_coords), unique_c, ids)
end

# ============================================================
# Phase & group velocity
# ============================================================
@inline function Γn(c::Stiffness2D, n::SVector{2})
    n1, n2 = n
    return @SMatrix [
        c.c11*n1^2 + c.c55*n2^2      (c.c13+c.c55)*n1*n2;
        (c.c13+c.c55)*n1*n2           c.c55*n1^2 + c.c33*n2^2
    ]
end

@inline function Γn(c::Stiffness3D, n::SVector{3,T}) where T
    n1, n2, n3 = n
    return @SMatrix [
         c.c11*n1^2+c.c66*n2^2+c.c55*n3^2 (c.c12+c.c66)*n1*n2                (c.c13+c.c55)*n1*n3;
        (c.c12+c.c66)*n1*n2                c.c66*n1^2+c.c22*n2^2+c.c44*n3^2  (c.c23+c.c44)*n2*n3;
        (c.c13+c.c55)*n1*n3               (c.c23+c.c44)*n2*n3                 c.c55*n1^2+c.c44*n2^2+c.c33*n3^2
    ]
end

@inline _eigvals_2d(a, b, d) = begin
    s = (a + d) * 0.5; t = sqrt(((a - d) * 0.5)^2 + b^2)
    s + t, s - t   # λP, λS
end

# Compute eigenvalues of symmetric 3x3 matrix using closed-form solution (Cardano's method). see, e.g.,
# Siddique, Abu Bakar, and Tariq A. Khraishi. 
# Eigenvalues and eigenvectors for 3×3 symmetric matrices: an analytical approach."
# Journal of Advances in Mathematics and Computer Science 35.7 (2020): 106-118.
@inline function _eigvals_3d(A11, A22, A33, A12, A13, A23)
    q   = (A11 + A22 + A33) / 3.0
    B11 = A11-q;  B22 = A22-q;  B33 = A33-q
    p2  = B11*B11 + B22*B22 + B33*B33 + 2.0*(A12*A12 + A13*A13 + A23*A23)
    p   = sqrt(p2 / 6.0)
    p < 1e-14 && return q, q, q, true   # degenerate (isotropic)
    inv_p = 1.0 / p
    C11=B11*inv_p; C22=B22*inv_p; C33=B33*inv_p
    C12=A12*inv_p; C13=A13*inv_p; C23=A23*inv_p
    r   = (C11*(C22*C33 - C23*C23) - C12*(C12*C33 - C23*C13) + C13*(C12*C23 - C22*C13)) * 0.5
    phi = acos(clamp(r, -1.0, 1.0)) / 3.0
    λ1  = q + 2p*cos(phi)
    λ3  = q + 2p*cos(phi + 2π/3)
    return λ1, 3q - λ1 - λ3, λ3, false
end

@inline function phase_velocity(c::Stiffness2D, n::SVector{2,Float64}, mode::Int)
    Γ = Γn(c, n)
    λP, λS = _eigvals_2d(Γ[1,1], Γ[1,2], Γ[2,2])
    return mode == 1 ? sqrt(max(λP, 0.0)) : sqrt(max(λS, 0.0))
end

@inline function phase_velocity(c::Stiffness3D, n::SVector{3,Float64}, mode::Int)
    Γ = Γn(c, n)
    λ1, λ2, λ3, _ = _eigvals_3d(Γ[1,1], Γ[2,2], Γ[3,3], Γ[1,2], Γ[1,3], Γ[2,3])
    return mode == 1 ? sqrt(max(λ1, 0.0)) : mode == 2 ? sqrt(max(λ2, 0.0)) : sqrt(max(λ3, 0.0))
end

# Group velocity g_m = Γ(u_m) · (n / v_m).
function group_velocity(c::Union{Stiffness2D,Stiffness3D}, n::SVector{D,Float64}, mode::Int) where D
    Γ = Γn(c, n)
    F = eigen(Γ)
    idx = D - mode + 1 # ascending → mode 1 = last
    λ = real(F.values[idx])
    v = sqrt(max(λ, 0.0))
    v < 1e-15 && return zero(SVector{D,Float64})
    u = SVector{D,Float64}(real.(F.vectors[:, idx]))
    return Γn(c, u) * (n / v)
end

# ============================================================
# Artificial viscosity 
# ============================================================

function _sample_directions(::Val{2}, deg_increment)
    [(cos(θ), sin(θ)) for θ in deg2rad.(0:deg_increment:360)]
end

function _sample_directions(::Val{3}, deg_increment)
    [(sin(φ)*cos(θ), sin(φ)*sin(θ), cos(φ))
     for θ in deg2rad.(0:deg_increment:360)
     for φ in deg2rad.(0:deg_increment:90)]
end

function compute_viscosities(solid::Solid{D,S}; deg_increment=3, viscosity_buffer=2.0) where {D,S}
    n_modes = D
    directions = _sample_directions(Val(D), deg_increment)

    # [mode][dim] → maximum group velocity component seen over all stiffnesses/directions
    max_visc = zeros(Float64, n_modes, D)

    for c in solid.unique_stiffnesses
        for raw_n in directions
            n_sv = SVector{D,Float64}(raw_n)
            nn   = norm(n_sv)
            nn < 1e-15 && continue
            n = n_sv / nn
            for m in 1:n_modes
                v_m = phase_velocity(c, n, m)
                v_m < 1e-15 && continue
                g_m = group_velocity(c, n, m)
                for d in 1:D
                    max_visc[m, d] = max(max_visc[m, d], abs(g_m[d] / v_m * viscosity_buffer))
                end
            end
        end
    end

    return ntuple(m -> SVector{D,Float64}(ntuple(d -> max_visc[m, d], D)), n_modes)
end

# ============================================================
# Source initialization
# ============================================================
function parse_source(sources::Union{String, AbstractVector, Tuple})
    if sources isa String
        isfile(sources) || error("Source file not found: $sources")
        mat = readdlm(sources, Float64)
        nrow = size(mat, 1)
        sources = [Tuple(mat[i, :]) for i in 1:nrow]
    end
    for src in sources
        @assert length(src) in (2, 3) "Each source must be (x,z) or (x,y,z)"
        @assert all(isfinite, src) "Source coordinates must be finite"
    end
    return sources
end

# Dirty workaround to initiate source cube
function traveltime_straight_ray(solid::Solid{D}, src, x_grid, mode; n_segments=10) where D
    r = sqrt(sum((x_grid[d] - src[d])^2 for d in 1:D))
    r < 1e-15 && return 0.0

    n_hat    = SVector{D,Float64}(ntuple(d -> (x_grid[d] - src[d]) / r, Val(D)))
    ds       = r / n_segments
    spacings = ntuple(d -> solid.coords[d][2] - solid.coords[d][1], Val(D))
    origins  = ntuple(d -> solid.coords[d][1], Val(D))
    gsizes   = ntuple(d -> length(solid.coords[d]), Val(D))

    t_total = 0.0
    for seg in 1:n_segments
        frac = (seg - 0.5) / n_segments
        I_near = CartesianIndex(ntuple(Val(D)) do d
            xm = src[d] + frac * (x_grid[d] - src[d])
            clamp(round(Int, (xm - origins[d]) / spacings[d]) + 1, 1, gsizes[d])
        end)
        c = get_stiffness(solid, I_near)
        g_m = group_velocity(c, n_hat, mode)
        t_total += ds / (norm(g_m) + 1e-15)
    end
    return t_total
end

# TODO: local ray tracing to refine traveltime estimates in source region
function init_source_region!(solid::Solid{D}, T, source_mask, src, cell_size::Int, mode::Int) where D
    gsizes = ntuple(d -> length(solid.coords[d]), Val(D))
    center = ntuple(Val(D)) do d
        argmin(abs.(solid.coords[d] .- src[d]))
    end
    ranges = ntuple(Val(D)) do d
        clamp(center[d] - cell_size, 1, gsizes[d]):clamp(center[d] + cell_size, 1, gsizes[d])
    end
    for idx in Iterators.product(ranges...)
        I = CartesianIndex(idx)
        source_mask[I] = true
        x_grid = ntuple(d -> solid.coords[d][idx[d]], Val(D))
        T[I] = traveltime_straight_ray(solid, src, x_grid, mode)
    end
end

# ============================================================
# LxFS scheme (per-dimension stencils)
# ============================================================

abstract type LxFSScheme end
struct LxFS1 <: LxFSScheme end
struct LxFS3 <: LxFSScheme end
struct LxFS5 <: LxFSScheme end

stencil_width(::LxFS1) = 1
stencil_width(::LxFS3) = 2
stencil_width(::LxFS5) = 3

@inline function stencil_dim(::LxFS1, T, I, ed, Δd)
    return (T[I + ed], T[I - ed])
end

@inline function stencil_dim(::LxFS3, T, I, ed, Δd)
    ϵ = 1e-8
    T0  = T[I]
    Tp1 = T[I + ed];   Tm1 = T[I - ed]
    Tp2 = T[I + 2*ed]; Tm2 = T[I - 2*ed]
    denom = ϵ + (Tp1 - 2*T0 + Tm1)^2
    γ_p   = (ϵ + (T0 - 2*Tp1 + Tp2)^2) / denom
    γ_m   = (ϵ + (T0 - 2*Tm1 + Tm2)^2) / denom
    ω_p   = 1 / (1 + 2*γ_p^2)
    ω_m   = 1 / (1 + 2*γ_m^2)
    tp = (1-ω_p)/(2*Δd)*(Tp1-Tm1) + ω_p/(2*Δd)*(-3*T0 + 4*Tp1 - Tp2)
    tm = (1-ω_m)/(2*Δd)*(Tp1-Tm1) + ω_m/(2*Δd)*( 3*T0 - 4*Tm1 + Tm2)
    return (T0 + Δd*tp, T0 - Δd*tm)
end

@inline function ΦWENO(a, b, c, d)
    ϵ = 1e-8
    IS0 = 13*(a-b)^2 + 3*(a-3*b)^2
    IS1 = 13*(b-c)^2 + 3*(b+c)^2
    IS2 = 13*(c-d)^2 + 3*(3*c-d)^2
    α0 = 1/(ϵ+IS0)^2
    α1 = 6/(ϵ+IS1)^2
    α2 = 3/(ϵ+IS2)^2
    Σ  = α0 + α1 + α2
    return (1/3)*(α0/Σ)*(a-2*b+c) + (1/6)*((α2/Σ)-0.5)*(b-2*c+d)
end

@inline function stencil_dim(::LxFS5, T, I, ed, Δd)
    T0  = T[I]
    Tp1 = T[I + 1*ed]; Tm1 = T[I - 1*ed]
    Tp2 = T[I + 2*ed]; Tm2 = T[I - 2*ed]
    Tp3 = T[I + 3*ed]; Tm3 = T[I - 3*ed]
    inv_Δd = 1 / Δd
    Δp_m2 = (Tm1 - Tm2) * inv_Δd
    Δp_m1 = (T0  - Tm1) * inv_Δd
    Δp_0  = (Tp1 - T0)  * inv_Δd
    Δp_p1 = (Tp2 - Tp1) * inv_Δd
    a_p = (Tp3 - 2*Tp2 + Tp1) * inv_Δd
    a_m = (Tm3 - 2*Tm2 + Tm1) * inv_Δd
    b_p = (Tp2 - 2*Tp1 + T0)  * inv_Δd
    b_m = (Tm2 - 2*Tm1 + T0)  * inv_Δd
    c_  = (Tp1 - 2*T0  + Tm1) * inv_Δd
    base = (1/12) * (-Δp_m2 + 7*Δp_m1 + 7*Δp_0 - Δp_p1)
    tp   = base + ΦWENO(a_p, b_p, c_, b_m)
    tm   = base - ΦWENO(a_m, b_m, c_, b_p)
    return (T0 + Δd*tp, T0 - Δd*tm)
end

# ============================================================
# LxFS update (multi-dimensional)
# ============================================================

@inline function unit_offset(::Val{D}, d::Int) where D
    CartesianIndex(ntuple(i -> i == d ? 1 : 0, Val(D)))
end

@inline function calc_time!(
        T, I::CartesianIndex{D}, scheme::LxFSScheme,
        spacings::NTuple{D,Float64}, solid::Solid{D},
        viscosities::SVector{D,Float64},
        velocity_index::Int) where D

    Φ = ntuple(Val(D)) do d
        ed = unit_offset(Val(D), d)
        stencil_dim(scheme, T, I, ed, spacings[d])
    end

    p = SVector{D,Float64}(ntuple(Val(D)) do d
        (Φ[d][1] - Φ[d][2]) / (2 * spacings[d])
    end)

    p_norm = norm(p)
    (!isfinite(p_norm) || p_norm == 0) && return
    n_hat = p / p_norm

    c = get_stiffness(solid, I)
    v_phase = phase_velocity(c, n_hat, velocity_index)

    H = 1 / v_phase - p_norm
    A = 1 / sum(ntuple(d -> viscosities[d] / spacings[d], Val(D)))
    C = sum(ntuple(Val(D)) do d
        viscosities[d] * (Φ[d][1] + Φ[d][2]) / (2 * spacings[d])
    end)

    T[I] = min(A * (H + C), T[I])
end

# ============================================================
# Outflow boundary conditions
# ============================================================

# TODO: refactor to multidimensional loop with CartesianIndices and unit offsets
function calc_bcs!(T, grid_sizes::NTuple{2,Int}, N)
    nx, nz = grid_sizes
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

function calc_bcs!(T, grid_sizes::NTuple{3,Int}, N)
    nx, ny, nz = grid_sizes
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

# ============================================================
# Fast Sweeping
# ============================================================

const INF_VAL = 1e5
wavemode_index(::Val{2}, m::Symbol) = m == :P ? 1 : m == :S  ? 2 : error("wavemode $m ∉ [:P, :S]")
wavemode_index(::Val{3}, m::Symbol) = m == :P ? 1 : m == :S1 ? 2 : m == :S2 ? 3 : error("wavemode $m ∉ [:P, :S1, :S2]")

function scheme_from_symbol(s::Symbol)
    s == :LxFS1 && return LxFS1()
    s == :LxFS3 && return LxFS3()
    s == :LxFS5 && return LxFS5()
    error("scheme $s ∉ [:LxFS1, :LxFS3, :LxFS5]")
end

function generate_sweeps(grid_sizes::NTuple{D,Int}, Ns::Int) where D
    sweeps = NTuple{D,StepRange{Int,Int}}[]
    for bits in 0:(2^D - 1)
        push!(sweeps, ntuple(Val(D)) do d
            lo = 1 + Ns; hi = grid_sizes[d] - Ns
            (bits >> (d-1)) & 1 == 0 ? (lo:1:hi) : (hi:-1:lo)
        end)
    end
    return sweeps
end

# Progress logger
struct SweepLogger
    nsrc::Int
    lock::ReentrantLock
    active::Bool
    grid_sizes::Tuple
    wavemode::Symbol
end

SweepLogger(nsrc::Int, active::Bool, grid_sizes::Tuple, wavemode::Symbol) =
    SweepLogger(nsrc, ReentrantLock(), active, grid_sizes, wavemode)

function _log_init!(lg::SweepLogger)
    lg.active || return
    grid_str = join(lg.grid_sizes, " × ")
    println("─"^75)
    println("Fast Sweeping for Eikonal Equation")
    println("  Grid:     ", grid_str)
    println("  Wavemode: ", lg.wavemode)
    println("  Sources:  ", lg.nsrc,  "  | Available threads: ", Threads.nthreads())
    println("─"^75)
    for i in 1:lg.nsrc
        @printf("Source %4d | Iter: %4s | L2: %11s | L∞: %11s\n", i, "—", "—", "—")
    end
    flush(stdout)
end

function _log_update!(lg::SweepLogger, isrc::Int, iter::Int, l2::Float64, linf::Float64, note::String="")
    lg.active || return
    n_up = lg.nsrc - isrc + 1
    lock(lg.lock) do
        print("\e[$(n_up)A\r")
        if isempty(note)
            @printf("Source %4d | Iter: %4d | L2: %.5e | L∞: %.5e", isrc, iter, l2, linf)
        else
            @printf("Source %4d | Iter: %4d | L2: %.5e | L∞: %.5e  ← %s", isrc, iter, l2, linf, note)
        end
        print("\e[$(n_up)B\r")
        flush(stdout)
    end
end

_log_finish!(lg::SweepLogger) = lg.active && println()

# IO 
struct FastSweepResult{D}
    traveltimes::AbstractArray{Float64,D}
    converged::Bool
    L2_error::Vector{Float64}
    L∞_error::Vector{Float64}
    time_taken::Float64
end

function save_fast_sweep_result(results::Vector{FastSweepResult{D}}, path::String) where D
    h5open(path, "w") do fid
        for (isrc, res) in enumerate(results)
            grp = create_group(fid, "source_$isrc")
            for fname in fieldnames(FastSweepResult{D})
                val = getfield(res, fname)
                if val isa AbstractArray
                    grp[string(fname)] = Array(val)
                elseif val isa Bool
                    grp[string(fname)] = Int8(val)
                else
                    grp[string(fname)] = val
                end
            end
        end
    end
end

function load_fast_sweep_result(path::String, ::Val{D}) where D
    path = endswith(path, ".h5") ? path : path * ".h5"
    results = h5open(path, "r") do fid
        nsrc = length(keys(fid))
        map(1:nsrc) do isrc
            grp = fid["source_$isrc"]
            traveltimes = read(grp, "traveltimes")::Array{Float64,D}
            converged   = Bool(read(grp, "converged"))
            L2_error    = read(grp, "L2_error")::Vector{Float64}
            L∞_error    = read(grp, "L∞_error")::Vector{Float64}
            time_taken  = read(grp, "time_taken")::Float64
            FastSweepResult{D}(traveltimes, converged, L2_error, L∞_error, time_taken)
        end
    end
    return results
end


# Main
function fast_sweep(
    velmod::Union{String, AbstractArray},
    sources::Union{String, AbstractVector, Tuple},
    wavemode::Symbol,
    scheme::Symbol;
    max_iter::Int  = 1000,
    criterion::Symbol = :L8,
    tol::Float64   = 1e-6,
    viscosity_buffer::Float64 = 2.0,
    verbose::Bool  = false,
    save::Union{String, Nothing} = nothing,
)
    # check path
    if !isnothing(save)
        isempty(strip(save)) && error("save path must not be an empty string")
        save = endswith(save, ".h5") || endswith(save, ".hdf5") ? save : save * ".h5"
        save_dir = dirname(save)
        if !isempty(save_dir) && !isdir(save_dir)
            error("Save directory does not exist: $save_dir")
        end
    end

    solid       = Solid(parse_velmod(velmod))
    sources     = parse_source(sources)
    viscosities = compute_viscosities(solid, viscosity_buffer=viscosity_buffer)

    D          = length(solid.coords)
    wm_index   = wavemode_index(Val(D), wavemode)
    lxfs       = scheme_from_symbol(scheme)
    ns         = stencil_width(lxfs)
    grid_sizes = ntuple(d -> length(solid.coords[d]), D)
    spacings   = ntuple(d -> solid.coords[d][2] - solid.coords[d][1], Val(D))
    colons     = ntuple(_ -> (:), D)
    sweeps     = generate_sweeps(grid_sizes, ns)

    T_grid = fill(INF_VAL, length(sources), grid_sizes...)
    T_mask = falses(length(sources), grid_sizes...)

    for s in eachindex(sources)
        init_source_region!(solid, view(T_grid, s, colons...), view(T_mask, s, colons...),sources[s], ns, wm_index)
    end

    nsrc    = length(sources)
    results = Vector{FastSweepResult{D}}(undef, nsrc)
    lg      = SweepLogger(nsrc, verbose, grid_sizes, wavemode)

    _log_init!(lg)

    Threads.@threads for isrc in eachindex(sources)

        t_grid       = view(T_grid, isrc, colons...)
        t_mask       = view(T_mask, isrc, colons...)
        t_old        = copy(t_grid)
        error_buffer = similar(Array(t_grid))

        t_start = time()
        converged = false
        L2_error  = Float64[INF_VAL]
        L∞_error  = Float64[INF_VAL]

        for iter in 1:max_iter

            _log_update!(lg, isrc, iter, L2_error[end], L∞_error[end])

            @inbounds for sweep_ranges in sweeps
                for idx in Iterators.product(sweep_ranges...)
                    I = CartesianIndex(idx)
                    t_mask[I] || calc_time!(t_grid, I, lxfs, spacings, solid, viscosities[wm_index], wm_index)
                end
                calc_bcs!(t_grid, grid_sizes, ns)
            end

            if any(<(0), t_grid)
                _log_update!(lg, isrc, iter, L2_error[end], L∞_error[end], "Unstable, try larger viscosity_buffer.")
                break
            end

            @. error_buffer = t_grid - t_old
            push!(L2_error, norm(error_buffer, 2))
            push!(L∞_error, norm(error_buffer, Inf))

            if criterion == :L2
                L2_error[end] < tol && (converged = true; break)
            elseif criterion == :L8
                L∞_error[end] < tol && (converged = true; break)
            else 
                error("Unsupported convergence criterion: $criterion. Use :L2 or :L8.")
            end

            copyto!(t_old, t_grid)
        end

        results[isrc] = FastSweepResult{D}(t_grid, converged, L2_error, L∞_error, time() - t_start)
    end

    _log_finish!(lg)
    !isnothing(save) && save_fast_sweep_result(results, save)

    return results
end