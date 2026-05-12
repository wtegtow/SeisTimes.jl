# SeisTimes

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://wtegtow.github.io/SeisTimes.jl/dev/)
[![Build Status](https://github.com/wtegtow/SeisTimes.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/wtegtow/SeisTimes.jl/actions/workflows/CI.yml?query=branch%3Amain)

SeisTimes is a package for computing first-arrival seismic traveltimes in heterogeneous 2D and 3D anisotropic elastic media.

# Implemented Methods

- Wavefront construction via the Lax-Friedrichs (LxFS) approximation of the static Hamilton-Jacobi for the anisotropic eikonal equation.
The solver uses a Fast Sweeping scheme on regular grids and supports 1st-, 3rd-, and 5th-order LxFS stencils.

# Installation

```julia
using Pkg
Pkg.add(url="https://github.com/wtegtow/SeisTimes.jl")
```

# Quick Start

The main function is `fast_sweep`. It takes a **velocity model array** and a list of sources, and returns a `Vector{FastSweepResult}` — one result per source.

## Velocity Model Format

The velocity model is packed into a single array that stores both the coordinate meshgrid and all elastic parameters. 

**2D model** — array of shape `(7, nx, nz)`:

| Index | Field |
|-------|-------|
| `[1,:,:]` | x-coordinate meshgrid |
| `[2,:,:]` | z-coordinate meshgrid |
| `[3,:,:]` | P-wave velocity `vp` |
| `[4,:,:]` | S-wave velocity `vs` |
| `[5,:,:]` | density `rho` |
| `[6,:,:]` | Thomsen `ε` (zero for isotropic) |
| `[7,:,:]` | Thomsen `δ` (zero for isotropic) |

**3D model** — array of shape `(13, nx, ny, nz)`:

| Index | Field |
|-------|-------|
| `[1,:,:,:]` | x-coordinate meshgrid |
| `[2,:,:,:]` | y-coordinate meshgrid |
| `[3,:,:,:]` | z-coordinate meshgrid |
| `[4,:,:,:]` | P-wave velocity `vp` |
| `[5,:,:,:]` | S-wave velocity `vs` |
| `[6,:,:,:]` | density `rho` |
| `[7,:,:,:]` | Tsvankin `ε₂` |
| `[8,:,:,:]` | Tsvankin `ε₁` |
| `[9,:,:,:]` | Tsvankin `γ₁` |
| `[10,:,:,:]` | Tsvankin `γ₂` |
| `[11,:,:,:]` | Tsvankin `δ₁` |
| `[12,:,:,:]` | Tsvankin `δ₂` |
| `[13,:,:,:]` | Tsvankin `δ₃` |

The velmod array can also be passed as a file path (`.npy`, `.npz`, or `.jld2`).

## `fast_sweep`

```julia
results = fast_sweep(velmod, sources, wavemode, scheme;
    max_iter        = 1000,
    tol             = 1e-6,
    criterion       = :L8,       # :L2 or :L8 (L-infinity)
    viscosity_buffer = 2.0,      # numerical stabilizer (larger values stabilize, but slows down convergence)
    verbose         = false,     # print progress in REPL or terminal
    save            = nothing,   # optional HDF5 output path
)
```

**Arguments:**
- `velmod` — velocity model array `(7,nx,nz)` / `(13,nx,ny,nz)`, or a path to a `.npy` / `.npz` / `.jld2` file
- `sources` — vector of `(x,z)` or `(x,y,z)` tuples, or a path to a whitespace-delimited text file
- `wavemode` — `:P` or `:S` for 2D; `:P`, `:S1`, or `:S2` for 3D
- `scheme` — `:LxFS1`, `:LxFS3`, or `:LxFS5` (1st, 3rd, 5th order)

Multiple sources are computed in parallel via Julia threads. Set the number of threads with the `JULIA_NUM_THREADS` environment variable or the VS Code Julia extension settings.

**Returns** a `Vector{FastSweepResult}` with one entry per source. Each result contains:

| Field | Description |
|-------|-------------|
| `.traveltimes` | traveltime grid array |
| `.converged` | `true` if solution converged |
| `.L2_error` | L2 error history per iteration |
| `.L∞_error` | L∞ error history per iteration |
| `.time_taken` | wall-clock time in seconds |

Results can be saved to HDF5 via the `save` keyword and reloaded with `load_fast_sweep_result`.

## 2D Example

```julia
using SeisTimes
using GLMakie; Makie.inline!(true)

h = 5
x_coords = 0:h:500
z_coords = 0:h:500
nx, nz = length(x_coords), length(z_coords)

# heterogeneous velocity model
vp  = fill(2000.0, nx, nz)
vs  = fill(1000.0, nx, nz)
vp[:, 18:34] .= 2200.0
vs[:, 18:34] .=  750.0

# anisotropy parameters (VTI)
eps = fill( 0.20, nx, nz)
del = fill(-0.10, nx, nz)

# meshgrid coordinates
X = repeat(x_coords, 1, nz)
Z = repeat(reshape(z_coords, 1, :), nx, 1)

# build isotropic and VTI velocity model arrays
velmod_iso = zeros(7, nx, nz)
velmod_iso[1,:,:] .= X;  velmod_iso[2,:,:] .= Z
velmod_iso[3,:,:] .= vp; velmod_iso[4,:,:] .= vs

velmod_vti = copy(velmod_iso)
velmod_vti[6,:,:] .= eps
velmod_vti[7,:,:] .= del

# source
sources = [(100.0, 400.0)]

# compute traveltimes (S-wave, 5th-order scheme)
tt_iso = fast_sweep(velmod_iso, sources, :S, :LxFS5; tol=1e-5, viscosity_buffer=2.5)
tt_vti = fast_sweep(velmod_vti, sources, :S, :LxFS5; tol=1e-5, viscosity_buffer=2.5)

@assert tt_iso[1].converged
@assert tt_vti[1].converged

# visualize
fig = Figure(size=(800, 420))
for (i, (tt, title)) in enumerate(zip(results, titles))
  ax = Axis(fig[1,i], title=title, yreversed=true,
            xlabel="x [m]", ylabel="z [m]")
  im = contourf!(ax, collect(x_coords), collect(z_coords),
                  tt[1].traveltimes, levels=100, colormap=:glasbey_bw_minc_20_n256)
  Colorbar(fig[2,i], im, vertical=false, height=8, label="time [s]")
end
display(fig)

```

![2D Traveltime Example](docs/assets/img1.png)

## Save and Load Results

```julia
# save to HDF5
fast_sweep(velmod, sources, :P, :LxFS5; save="results/tt.h5")

# reload
results = load_fast_sweep_result("results/tt.h5", Val(2)) # Val(2) for 2D, Val(3) for 3D
tt = results[1].traveltimes
```

## 3D Example

The 3D workflow is identical. Use a `(13, nx, ny, nz)` velmod array, provide `(x, y, z)` source tuples, and pick `:P`, `:S1`, or `:S2` as the wavemode. See the `examples/` folder for full use cases including multi-receiver traveltime grids and validation against full-waveform modelling.

# References

- Grechka, V., Anisotropy and Microseismics: Theory and Practice. Society of Exploration Geophysicists, 2020, Chapter 6.
- Jiang, G. S., and D. Peng, Weighted ENO schemes for Hamilton-Jacobi equations, 2000.
- Kao, C. Y., S. Osher, and J. Qian, Lax-Friedrichs sweeping schemes for static Hamilton-Jacobi equations, 2004.