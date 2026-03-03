# SeisTimes

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://wtegtow.github.io/SeisTimes.jl/dev/)
[![Build Status](https://github.com/wtegtow/SeisTimes.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/wtegtow/SeisTimes.jl/actions/workflows/CI.yml?query=branch%3Amain)


SeisTimes can be used to compute first-arrival seismic traveltimes in heterogeneous 2D and 3D anisotropic media.

# Method
- Wavefront construction using the Lax-Friedrichs approximation of the static Hamilton-Jacobi (Eikonal) equation for weakly anisotropic media.
  - Fast Sweeping numerical scheme for general heterogeneous media on regular grids.
  - 1st-, 3rd-, and 5th-order Lax-Friedrichs schemes for flexible accuracy.

# Installation
``` julia 
using Pkg
Pkg.add(url="https://github.com/wtegtow/SeisTimes.jl")
```

# Quick Start 

SeisTimes exports 3 functionalities:
- Solid2D 
- Solid3D
- fast sweep

The solid object constructors require:
- **1D coordinate arrays**: `x_coordinates`, `z_coords` (for 2D) or `x_coords`, `y_coords`, `z_coords` (for 3D)
- **Regular velocity grids**: 2D arrays `vp`, `vs` (for 2D problems) or 3D arrays `vp`, `vs` (for 3D problems)

Optional keyword arguments can be used to define anisotropic media:

```julia

iso2D = Solid2D(x_coords, z_coords, vp, vs)
vti2D = Solid2D(x_coords, z_coords, vp, vs; eps=eps, gam=gam, del=del)

iso3D = Solid3D(x_coords, y_coords, z_coords, vp, vs)
ort3D = Solid3D(x_coords, y_coords, z_coords, vp, vs; 
                eps1=eps1, eps2=eps2, gam1=gam1, gam2=gam2,
                del1=del1, del2=del2, del3=del3)

```

With:

- **2D Thomsen parameters:**  
  `eps`, `gam`, `del`

- **3D Tsvankin parameters:**  
  `eps1`, `eps2`, `gam1`, `gam2`, `del1`, `del2`, `del3`

Once defined, the solid objects can be passed to the **fast_sweep** function to compute traveltimes.
**fast_sweep** returns a named tuple `(T=T, converged=converged)`, where `T` is the traveltime grid array and `converged` is a boolean flag indicating whether the solution converged within the specified tolerance. 
The following example demonstrate basic 2D usage:

```julia

using SeisTimes
using GLMakie
Makie.inline!(true)

# 2d example
h = 5
x_coords = 0:h:500 
z_coords = 0:h:500 

vp  = zeros(length(x_coords), length(z_coords)) .+ 2000
vs  = zeros(length(x_coords), length(z_coords)) .+ 1000
eps = zeros(length(x_coords), length(z_coords)) .+ 0.25
del = zeros(length(x_coords), length(z_coords)) .- 0.1

# add some heterogeneity 
vp[:, 18:34] .= 2200.0  
vs[:, 18:34] .= 750.0 

# source location
source = [(100, 400)]

# create solid objects
iso = Solid2D(x_coords, z_coords, vp, vs)
vti = Solid2D(x_coords, z_coords, vp, vs; eps=eps, del=del)

# algorithm parameter 
wavemode = :S # :P, :S 
scheme = :LxFS5 # LxFS1 -> 1st order, LxFS3 -> 3rd order, LxFS5 -> 5th order Lax-Friedrich schemes 
verbose = false  
max_iter=200
max_error_tol = 1e-5   # convergence criterium
viscosity_buffer = 2.5 # stabilizer. If too small, solution diverge, if too large, computations take longer

# compute travel times
tt_iso = fast_sweep(iso, source, wavemode, scheme;
                    max_iter=max_iter, 
                    max_error_tol=max_error_tol,
                    viscosity_buffer=viscosity_buffer)
@assert tt_iso.converged == true


tt_vti = fast_sweep(vti, source, wavemode, scheme;
                    max_iter=max_iter, 
                    max_error_tol=max_error_tol,
                    viscosity_buffer=viscosity_buffer)
@assert tt_vti.converged == true

# visualize
name = ["S", "qS"]
imgs = [tt_iso.T, tt_vti.T] # .T for travel time arrays
fig = Figure(size=(700,400)) 
for (i, img) in enumerate(imgs)
    ax = Axis(fig[1,i], title=name[i])
    im = contourf!(ax, x_coords, z_coords, img, levels=100, colormap=:glasbey_bw_minc_20_n256)
    Colorbar(fig[2,i], im, vertical=false, height=5, label="sec")
end
save("docs/assets/img1.png", fig; px_per_unit=2)
display(fig)

```

![2D Iso Traveltime](docs/assets/img1.png)


The 3D variants work analogously.
See the examples/ folder for more complex use cases.

# References

- Grechka, V., Anisotropy and Microseismics: Theory and Practice. Society of Exploration Geophysicists, 2020, Chapter 6.
- Jiang, G. S., and D. Peng, Weighted ENO schemes for Hamilton-Jacobi equations, 2000.
- Kao, C. Y., S. Osher, and J. Qian, Lax-Friedrichs sweeping schemes for
static Hamilton-Jacobi equations, 2004.