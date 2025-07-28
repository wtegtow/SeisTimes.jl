# SeisTimes

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://wtegtow.github.io/SeisTimes.jl/dev/)
[![Build Status](https://github.com/wtegtow/SeisTimes.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/wtegtow/SeisTimes.jl/actions/workflows/CI.yml?query=branch%3Amain)

SeisTimes computes first-arrival traveltimes in heterogeneous 2D and 3D anisotropic media. 
It solves the Lax-Friedrich approximation of Hamilton-Jacobi equations applied to the Eikonal equation using a Fast Sweeping numerical scheme. 
Implemented are 1st-, 3rd-, and 5th-order Lax-Friedrichs schemes.

# References

- Grechka, V., Anisotropy and Microseismics: Theory and Practice. Society of Exploration Geophysicists, 2020, Chapter 6.

- Jiang, G. S., and D. Peng, Weighted ENO schemes for Hamilton-Jacobi equations, 2000.

- Kao, C. Y., S. Osher, and J. Qian, Lax-Friedrichs sweeping schemes for
static Hamilton-Jacobi equations, 2004.