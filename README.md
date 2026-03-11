# rmhdgpu

`rmhdgpu` is a foundation for a single-GPU Fourier pseudo-spectral RMHD-style code.

Current scope:

- backend selection for NumPy or CuPy
- periodic 3D grid and Fourier wavenumber setup
- thin real-to-complex / complex-to-real FFT wrappers
- tensor-product 2/3 dealiasing mask
- basic spectral operators and Poisson bracket infrastructure
- random band-limited initial-condition utilities
- a first-pass pytest suite

The full PDE solver, Runge-Kutta timestepper, forcing, and HDF5 output are intentionally not implemented yet.

