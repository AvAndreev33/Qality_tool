# Current iteration

## Status

v1.0 quality-analysis core is complete. GUI polished. CUDA backend integrated.

## Recently completed

- GUI visual polish: dark theme, style module, spacing, consistency
- progress bar in status bar (loading + computing)
- CUDA backend: all 39 metrics accelerated via CuPy
- auto GPU/CPU dispatch with silent fallback
- 3D map viewer: hardware-accelerated OpenGL via pyqtgraph, interactive LOD, reusable GL context
- documentation: cuda_backend_spec.md, architecture and roadmap updates

## Next direction (to be scoped)

- height-map computation from WLI signals
- settings-driven algorithm configuration
- research-agent tooling for literature monitoring
