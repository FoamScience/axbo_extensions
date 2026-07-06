# Bayesian (BOTorch) extensions for the Ax platform

> This is largely an experimental library basically glueing **multi-fidelity** and **robust** optimization
> into the ax-platform internals. Refactored out of
> [foamBO](https://github.com/FoamScience/OpenFOAM-Multi-Objective-Optimization) to decouple it from its orchestration capabilities.

Extracts foamBO's self-contained (historic) `robustness.py` (qMultiFidHVKG cost loop,
`SubstituteContextFeatures`, `RobustAcquisition` CVaR/MARS, `augment_generator_specs` /
`cycle_context`) into a separate package so more tools can have these capabilities
without getting foamBO's orchestration if they don't want it.
