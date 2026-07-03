"""axbo_extensions — shared Ax/BoTorch GLUE for multi-fidelity + robust optimization.

Home for foamBO's ported MultiFid/robust glue (qMultiFidHVKG cost loop, SubstituteContextFeatures,
RobustAcquisition CVaR/MARS, augment_generator_specs/cycle_context).
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("axbo_extensions")
except PackageNotFoundError:  # running from source without an install
    __version__ = "0.0.0"
