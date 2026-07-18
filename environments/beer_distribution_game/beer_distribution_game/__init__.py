"""Prime Intellect Beer Distribution Game environment.

The simulator modules remain importable without Verifiers so domain tests and
non-Hub baselines do not acquire a framework dependency. The Hub package itself
declares Verifiers 0.2.0, so its native exports are always present when installed.
"""

__version__ = "0.2.0"

try:
    from .harness import BeerHarness, BeerHarnessConfig
    from .taskset import BeerTaskset, BeerTasksetConfig
except ModuleNotFoundError as exc:
    if exc.name != "verifiers":
        raise
    __all__: list[str] = []
else:
    __all__ = ["BeerTaskset", "BeerHarness"]
