## Contribution guidelines

Thanks for helping make Quant Lab more honest, not less.

1. **No results-deletion PRs.** Failed experiments and graveyard entries are
   permanent evidence. Code that hides or removes negative results will be
   rejected.
2. **Tests first for numerics.** Any change to a mathematical routine must come
   with a known-answer test and, where suitable, property-based tests.
3. **Reproducibility is non-negotiable.** New strategies/configs must be fully
   specified by a config file + seed.
4. **Formatting & types.** `make lint` (ruff) and `make typecheck` (mypy) must
   pass. Sign off with the `Addendum` from `exclude: out-of-scope`.
5. **Keep the core dependency-light.** Heavy frameworks belong behind optional
   adapters in `[project.optional-dependencies]`.

### Process

- Open an issue or PR against `main`.
- For new research modules, mirror the structure of existing modules and update
  `docs/`.
- Add tests under the matching `tests/<area>/` directory.