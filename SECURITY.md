# Security Policy

## Reporting vulnerabilities

Please do **not** open a public issue for security problems. Report privately by
emailing the maintainer, or open a security advisory on GitHub
(https://github.com/honeyamn10-source/Quant_Lab/security/advisories/new).

Please include:

- the affected module and version,
- a minimal reproduction,
- the impact.

## Scope

- The research pipeline runs on **local data only**; no secrets are required.
- The FastAPI dashboard is intended for **loopback / trusted networks**.
  Configure authentication and host binding before exposing it publicly.
- Datasets are user-provided; treat external data as untrusted input.
- Never commit API keys or dataset credentials to the repository.