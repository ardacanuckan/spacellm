# Security policy

## Reporting a vulnerability

If you discover a security vulnerability in SpaceLLM, please report it privately. Do **not** file a public issue.

Preferred channels (in order):

1. GitHub security advisory, open a private advisory on the repository (will be available once the repo is published).
2. Email, `security@spacellm.org` (placeholder until publication).

Please include:

- A description of the vulnerability.
- The affected component (`packages/spacellm/...`, `packages/web/...`, etc.) and version.
- A minimal reproducer or proof-of-concept.
- Your assessment of severity and any mitigations.

We aim to acknowledge reports within **3 business days** and provide a status update within **10 business days**.

## Scope

In scope:

- Code execution, data exfiltration, or privilege escalation in any package.
- Logic flaws in protection strategies that allow undetected silent corruption.
- Vulnerabilities in published Docker images or distributable artifacts.

Out of scope (please do not report):

- Theoretical attacks against rad-tolerance assumptions made explicit in the documentation (e.g., "this strategy is a best-effort heuristic, not a formal guarantee").
- Issues in unsupported environments (Windows-native, Python ≤ 3.10).
- Denial-of-service via excessive computation requests against developer-mode services running on the user's local machine.

## Disclosure

We follow coordinated disclosure. After a fix lands, we publish a CVE (when applicable) and a release note. Reporters are credited in the advisory unless they request otherwise.

## Threat model context

SpaceLLM specifically defends against radiation-induced bit-flips in transformer inference and training. Protection strategies are best-effort engineering mitigations, not cryptographic guarantees, and are calibrated to published orbital flux profiles. Use cases that depend on adversarial robustness against attackers with weight-write access are out of scope; consider hardware attestation in addition to SpaceLLM.
