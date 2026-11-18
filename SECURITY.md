# Security policy

## Supported versions

OmniEval is a development preview. Security fixes are considered for the current 0.2 development line on the default branch. Older versions do not have a maintained backport policy.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting when available: open the repository's [Security advisories](https://github.com/nehalvaghasiya/omni-eval/security/advisories) and choose **Report a vulnerability**.

If that button is unavailable, open an issue requesting a private reporting channel, without vulnerability details. Maintainers must enable private vulnerability reporting in the repository's security settings before the GitHub reporting form is available.

Include the following in the private report:

- Affected version or commit, Python version, and platform.
- A minimal reproduction with synthetic data and the relevant configuration.
- Expected and observed behavior, potential impact, and sanitized logs.
- A suggested fix or mitigation, if known.

Keep exploit details and credentials out of public issues. Reports are handled on a best-effort basis; this project does not promise a response or remediation deadline.

## Deployment boundaries

- The optional read-only API has no production authentication. Bind it to loopback.
- HTTP sources and judges send data only through explicit configuration and endpoint policy. Judge use requires consent to forward case data.
- Host allowlists and redirect refusal do not provide a complete isolation boundary or DNS rebinding defense.
- Artifact checksums detect file changes; they are not signatures and do not establish authenticity.
- Format regular expressions are intended for trusted configuration. The package does not provide a sandbox for adversarial inputs.
- Code and SQL metrics inspect predictions without executing them. Security evaluation scores require explicit oracle evidence and do not establish that a system is secure.

The security workflow runs regression tests for implemented controls. It does not certify the package or perform a comprehensive dependency or deployment audit.
