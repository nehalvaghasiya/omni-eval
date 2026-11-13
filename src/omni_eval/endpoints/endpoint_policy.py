"""Default-deny endpoint URL policy; HTTP is loopback-only when explicitly approved."""

import ipaddress
import socket
from urllib.parse import urlparse

from omni_eval.exceptions import SecurityError
from omni_eval.schemas.target_spec import TargetSpec


class EndpointPolicy:
    """Strict host allowlist, DNS IP checks and no redirect following.

    For strong DNS-rebinding prevention, deploy behind a network egress filter;
    application pre-resolution alone cannot guarantee connection-time pinning.
    """

    def validate(self, url: str, spec: TargetSpec) -> None:
        parsed = urlparse(url)
        hostname = parsed.hostname

        if (
            parsed.scheme not in ("http", "https")
            or not hostname
            or parsed.username
            or parsed.password
            or parsed.fragment
        ):
            raise SecurityError(
                "target must be a credential-free HTTP(S) URL without fragments"
            )

        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        allowed = {entry.lower() for entry in spec.allowed_hosts}

        if (
            parsed.port is not None and f"{hostname.lower()}:{port}" not in allowed
        ) or (
            parsed.port is None
            and hostname.lower() not in allowed
            and f"{hostname.lower()}:{port}" not in allowed
        ):
            raise SecurityError("target host and port are not explicitly allowlisted")

        if parsed.scheme == "http" and not spec.allow_localhost:
            raise SecurityError(
                "plain HTTP is restricted to explicitly allowed localhost development"
            )

        try:
            addresses = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
        except OSError as exc:
            raise SecurityError("unable to validate target DNS addresses") from exc

        if not addresses:
            raise SecurityError("target DNS returned no addresses")

        for info in addresses:
            address = ipaddress.ip_address(info[4][0])

            if address.is_loopback:
                if not spec.allow_localhost:
                    raise SecurityError("localhost requires allow_localhost=true")
            elif not address.is_global:
                raise SecurityError(
                    "private, link-local, multicast and metadata IP ranges are blocked"
                )
            elif parsed.scheme != "https":
                raise SecurityError("public endpoints require HTTPS")
