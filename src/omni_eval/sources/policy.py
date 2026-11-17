"""No arbitrary URLs: validate target host, IP, port and scheme."""

import ipaddress
import socket
from urllib.parse import urlsplit


class HostPolicy:
    def check_url(
        self,
        url: str,
        allowed_hosts: list[str],
        *,
        local_http: bool = False,
        allowed_ports: list[int] | None = None,
    ) -> None:
        parsed = urlsplit(url)
        host = parsed.hostname
        if not host or host not in allowed_hosts:
            raise PermissionError("target host is not explicitly allowed")
        if parsed.username or parsed.password or parsed.fragment:
            raise PermissionError("credentials or fragments in URL are forbidden")
        if parsed.scheme != "https" and (
            not (
                local_http
                and parsed.scheme == "http"
                and (host in {"localhost", "127.0.0.1", "::1"})
            )
        ):
            raise PermissionError("HTTPS required, except explicitly approved local HTTP")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if not 1 <= port <= 65535 or port not in (
            allowed_ports if allowed_ports is not None else [443]
        ):
            raise PermissionError("target port is not explicitly allowed")
        try:
            infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except OSError as exc:
            raise PermissionError("target hostname cannot be resolved") from exc
        if not infos:
            raise PermissionError("hostname has no addresses")
        for info in infos:
            ip = ipaddress.ip_address(info[4][0])
            if not ip.is_global and (not (local_http and ip.is_loopback)):
                raise PermissionError("private, metadata, and non-global endpoints are denied")
