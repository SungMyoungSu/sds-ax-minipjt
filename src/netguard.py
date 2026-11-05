"""외부 통신 허용 목록 (G9, AC8).

MCP 서버와 앱 프로세스에서 설치한다. 허용 호스트(Bedrock 런타임, apis.data.go.kr, 설정된 SMTP, 로컬) 말고는
이름 조회와 연결 단계에서 막는다. 막힌 시도는 NetworkBlocked 예외가 된다(호스트 이름만 담고 비밀 값은 담지 않음).
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable

from config import Settings

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class NetworkBlocked(PermissionError):
    pass


def allowed_hosts(settings: Settings) -> set[str]:
    region = settings.aws_region or "us-east-1"
    hosts = set(LOCAL_HOSTS)
    hosts |= {f"bedrock-runtime.{region}.amazonaws.com", f"bedrock-runtime-fips.{region}.amazonaws.com"}
    hosts.add("apis.data.go.kr")  # 공공 API는 이 호스트만(설정의 Base URL이 다른 호스트면 막힌다)
    if settings.smtp_host:
        hosts.add(settings.smtp_host.lower())
    return {h for h in hosts if h}


def _is_local_ip(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def install_network_guard(settings: Settings) -> Callable[[], None]:
    """socket.getaddrinfo와 socket.create_connection을 감싸 허용 호스트만 통과시킨다.
    되돌리는 함수를 돌려준다(테스트용)."""
    allowed = allowed_hosts(settings)
    original_getaddrinfo = socket.getaddrinfo
    original_create_connection = socket.create_connection

    def check(host: object) -> None:
        name = (host.decode() if isinstance(host, bytes) else str(host or "")).lower().rstrip(".")
        if name in allowed or _is_local_ip(name):
            return
        raise NetworkBlocked(f"허용되지 않은 외부 통신을 막았습니다: {name}")

    def guarded_getaddrinfo(host, *args, **kwargs):
        check(host)
        return original_getaddrinfo(host, *args, **kwargs)

    def guarded_create_connection(address, *args, **kwargs):
        check(address[0])
        return original_create_connection(address, *args, **kwargs)

    socket.getaddrinfo = guarded_getaddrinfo
    socket.create_connection = guarded_create_connection

    def uninstall() -> None:
        socket.getaddrinfo = original_getaddrinfo
        socket.create_connection = original_create_connection

    return uninstall
