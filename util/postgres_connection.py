"""PostgreSQL 연결 (``.env`` 의 ``DATABASE_URL`` / ``POSTGRES_*``)."""

from __future__ import annotations

import os
from urllib.parse import quote_plus

from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase


def postgres_conninfo_from_env() -> str:
    """``.env`` 의 ``DATABASE_URL`` 또는 ``POSTGRES_*`` 로 SQLAlchemy URL을 만든다."""
    if url := os.environ.get("DATABASE_URL"):
        return url
    host = os.environ.get("POSTGRES_HOST")
    if not host:
        msg = (
            "DATABASE_URL 또는 POSTGRES_HOST 등 POSTGRES_* 환경 변수가 "
            "필요합니다. (.env.example 참고)"
        )
        raise ValueError(msg)
    port = os.environ.get("POSTGRES_PORT", "5432")
    user = quote_plus(os.environ.get("POSTGRES_USER", "postgres"))
    password = quote_plus(os.environ.get("POSTGRES_PASSWORD", ""))
    dbname = quote_plus(os.environ.get("POSTGRES_DB", "postgres"))
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{dbname}"


def connect_postgres(
    conninfo: str | None = None,
    *,
    load_env: bool = True,
) -> SQLDatabase:
    """PostgreSQL 연결.

    ``conninfo`` 미지정 시 프로젝트 루트 ``.env`` 를 로드한 뒤
    ``DATABASE_URL`` (우선) 또는 ``POSTGRES_*`` 를 사용한다.
    """
    if load_env:
        load_dotenv(override=True)
    return SQLDatabase.from_uri(conninfo or postgres_conninfo_from_env())


def try_connect_postgres(
    conninfo: str | None = None,
    *,
    load_env: bool = True,
) -> SQLDatabase | None:
    """연결 실패 시 예외 대신 ``None`` 을 반환 (노트북·로컬 점검용)."""
    try:
        return connect_postgres(conninfo, load_env=load_env)
    except Exception as exc:  # noqa: BLE001 — 데모: DB 미기동 시 노트북 계속 진행
        print(f"[PostgreSQL] 연결 실패: {exc}")
        return None


__all__ = [
    "connect_postgres",
    "postgres_conninfo_from_env",
    "try_connect_postgres",
]
