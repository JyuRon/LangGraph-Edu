"""PostgreSQL 연결 유틸 (``.env`` 의 ``POSTGRES_*``).

- ``postgres_url`` — ``.env`` 에서 접속 URL 조회 (``dialect`` 로 형식 선택)
- ``connect_postgres`` — ``SQLDatabase`` (에이전트 SQL 도구)
- ``create_postgres_checkpointer`` — HITL 스레드 상태용 ``PostgresSaver``

``dialect``:
  - ``"sqlalchemy"`` → ``postgresql+psycopg://...`` (LangChain ``SQLDatabase``)
  - ``"libpq"`` → ``postgresql://...`` (``PostgresSaver``, ``pg_dump``)
"""

from __future__ import annotations

import os
from typing import Literal
from urllib.parse import quote_plus

from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

PostgresUrlDialect = Literal["sqlalchemy", "libpq"]

_SQLALCHEMY_DRIVER_PREFIXES = ("postgresql+psycopg://", "postgresql+psycopg2://")


def postgres_url(
    *,
    load_env: bool = True,
    dialect: PostgresUrlDialect = "sqlalchemy",
) -> str:
    """``.env`` 의 ``POSTGRES_*`` 로 접속 URL을 만들고 ``dialect`` 형식으로 반환.

    Args:
        load_env: ``True`` 면 ``.env`` 를 먼저 로드한다.
        dialect: ``"sqlalchemy"`` (``+psycopg``) 또는 ``"libpq"`` (``postgresql://``).
    """
    if load_env:
        load_dotenv(override=True)
    host = os.environ.get("POSTGRES_HOST")
    if not host:
        msg = (
            "POSTGRES_HOST 등 POSTGRES_* 환경 변수가 "
            "필요합니다. (.env.example 참고)"
        )
        raise ValueError(msg)
    port = os.environ.get("POSTGRES_PORT", "5432")
    user = quote_plus(os.environ.get("POSTGRES_USER", "postgres"))
    password = quote_plus(os.environ.get("POSTGRES_PASSWORD", ""))
    dbname = quote_plus(os.environ.get("POSTGRES_DB", "postgres"))
    raw = f"postgresql+psycopg://{user}:{password}@{host}:{port}/{dbname}"

    if dialect == "libpq":
        for prefix in _SQLALCHEMY_DRIVER_PREFIXES:
            if raw.startswith(prefix):
                return "postgresql://" + raw.removeprefix(prefix)
        return raw

    return raw


def connect_postgres(
    *,
    load_env: bool = True,
    optional: bool = False,
) -> SQLDatabase | None:
    """PostgreSQL ``SQLDatabase`` (에이전트·노트북 SQL 실행용).

    ``optional=True`` 이면 연결 실패 시 예외 대신 ``None`` 을 반환한다 (DB 미기동 노트북용).
    """
    try:
        url = postgres_url(load_env=load_env, dialect="sqlalchemy")
        return SQLDatabase.from_uri(url)
    except Exception as exc:  # noqa: BLE001 — optional 데모용
        if optional:
            print(f"[PostgreSQL] 연결 실패: {exc}")
            return None
        raise


def create_postgres_checkpointer(
    *,
    load_env: bool = True,
    setup: bool = True,
    max_pool_size: int = 10,
) -> PostgresSaver:
    """HITL interrupt 재개용 ``PostgresSaver`` (전용 ConnectionPool).

    ``setup=True`` 이면 체크포인트 테이블 마이그레이션을 실행한다 (최초 1회).
    """
    pool = ConnectionPool(
        conninfo=postgres_url(load_env=load_env, dialect="libpq"),
        max_size=max_pool_size,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
    )
    checkpointer = PostgresSaver(pool)
    if setup:
        checkpointer.setup()
    return checkpointer


__all__ = [
    "PostgresSaver",
    "PostgresUrlDialect",
    "connect_postgres",
    "create_postgres_checkpointer",
    "postgres_url",
]
