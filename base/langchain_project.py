"""LangChain/LangGraph 앱 공통: ``.env`` 로드, LangSmith 추적 설정."""

from dotenv import load_dotenv

from util import logging


class LangChainProjectSetup:
    """스크립트·노트북·유스케이스 등 어디서든 쓸 수 있는 프로젝트 초기화.

    ``load_env`` / ``langsmith_project`` 는 하위 클래스 ``__init__`` 에서
    ``super().__init__(...)`` 로 넘기면 된다.
    """

    def __init__(
        self,
        *,
        load_env: bool = True,
        langsmith_project: str | None = None,
    ) -> None:
        if load_env:
            load_dotenv(override=True)
        if langsmith_project:
            logging.langsmith(langsmith_project)


__all__ = ["LangChainProjectSetup"]
