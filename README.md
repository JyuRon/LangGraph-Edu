

## 환경 설정

### 1. UV 설치

UV는 빠르고 효율적인 Python 패키지 관리자입니다.

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. 가상 환경 생성 및 활성화

```bash
# 가상 환경 생성
uv venv

# 가상 환경 활성화 (macOS/Linux)
source .venv/bin/activate

# 가상 환경 활성화 (Windows)
.venv\Scripts\activate
```

### 3. 의존성 설치

```bash
# pyproject.toml 기반 설치
uv sync

# 또는 직접 패키지 설치
uv add install langchain langchain-openai langchain-anthropic langchain-community langgraph python-dotenv
```

### 4. 환경 변수 설정

`.env.example` 파일을 `.env`로 복사하고 API 키를 설정합니다:

```bash
cp .env.example .env
```

`.env` 파일 내용:
```
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
```

