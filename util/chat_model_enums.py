"""`init_chat_model`용 제공자·모델 식별자 enum.

LangChain `langchain.chat_models.base.init_chat_model`이 받는 문자열은
`"{model_provider}:{model}"` 형식이거나, `model` + `model_provider` 인자 조합이다.

- `ChatModelProvider`: LangChain 소스의 `_SUPPORTED_PROVIDERS`와 동일한 **공급자** 전체.
- `*ChatModel` (접미사만): 해당 공급자에 넘기는 **모델 ID** (콜론 뒤 부분).
- `LangChainChatModel`: 자주 쓰는 **완전한** `provider:model` 문자열 (벤더가 모델을 추가·폐기하면
  여기 목록을 수동으로 맞춰야 한다).
"""

from __future__ import annotations

from enum import StrEnum


def chat_model_id(provider: ChatModelProvider, model: str | StrEnum) -> str:
    """`provider`와 모델 접미사를 `init_chat_model`이 받는 단일 문자열로 합친다."""
    suffix = model if isinstance(model, str) else str(model)
    return f"{provider.value}:{suffix}"


class ChatModelProvider(StrEnum):
    """`init_chat_model(..., model_provider=...)` 및 `provider:model` 접두사에 쓰이는 값."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    AZURE_AI = "azure_ai"
    COHERE = "cohere"
    GOOGLE_VERTEXAI = "google_vertexai"
    GOOGLE_GENAI = "google_genai"
    FIREWORKS = "fireworks"
    OLLAMA = "ollama"
    TOGETHER = "together"
    MISTRALAI = "mistralai"
    HUGGINGFACE = "huggingface"
    GROQ = "groq"
    BEDROCK = "bedrock"
    BEDROCK_CONVERSE = "bedrock_converse"
    GOOGLE_ANTHROPIC_VERTEX = "google_anthropic_vertex"
    DEEPSEEK = "deepseek"
    IBM = "ibm"
    XAI = "xai"
    PERPLEXITY = "perplexity"


# --- 모델 접미사만 (콜론 뒤) -------------------------------------------------


class OpenAIChatModel(StrEnum):
    GPT_5 = "gpt-5"
    GPT_5_MINI = "gpt-5-mini"
    GPT_5_NANO = "gpt-5-nano"
    GPT_4_1 = "gpt-4.1"
    GPT_4_1_MINI = "gpt-4.1-mini"
    GPT_4_1_NANO = "gpt-4.1-nano"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O_AUDIO_PREVIEW = "gpt-4o-audio-preview"
    GPT_4O_MINI_AUDIO_PREVIEW = "gpt-4o-mini-audio-preview"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4_TURBO_PREVIEW = "gpt-4-turbo-preview"
    GPT_4 = "gpt-4"
    GPT_4_32K = "gpt-4-32k"
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    O1 = "o1"
    O1_PREVIEW = "o1-preview"
    O1_MINI = "o1-mini"
    O3 = "o3"
    O3_MINI = "o3-mini"
    O3_PRO = "o3-pro"
    O4_MINI = "o4-mini"
    CHATGPT_4O_LATEST = "chatgpt-4o-latest"


class AnthropicChatModel(StrEnum):
    CLAUDE_OPUS_4_5 = "claude-opus-4-5"
    CLAUDE_SONNET_4_5 = "claude-sonnet-4-5"
    CLAUDE_HAIKU_4_5 = "claude-haiku-4-5"
    CLAUDE_OPUS_4_5_20251101 = "claude-opus-4-5-20251101"
    CLAUDE_SONNET_4_5_20250929 = "claude-sonnet-4-5-20250929"
    CLAUDE_HAIKU_4_5_20251015 = "claude-haiku-4-5-20251015"
    CLAUDE_3_5_SONNET_20241022 = "claude-3-5-sonnet-20241022"
    CLAUDE_3_5_HAIKU_20241022 = "claude-3-5-haiku-20241022"
    CLAUDE_3_OPUS_20240229 = "claude-3-opus-20240229"
    CLAUDE_3_SONNET_20240229 = "claude-3-sonnet-20240229"
    CLAUDE_3_HAIKU_20240307 = "claude-3-haiku-20240307"


class GoogleGenAIChatModel(StrEnum):
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite"
    GEMINI_2_0_FLASH = "gemini-2.0-flash"
    GEMINI_2_0_FLASH_LITE = "gemini-2.0-flash-lite"
    GEMINI_1_5_PRO = "gemini-1.5-pro"
    GEMINI_1_5_FLASH = "gemini-1.5-flash"
    GEMINI_1_5_FLASH_8B = "gemini-1.5-flash-8b"


class GoogleVertexAIChatModel(StrEnum):
    """Vertex AI에 배포된 Gemini·기타 모델 ID (GenAI와 이름이 겹치는 경우가 많다)."""

    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite"
    GEMINI_2_0_FLASH = "gemini-2.0-flash"
    GEMINI_1_5_PRO = "gemini-1.5-pro"
    GEMINI_1_5_FLASH = "gemini-1.5-flash"


class MistralAIChatModel(StrEnum):
    MISTRAL_LARGE_LATEST = "mistral-large-latest"
    MISTRAL_SMALL_LATEST = "mistral-small-latest"
    OPEN_MISTRAL_7B = "open-mistral-7b"
    MINISTRAL_8B_LATEST = "ministral-8b-latest"
    PIXTRAL_12B_2409 = "pixtral-12b-2409"


class CohereChatModel(StrEnum):
    COMMAND_R_PLUS = "command-r-plus"
    COMMAND_R = "command-r"
    COMMAND = "command"
    COMMAND_LIGHT = "command-light"


class GroqChatModel(StrEnum):
    LLAMA_3_3_70B_VERSATILE = "llama-3.3-70b-versatile"
    LLAMA_3_1_8B_INSTANT = "llama-3.1-8b-instant"
    MIXTRAL_8X7B_32768 = "mixtral-8x7b-32768"
    GEMMA2_9B_IT = "gemma2-9b-it"


class DeepSeekChatModel(StrEnum):
    DEEPSEEK_CHAT = "deepseek-chat"
    DEEPSEEK_REASONER = "deepseek-reasoner"


class XAIChatModel(StrEnum):
    GROK_3 = "grok-3"
    GROK_3_MINI = "grok-3-mini"
    GROK_2_LATEST = "grok-2-latest"
    GROK_2_VISION_LATEST = "grok-2-vision-latest"
    GROK_BETA = "grok-beta"


class PerplexityChatModel(StrEnum):
    SONAR = "sonar"
    SONAR_PRO = "sonar-pro"
    SONAR_REASONING = "sonar-reasoning"


class OllamaChatModel(StrEnum):
    """로컬 Ollama에 흔히 당겨 쓰는 태그 예시 (환경마다 다름)."""

    LLAMA3_2 = "llama3.2"
    LLAMA3_1 = "llama3.1"
    MISTRAL = "mistral"
    PHI3 = "phi3"


class TogetherChatModel(StrEnum):
    LLAMA_3_3_70B_INSTRUCT_TURBO = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    LLAMA_3_1_70B_INSTRUCT_TURBO = "meta-llama/Llama-3.1-70B-Instruct-Turbo"


class FireworksChatModel(StrEnum):
    LLAMA_V3P3_70B_INSTRUCT = "accounts/fireworks/models/llama-v3p3-70b-instruct"


class BedrockChatModel(StrEnum):
    """Bedrock `model_id` (ChatBedrock)."""

    CLAUDE_3_5_SONNET_V2 = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    CLAUDE_3_5_HAIKU_V1 = "anthropic.claude-3-5-haiku-20241022-v1:0"
    CLAUDE_3_HAIKU_V1 = "anthropic.claude-3-haiku-20240307-v1:0"
    CLAUDE_3_OPUS_V1 = "anthropic.claude-3-opus-20240229-v1:0"


class HuggingFaceChatModel(StrEnum):
    """`ChatHuggingFace`용 `model_id` 예시."""

    LLAMA_3_8B_INSTRUCT = "meta-llama/Meta-Llama-3-8B-Instruct"
    MISTRAL_7B_INSTRUCT = "mistralai/Mistral-7B-Instruct-v0.2"


class IBMChatModel(StrEnum):
    """IBM watsonx 등에서 쓰는 `model_id` 예시."""

    GRANITE_13B_CHAT_V2 = "ibm/granite-13b-chat-v2"


# --- 완전한 init_chat_model 문자열 -------------------------------------------


class LangChainChatModel(StrEnum):
    """`init_chat_model(...)`에 그대로 넣을 수 있는 `provider:model` 조합."""

    # OpenAI
    OPENAI_GPT_5 = "openai:gpt-5"
    OPENAI_GPT_5_MINI = "openai:gpt-5-mini"
    OPENAI_GPT_5_NANO = "openai:gpt-5-nano"
    OPENAI_GPT_4_1 = "openai:gpt-4.1"
    OPENAI_GPT_4_1_MINI = "openai:gpt-4.1-mini"
    OPENAI_GPT_4_1_NANO = "openai:gpt-4.1-nano"
    OPENAI_GPT_4O = "openai:gpt-4o"
    OPENAI_GPT_4O_MINI = "openai:gpt-4o-mini"
    OPENAI_GPT_4_TURBO = "openai:gpt-4-turbo"
    OPENAI_GPT_3_5_TURBO = "openai:gpt-3.5-turbo"
    OPENAI_O1 = "openai:o1"
    OPENAI_O1_MINI = "openai:o1-mini"
    OPENAI_O3 = "openai:o3"
    OPENAI_O3_MINI = "openai:o3-mini"
    OPENAI_O3_PRO = "openai:o3-pro"
    OPENAI_O4_MINI = "openai:o4-mini"
    OPENAI_CHATGPT_4O_LATEST = "openai:chatgpt-4o-latest"

    # Anthropic
    ANTHROPIC_CLAUDE_OPUS_4_5 = "anthropic:claude-opus-4-5"
    ANTHROPIC_CLAUDE_SONNET_4_5 = "anthropic:claude-sonnet-4-5"
    ANTHROPIC_CLAUDE_HAIKU_4_5 = "anthropic:claude-haiku-4-5"
    ANTHROPIC_CLAUDE_SONNET_4_5_20250929 = "anthropic:claude-sonnet-4-5-20250929"
    ANTHROPIC_CLAUDE_3_5_SONNET_20241022 = "anthropic:claude-3-5-sonnet-20241022"
    ANTHROPIC_CLAUDE_3_5_HAIKU_20241022 = "anthropic:claude-3-5-haiku-20241022"
    ANTHROPIC_CLAUDE_3_OPUS_20240229 = "anthropic:claude-3-opus-20240229"
    ANTHROPIC_CLAUDE_3_HAIKU_20240307 = "anthropic:claude-3-haiku-20240307"

    # Google GenAI
    GOOGLE_GENAI_GEMINI_2_5_PRO = "google_genai:gemini-2.5-pro"
    GOOGLE_GENAI_GEMINI_2_5_FLASH = "google_genai:gemini-2.5-flash"
    GOOGLE_GENAI_GEMINI_2_0_FLASH = "google_genai:gemini-2.0-flash"
    GOOGLE_GENAI_GEMINI_1_5_PRO = "google_genai:gemini-1.5-pro"
    GOOGLE_GENAI_GEMINI_1_5_FLASH = "google_genai:gemini-1.5-flash"

    # Google Vertex AI
    GOOGLE_VERTEXAI_GEMINI_2_5_PRO = "google_vertexai:gemini-2.5-pro"
    GOOGLE_VERTEXAI_GEMINI_2_5_FLASH = "google_vertexai:gemini-2.5-flash"
    GOOGLE_VERTEXAI_GEMINI_2_0_FLASH = "google_vertexai:gemini-2.0-flash"

    # Mistral
    MISTRALAI_MISTRAL_LARGE_LATEST = "mistralai:mistral-large-latest"
    MISTRALAI_MISTRAL_SMALL_LATEST = "mistralai:mistral-small-latest"

    # Cohere
    COHERE_COMMAND_R_PLUS = "cohere:command-r-plus"
    COHERE_COMMAND_R = "cohere:command-r"

    # Groq
    GROQ_LLAMA_3_3_70B = "groq:llama-3.3-70b-versatile"
    GROQ_MIXTRAL_8X7B = "groq:mixtral-8x7b-32768"

    # DeepSeek
    DEEPSEEK_CHAT = "deepseek:deepseek-chat"
    DEEPSEEK_REASONER = "deepseek:deepseek-reasoner"

    # xAI
    XAI_GROK_3 = "xai:grok-3"
    XAI_GROK_2_LATEST = "xai:grok-2-latest"

    # Perplexity
    PERPLEXITY_SONAR = "perplexity:sonar"
    PERPLEXITY_SONAR_PRO = "perplexity:sonar-pro"

    # Together
    TOGETHER_LLAMA_3_3_70B = "together:meta-llama/Llama-3.3-70B-Instruct-Turbo"

    # Fireworks
    FIREWORKS_LLAMA_V3P3_70B = "fireworks:accounts/fireworks/models/llama-v3p3-70b-instruct"

    # Bedrock
    BEDROCK_CLAUDE_3_5_SONNET_V2 = "bedrock:anthropic.claude-3-5-sonnet-20241022-v2:0"

    # Bedrock Converse
    BEDROCK_CONVERSE_CLAUDE_3_5_SONNET = (
        "bedrock_converse:anthropic.claude-3-5-sonnet-20241022-v2:0"
    )

    # Azure OpenAI (실제 값은 Azure 포털의 배포 이름과 맞춰야 함)
    AZURE_OPENAI_GPT_4O = "azure_openai:gpt-4o"

    # Azure AI (엔드포인트·카탈로그에 따름 — 예시)
    AZURE_AI_GPT_4O = "azure_ai:gpt-4o"

    # Vertex Model Garden — Anthropic (`ChatAnthropicVertex`용 게이트웨이 모델명)
    GOOGLE_ANTHROPIC_VERTEX_CLAUDE_SONNET_4_5 = (
        "google_anthropic_vertex:claude-sonnet-4-5@20250929"
    )

    # Ollama (예시 태그)
    OLLAMA_LLAMA3_2 = "ollama:llama3.2"
    OLLAMA_MISTRAL = "ollama:mistral"

    # HuggingFace
    HUGGINGFACE_LLAMA3_8B = "huggingface:meta-llama/Meta-Llama-3-8B-Instruct"

    # IBM
    IBM_GRANITE_13B = "ibm:ibm/granite-13b-chat-v2"


__all__ = [
    "AnthropicChatModel",
    "BedrockChatModel",
    "ChatModelProvider",
    "CohereChatModel",
    "DeepSeekChatModel",
    "FireworksChatModel",
    "GoogleGenAIChatModel",
    "GoogleVertexAIChatModel",
    "GroqChatModel",
    "HuggingFaceChatModel",
    "IBMChatModel",
    "LangChainChatModel",
    "MistralAIChatModel",
    "OllamaChatModel",
    "OpenAIChatModel",
    "PerplexityChatModel",
    "TogetherChatModel",
    "XAIChatModel",
    "chat_model_id",
]
