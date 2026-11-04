"""llm_judge.py - 고정 루브릭 기반 LLM 심사"""
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_aws import ChatBedrockConverse

load_dotenv()

# judge 모델과 파라미터를 상수로 고정합니다. 버전 비교는 항상 같은 judge로 합니다
JUDGE_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
judge_llm = ChatBedrockConverse(
    model=JUDGE_MODEL,
    region_name="us-east-1",
    temperature=0,
)


class JudgeResult(BaseModel):
    score: int = Field(ge=1, le=5)
    is_correct: bool = Field(description="의미가 정답과 일치하는가")
    missing: list[str] = Field(default_factory=list, description="답변에 빠진 핵심")
    reasoning: str


judge = judge_llm.with_structured_output(JudgeResult)

RUBRIC = """다음 기준으로 평가하세요. 길이가 아니라 정확성으로 판단하세요.
5점: 질문에 직접 답하고 정답의 핵심 정보를 정확히 포함하며 추측이 없다
3점: 관련은 있지만 일부 정보가 빠졌거나 장황하다
1점: 질문에 답하지 않거나 사실이 틀리다"""


def evaluate_with_reference(question: str, answer: str, reference: str) -> JudgeResult:
    return judge.invoke(
        f"{RUBRIC}\n\n질문: {question}\n정답: {reference}\n모델 답변: {answer}"
    )
