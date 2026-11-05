"""공용 예외."""


class InputError(ValueError):
    """입력 오류. 메시지는 사용자에게 그대로 보여 줄 수 있는 '입력 오류: …' 문장이다.
    (허위 기록 거절 고정 문구와 다르다. 가드레일이 아니라 입력 검증이다.)"""

    def __init__(self, reason: str):
        super().__init__(f"입력 오류: {reason}")
