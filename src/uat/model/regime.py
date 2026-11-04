"""시장 폭(breadth) 기반 레짐 게이트.

전략은 롱 온리이고 스코어러는 `dist_from_high_252` 를 low 방향으로 가중한다 --
"눌린 종목이 다시 튄다"가 전제다. 전체가 눌린 국면에서는 "눌렸다"가 선별 정보를
담지 않는다. 2022 가 그 국면이었고, 실측 폭(게이트 통과 종목 중 SMA200 위 비율)의
연도별 중간값이 그것을 그대로 보여준다.

    2019 44%   2020 37%   2021 54%   2022 14%   2023 32%   2024 43%   2025 41%

레짐 필터는 **진입 억제이지 청산 규칙이 아니다.** 보유 포지션은 평소대로
스톱/트레일이 관리한다. 폭이 낮다는 이유로 들고 있는 것을 털면 그것은 레짐
판단이 아니라 또 하나의 청산 규칙이고, 청산 레이어는 이미 노이즈로 판정됐다.

인과성: 폭은 날짜 d 까지의 데이터로 계산되고 시그널도 d 종가에 나와 d+1 에
진입한다. 미래 정보가 없다.

`above_sma200` 은 피처 패널의 출력 컬럼이 아니고 `px_vs_sma200`(= close/sma200 - 1)
이 있으므로 그 부호로 센다. 같은 값이다.
"""
from __future__ import annotations

import pandas as pd

MODES = ("none", "level20", "level30", "level40", "sma50", "sma100")


def breadth(panel: pd.DataFrame) -> pd.Series:
    """날짜별 폭. 게이트 통과 행 중 SMA200 위 비율.

    게이트 통과 행만 센다 -- 진입 후보 모집단의 상태를 묻는 것이고, 패널은
    left join 이라 비게이트 행까지 들어 있기 때문이다.
    """
    if "px_vs_sma200" not in panel.columns:
        raise KeyError("px_vs_sma200 가 패널에 없다 -- 폭을 계산할 수 없다")
    pool = panel[panel.get("gated", True) & panel["px_vs_sma200"].notna()]
    if pool.empty:
        return pd.Series(dtype=float)
    return pool.groupby("dt")["px_vs_sma200"].apply(lambda x: (x > 0).mean())


def allow_entries(b: pd.Series, mode: str = "none") -> pd.Series:
    """진입을 허용할 날짜의 불리언 시리즈.

    수준(level*)은 임계값을 하나 고정하는 방식이고, 추세(sma*)는 폭이 자기
    이동평균 위인지만 본다 -- 후자는 "2022 의 폭이 몇 %였나"를 몰라도 되므로
    그 해에 맞춰 숫자를 고르는 형태의 과적합을 피한다.
    """
    if mode == "none":
        return pd.Series(True, index=b.index)
    if mode.startswith("level"):
        return b >= int(mode[5:]) / 100.0
    if mode.startswith("sma"):
        w = int(mode[3:])
        # min_periods 를 절반으로 둔다. 창 앞부분에서 평균이 비면 필터가 조용히
        # 꺼지는데, 그 구간을 "허용"으로 두는 쪽이 "차단"보다 보수적이다
        # (전략을 끄는 것이 수익을 만드는 방향으로 착각되지 않는다).
        return b > b.rolling(w, min_periods=w // 2).mean()
    raise ValueError(f"알 수 없는 레짐 모드: {mode} (가능: {MODES})")


def filter_signals(signals: pd.DataFrame, panel: pd.DataFrame,
                   mode: str = "none") -> pd.DataFrame:
    """레짐이 허용하지 않는 날짜의 시그널을 버린다."""
    if mode == "none":
        return signals
    b = breadth(panel)
    ok = set(pd.Index(b.index)[allow_entries(b, mode).fillna(True)])
    return signals[signals["dt"].isin(ok)]
