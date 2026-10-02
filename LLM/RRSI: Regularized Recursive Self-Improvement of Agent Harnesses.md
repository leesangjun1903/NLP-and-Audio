# RRSI: Regularized Recursive Self-Improvement of Agent Harnesses

**검토 대상:** 업로드된 *RRSI: Regularized Recursive Self-Improvement of Agent Harnesses*, arXiv:2609.24972v2, 총 24쪽입니다. 아래 쪽수는 PDF에 인쇄된 쪽수를 기준으로 하며, **[저자 보고]**, **[검토·해석]**, **[후속 연구 제안]**을 구분합니다. 외부 비교 문헌의 확인 기준일은 **2026년 10월 2일**입니다. :chatgpt-content-reference{index="0"}

## 1. Executive summary

RRSI는 언어모델의 가중치를 고정한 상태에서 주변 실행 프로그램인 **하네스**를 자동 개선할 때 발생하는 과적합을 줄이려는 연구입니다.  
연구의 출발점은 동일한 문제 집합을 반복해서 평가하며 하네스를 수정하면, 실제로 재사용할 수 있는 능력보다 해당 문제의 특수성이나 평가 잡음에 적응할 수 있다는 것입니다. **[pp.1–3]** :chatgpt-content-reference{index="1"} :chatgpt-content-reference{index="2"}

> **용어:** 하네스는 프롬프트, 도구 연결, 실행 순서, 메모리, 문맥 관리처럼 모델이 어떻게 일하게 할지를 결정하는 주변 소프트웨어를 뜻합니다.

제안 방법은 수정 후보를 만드는 단계에서는 수정량·과거 실험 이력·탐색 방향을 제어하고, 후보를 채택하는 단계에서는 벤치마크 특화 내용·성능 변동·추가 비용·불필요한 구조를 점검하는 **정규화**입니다. **[Figure 2, pp.4–6]** :chatgpt-content-reference{index="3"} :chatgpt-content-reference{index="4"}

> **용어:** 여기서 정규화는 값의 범위를 맞추는 normalization이 아니라, 제한된 평가 자료에 지나치게 맞추지 않도록 탐색과 선택에 제약을 주는 regularization입니다.

저자는 8개 벤치마크에서 진화에 사용하지 않은 6개 평가 분할 모두의 개선과 최대 4.7점의 OOD 개선을 보고하지만, 초록의 **최대 14.1점 개선은 Gemini 3.5 Flash의 진화용 벤치마크 성적**이지 OOD 개선량은 아닙니다. **[Figure 3, Table 3, pp.8–9]** :chatgpt-content-reference{index="5"} :chatgpt-content-reference{index="6"}

> **용어:** OOD는 진화에 사용한 데이터와 다른 분포의 평가 환경이며, 이 논문에서는 주로 다른 벤치마크로 옮겨 평가하는 상황을 가리킵니다.

업무 에이전트 실험에서 OOD 평균은 초기 하네스의 39.7에서 43.6으로 높아졌고, 정책 토큰 사용량은 비정규화 진화의 3.80백만에서 2.42백만으로 약 36.3% 감소했지만, 초기 하네스보다는 약 55.1% 증가했습니다. **[Table 2, p.9; 비율은 표의 수치로 계산]** :chatgpt-content-reference{index="7"}

진화에 참여하지 않은 더 작은 모델에도 개선된 하네스를 적용해 성능 상승을 관찰했지만, 이 실험은 진화에 사용했던 벤치마크에서 수행되어 **새 모델과 새 벤치마크로의 동시 일반화**를 입증하지는 않습니다. **[Table 4, p.10]** :chatgpt-content-reference{index="8"}

종합하면, 이 연구는 **모델 가중치가 아닌 에이전트 시스템 수준의 일반화 개선 가능성**을 보여주는 유망한 실증 연구이지만, 통계적 불확실성 보고와 일부 재현성 정보가 부족하므로 일반화의 보편적 보장이나 통계적으로 확정된 우월성으로 해석해서는 안 됩니다. **[Tables 1–5, pp.8–10, 23; Limitations, p.11]** :chatgpt-content-reference{index="9"} :chatgpt-content-reference{index="10"}

---

## 2. 연구의 목적·필요성과 핵심 주장

### 2.1 무엇을 해결하려는가?

**[저자 보고]** 연구 대상은 **적응적 과적합(adaptive overfitting)**입니다. 하네스 진화에서는 같은 진화용 문제 집합의 평가 결과가 다음 수정 후보를 만드는 데 사용되고, 그 후보를 다시 같은 문제 집합에서 평가합니다. 따라서 이 문제 집합은 사실상 최적화에 사용되는 데이터이며, 그 점수가 올랐다는 사실만으로 새로운 문제에서의 성능 향상을 보장할 수 없습니다. **[§2, p.3]** :chatgpt-content-reference{index="11"}

> **용어 풀이 — 적응적 과적합:** 시험 결과를 보고 공부 내용을 계속 바꾼 뒤 똑같은 시험을 다시 치르는 상황과 비슷합니다. 같은 시험의 점수 상승과 처음 보는 시험에서의 실력 향상은 다를 수 있습니다.

저자가 구분하는 주요 실패 원인은 **벤치마크 특화 규칙의 축적**, **평가 잡음에 대한 추종**, **전이 효과 없이 복잡성과 계산량만 증가하는 현상**입니다. RRSI는 수정 가능한 하네스 구성요소를 제한하기보다, 수정 후보의 생성과 채택 과정을 규제하려고 합니다. **[§1, p.2; §3.1, p.4]** :chatgpt-content-reference{index="12"} :chatgpt-content-reference{index="13"}

**[검토·해석]** 따라서 이 연구의 중요한 질문은 “자동으로 몇 점을 더 올렸는가?”보다 **“그 상승분 중 무엇이 새로운 문제에서도 남는가?”**입니다. 또한 계산량을 늘려 얻은 상승과 하네스의 재사용 가능한 개선을 구분해야 한다는 문제의식이 핵심입니다.

### 2.2 핵심 주장과 근거

| 핵심 주장 | 저자가 제시한 근거와 위치 | 검토·해석 및 주의점 |
|---|---|---|
| 진화용 점수 상승과 일반화 성능 상승은 다르다. | 업무 에이전트에서 기존 방법들은 진화용 성적을 높이지만 OOD 개선은 작거나 음수이다. **Figure 1, p.2; Table 1, p.8.** :chatgpt-content-reference{index="14"} :chatgpt-content-reference{index="15"} | 해당 실험에서는 분명한 경향이지만, 기존 방법이 모든 환경에서 일반화에 실패한다는 뜻은 아니다. |
| 수정 후보의 생성 과정도 정규화해야 한다. | 수정량 감소, 전체 실험 이력 활용, 정체 시 미탐색 구성요소 탐색을 사용한다. **§3.2, p.5.** :chatgpt-content-reference{index="16"} | 탐색이 특정 프롬프트 수정이나 이미 실패한 가설에 몰리는 것을 줄이려는 설계이다. |
| 점수가 오른 후보라도 그대로 채택하면 안 된다. | 사전 누출 검사, 최고 성적 기준 하한, 비용 조건, 구조 가지치기를 적용한다. **§3.3, pp.5–6; Appendix C.3, pp.21–22.** :chatgpt-content-reference{index="17"} :chatgpt-content-reference{index="18"} | 다만 작은 양의 개선만 허용하는 방식은 아니다. 비용 절감이나 구조적 새로움 때문에 잡음 범위 안의 후보도 채택할 수 있다. |
| 개선이 새로운 벤치마크로 전이된다. | 5개 OOD 벤치마크와 Harvey LAB의 ID 보류 분할에서 모두 초기 하네스보다 높은 점수를 보고한다. **Figure 3, p.8.** :chatgpt-content-reference{index="19"} | 여러 평가 환경에 걸친 긍정적 증거이다. 다만 도메인마다 별도로 진화했으므로 하나의 범용 하네스를 발견한 실험은 아니다. |
| 제안·선택 양쪽 제약이 모두 중요하다. | 어느 한쪽을 제거해도 OOD 평균이 낮아진다. **Table 2, p.9.** :chatgpt-content-reference{index="20"} | 두 묶음의 효과는 지지하지만, 7개 세부 장치 각각의 독립적 효과는 분리하지 못한다. |
| 특정 실행 모델에만 묶이지 않는다. | 두 모델로 별도 진화를 수행하고, 한 하네스를 더 작은 미참여 모델에 그대로 적용한다. **Tables 3–4, pp.9–10.** :chatgpt-content-reference{index="21"} :chatgpt-content-reference{index="22"} | “여러 모델에서 방법이 작동한다”와 “동일 하네스가 모든 모델에 최적이다”는 구분해야 한다. |
| 비정규화 진화보다 효율적이다. | 업무 에이전트의 최종 하네스 비용이 3.80백만에서 2.42백만 정책 토큰으로 감소한다. **Table 2, p.9; Figure 4, p.10.** :chatgpt-content-reference{index="23"} :chatgpt-content-reference{index="24"} | 초기 하네스보다 저렴한 것은 아니며, 진화 과정 전체 비용을 줄였다는 결과도 아니다. |

---

## 3. 모델 구조와 제안 방법: 수식 중심 설명

### 3.1 새로운 신경망 구조가 아니라, 새로운 하네스 최적화 절차다

**[저자 보고]** 에이전트는 다음과 같이 정의됩니다.

$$
A=(\pi,H)
$$

**기호:** $A$는 전체 에이전트, $\pi$는 고정된 실행 정책 모델, $H$는 수정 대상인 하네스입니다. **[§2, p.3]** :chatgpt-content-reference{index="25"}

> **용어 풀이 — 정책과 백본:** 정책은 현재 입력을 보고 다음 행동을 정하는 모델을 뜻하며, 백본은 이 역할을 수행하는 기반 언어모델입니다. “동결”은 이 모델의 가중치를 학습으로 바꾸지 않는다는 뜻입니다.

시스템은 두 층으로 이해할 수 있습니다.

| 층 | 구성과 역할 |
|---|---|
| **문제를 푸는 실행 층** | 하네스가 프롬프트·문맥·도구를 구성하고, 정책 모델이 행동하며, 환경과 상호작용한 뒤 결과물을 제출한다. |
| **하네스를 바꾸는 진화 층** | 실행 결과를 분석하고, 수정 후보를 만들고, 누출 여부를 검사하고, 후보의 점수와 비용을 평가해 다음 하네스를 선택한다. |

주 실험의 실행 정책, 후보 제안자, 실패 분석자, 누출 검사자는 모두 **Claude Opus 4.8**을 사용합니다. 코딩의 초기 하네스는 **Terminus-2**이며, 업무·공학 환경은 ReAct 실행 루프, 도구 게이트웨이, 동적 도구 목록과 문맥 관리 등을 사용합니다. **[§4.1, p.7]** :chatgpt-content-reference{index="26"}

> **용어 풀이 — ReAct:** 추론과 도구 실행을 번갈아 수행하는 에이전트 구성입니다. 생각만 이어가는 대신, 검색이나 코드 실행의 결과를 받아 다음 행동을 바꿉니다. :chatgpt-content-reference{index="27"}

**[검토·해석]** 따라서 이 논문은 모델의 층 수, 어텐션 구조, 파라미터 수를 바꾼 연구가 아닙니다. 또한 실행 결과로 하네스가 개선되는 의미의 RSI이지, 제안자와 평가자 자체까지 계속 재설계되어 자기개선 능력이 가속되는 것을 실험한 연구는 아닙니다.

### 3.2 무엇을 측정하는가? 성능과 정책 토큰 비용

**[저자 보고: 원문 식 (1)]**

$$
\begin{aligned}
S(H;\mathcal D)
&=
\mathbb E_{x\sim\mathcal D}
\mathbb E_{\tau\sim A(\cdot\mid x)}
\left[r(x,\tau)\right],\\
C(H;\mathcal D)
&=
\mathbb E_{x\sim\mathcal D}
\mathbb E_{\tau\sim A(\cdot\mid x)}
\left[c(\tau)\right].
\end{aligned}
$$

**기호:** $\mathcal D$는 문제 집합 또는 문제 분포, $x$는 문제, $\tau$는 한 번의 실행 과정, $r(x,\tau)\in[0,1]$은 검증기의 평가 점수, $c(\tau)$는 실행에 사용한 정책 토큰 수입니다. $S$는 기대 성능, $C$는 기대 비용이며, $\mathbb E$는 평균을 나타냅니다. **[p.3]** :chatgpt-content-reference{index="28"}

> **용어 풀이 — 실행 궤적·검증기:** 실행 궤적은 모델의 응답, 도구 호출, 도구 결과 등 한 문제를 푸는 동안 생긴 기록입니다. 검증기는 단위 테스트, 시뮬레이터 또는 채점 모델처럼 결과의 성공 여부나 품질을 판단하는 장치입니다.

실제로는 각 문제를 $k$번 실행해 추정합니다.

**[저자 보고: 원문 식 (3)]**

$$
\begin{aligned}
\widehat S(H)
&=
\frac{1}{k|\mathcal D_{\text{evolve}}|}
\sum_{x\in\mathcal D_{\text{evolve}}}
\sum_{j=1}^{k}
r\!\left(x,\tau_x^{(j)}\right),\\
\widehat C(H)
&=
\frac{1}{k|\mathcal D_{\text{evolve}}|}
\sum_{x\in\mathcal D_{\text{evolve}}}
\sum_{j=1}^{k}
c\!\left(\tau_x^{(j)}\right).
\end{aligned}
$$

**기호:** $\mathcal D_{\text{evolve}}$는 하네스 개선에 반복 사용하는 문제 집합, $|\mathcal D_{\text{evolve}}|$는 문제 수, $j$는 반복 실행 번호, $\tau_x^{(j)}$는 문제 $x$의 $j$번째 실행입니다. 모자 기호 $\widehat{S}$는 유한한 실행으로 얻은 추정값을 뜻합니다. **[p.3]** :chatgpt-content-reference{index="29"}

**주의:** 이는 본문의 일반적 정의입니다. 실제 Harvey LAB은 문제별 점수의 단순 평균이 아니라 **전체 채점 기준의 통과 비율**로 집계하므로, 벤치마크별 집계 방식은 Appendix A를 함께 봐야 합니다. **[Appendix A.3, p.17]** :chatgpt-content-reference{index="30"}

### 3.3 제안 단계: 한 번에 얼마나, 무엇을 바꿀 것인가?

#### A. 시간이 지날수록 수정량을 줄인다

**[저자 보고: 원문 식 (4), (9)]**

```math
b_t=
\left\lceil
b_{\min}
+
\frac{b_{\max}-b_{\min}}{2}
\left(
1+\cos\left(\frac{\pi t}{T}\right)
\right)
\right\rceil,
\qquad
\|z_t\|_0\le b_t.
```

**기호:** $t$는 현재 라운드, $T$는 전체 라운드 수, $b_t$는 후보 하나에 포함할 수 있는 수정 수의 상한, $b_{\max}$와 $b_{\min}$은 초기·최소 예산입니다. $\lceil\cdot\rceil$는 올림이며, $z_t$는 각 수정의 포함 여부를 나타내는 이진 벡터이고, $\|z_t\|_0$는 포함된 수정의 개수입니다. 이 식의 $\pi$는 앞서 정책을 나타낸 기호와 달리 **원주율**입니다. **[p.5; Appendix C.2, p.19]** :chatgpt-content-reference{index="31"} :chatgpt-content-reference{index="32"}

> **용어 풀이 — 어닐링·희소성·원자적 수정:** 어닐링은 탐색 강도를 점차 줄이는 일정입니다. 희소한 수정은 한 번에 적은 수의 사항만 바꾸는 것이며, 원자적 수정은 효과를 개별적으로 추적하려는 하나의 변경 단위입니다.

**[검토·해석]** 여러 사항을 동시에 바꾸면 어떤 수정이 효과를 냈는지 알기 어렵습니다. 수정 수를 줄이는 것은 원인 추적을 쉽게 할 수 있지만, 반드시 여러 수정이 함께 있어야 작동하는 개선을 놓칠 위험도 있습니다. 또한 수식의 마지막 라운드 처리에는 문서상 불일치가 있으며, 이는 6.2절에서 따로 설명합니다.

#### B. 성공뿐 아니라 실패한 수정도 기억한다

**[저자 보고]** 각 수정에 대해 라운드, 구성요소, 검증하려는 가설, 코드 차이, 점수 변화, 비용 변화, 최종 채택 여부를 기록합니다. 단순히 마지막 성공 사례만 참고하지 않고, 이전에 실패한 가설도 다음 제안에 반영합니다. **[§3.2, p.5; 식 (10), p.20]** :chatgpt-content-reference{index="33"} :chatgpt-content-reference{index="34"}

> **용어 풀이 — 기여도 배분:** 성능 변화가 어떤 수정에서 비롯되었는지 연결하는 작업입니다.

**중요한 제한:** 하나의 후보에 여러 수정이 포함되면 **모든 수정에 동일한 후보 단위 점수·비용 변화가 기록**됩니다. 따라서 이 기록은 인과적으로 분리된 기여도가 아니라, 해당 후보와 함께 관찰된 결과입니다. **[Appendix C.2, p.20]** :chatgpt-content-reference{index="35"}

#### C. 정체되면 아직 건드리지 않은 구성요소를 탐색한다

**[저자 보고: 원문 식 (13)]**

$$
\sigma_t=
\mathbf 1
\left[
\widehat S_t-\widehat S_{t-w}\le\delta
\right],
\qquad
\mathcal U_t=\mathcal K\setminus\mathcal T_t.
$$

**기호:** $\mathbf 1[\cdot]$은 조건이 참이면 1, 거짓이면 0인 표시 함수입니다. $\sigma_t$는 정체 여부, $\widehat S_t$는 현재 점수, $w$는 정체 판정 구간, $\delta$는 경험적 잡음 허용폭입니다. $\mathcal K$는 수정 가능한 구성요소 유형의 집합, $\mathcal T_t$는 이미 평가한 수정이 있는 유형, $\mathcal U_t$는 아직 탐색하지 않은 유형입니다. 정체 시 $m_{\text{draft}}$개의 후보 슬롯을 탐색용으로 확보합니다. **[pp.20–21]** :chatgpt-content-reference{index="36"}

구성요소 유형은 프롬프트, 제어 흐름, 설정, 출력 연결, 문맥 관리, 도구, 스킬, 메모리, 하위 에이전트입니다. **[식 (12), p.20]** :chatgpt-content-reference{index="37"}

#### D. 최근 도움이 되지 않은 구조는 제거 후보로 만든다

**[저자 보고: 원문 식 (11), (14)]**

```math
\begin{aligned}
g_t(\ell)
&=
\max
\left\{
\Delta S_i:
\ell_i=\ell,\;
t-t_i\le n_{\text{prune}}
\right\},\\
\mathcal B_t
&=
\left\{
\ell\in\mathcal T_t:
g_t(\ell)\le 0
\right\},
\qquad
\max\varnothing=-\infty.
\end{aligned}
```

**기호:** $\ell$은 구성요소 유형, $i$는 과거 수정 기록, $\ell_i$와 $t_i$는 해당 기록의 유형과 라운드, $\Delta S_i$는 관찰한 점수 변화입니다. $n_{\text{prune}}$는 최근 이력을 보는 범위, $g_t(\ell)$은 그 범위에서 해당 유형과 연관된 최대 개선량, $\mathcal B_t$는 제거 대상으로 제안할 유형의 집합입니다. $\varnothing$는 관련 기록이 없는 경우입니다. **[pp.20–21]** :chatgpt-content-reference{index="38"} :chatgpt-content-reference{index="39"}

> **용어 풀이 — 가지치기:** 더 이상 유용하다는 증거가 없는 기능이나 구조를 제거하는 것입니다.

**[검토·해석]** 실제 기능을 하나씩 꺼서 효용을 측정하는 방식은 아닙니다. 특히 여러 수정의 결과를 함께 기록하므로, 구성요소 간 상호작용이나 장기적으로 필요한 기능을 잘못 평가할 가능성은 남습니다.

### 3.4 선택 단계: 무엇이 영구적인 변경으로 남을 수 있는가?

#### A. 평가하기 전에 벤치마크 특화 내용을 걸러낸다

**[저자 보고]** 누출 검사자는 후보의 코드 차이를 읽고 문제명, 고유명사, 문제별 수치, 답, 진화용 벤치마크에만 통하는 로직 등을 포함한 수정을 거부합니다. 평가 전에 검사하므로, 이런 후보가 높은 점수를 받아 후속 탐색의 좋은 사례로 남는 것을 방지하려는 설계입니다. **[§3.3, p.5]** :chatgpt-content-reference{index="40"}

> **용어 풀이 — 누출:** 일반적인 해결 능력을 개선하는 대신, 평가 문제의 특정 정보나 정답에 접근해 점수를 올리는 현상입니다.

**[검토·해석]** 명시적인 문제명이나 답을 막는 것과, 의미를 바꿔 숨긴 벤치마크 특화 규칙까지 제거하는 것은 다른 문제입니다. 논문은 검사자의 탐지율·오탐률을 보고하지 않습니다.

#### B. 과거 최고 성적에서 지나치게 내려가는 후보를 막는다

**[저자 보고: 원문 식 (5)]**

$$
\widehat S(H')\ge S^\star-\delta.
$$

**기호:** $H'$는 후보 하네스, $\widehat S(H')$는 후보의 측정 점수, $S^\star$는 최고로 기록한 점수, $\delta$는 초기 하네스를 반복 평가해 설정한 잡음 허용폭입니다. **[p.6]** :chatgpt-content-reference{index="41"}

**핵심 해석:** 이 식은 **“통계적으로 유의한 향상만 채택한다”는 조건이 아닙니다.** 최고 성적보다 $\delta$만큼 낮은 후보도 통과할 수 있으며, 작은 하락이 누적되어 계속 성능이 내려가는 것을 막는 하한에 가깝습니다.

#### C. 성능 상승에 비해 비용이 너무 많이 늘지 않는지 확인한다

먼저 변화량을 정의합니다.

**[저자 보고: 원문 식 (6)]**

$$
\Delta S=
\widehat S(H')-\widehat S(H_t),
\qquad
\Delta C=
\frac{
\widehat C(H')-\widehat C(H_t)
}{
\widehat C(H_t)
}.
$$

**기호:** $H_t$는 현재 하네스, $\Delta S$는 점수의 절대 변화, $\Delta C$는 정책 토큰 비용의 상대 변화입니다. 논문의 수식에서 점수는 0–1 범위이므로, 2점 개선은 $\Delta S=0.02$입니다. **[p.6; Appendix D.1, p.22]** :chatgpt-content-reference{index="42"} :chatgpt-content-reference{index="43"}

잡음 허용폭을 넘는 개선에는 다음 조건을 적용합니다.

**[저자 보고: 원문 식 (7)]**

$$
\Delta S>\delta
\quad\Longrightarrow\quad
\Delta C\le\beta_0+\beta_1\Delta S.
$$

**기호:** $\beta_0$는 기본적으로 허용하는 비용 증가량, $\beta_1$은 성능 개선에 따라 추가 비용을 얼마나 허용할지 결정하는 계수입니다. 두 값은 진화용 환경에서 정하고 전이 평가에서는 고정합니다. **[p.6]** :chatgpt-content-reference{index="44"}

**[검토·해석]** 이 조건은 비용 증가를 금지하지 않습니다. 예를 들어 코딩 설정의 $\beta_0=0.10$, $\beta_1=44.5$를 적용하면 2점 개선에 허용되는 상대 비용 증가는 $0.10+44.5\times0.02=0.99$, 즉 **99%**입니다. 따라서 “비용을 매우 엄격하게 제한한다”기보다 **성능과 비용의 교환관계를 명시한다**는 해석이 정확합니다. **[Table 5, p.23; 계산은 검토자]** :chatgpt-content-reference{index="45"}

#### D. 잡음 범위 안에서는 비용 절감과 구조적 새로움도 고려한다

구조적 새로움은 다음과 같이 정의됩니다.

**[저자 보고: 원문 식 (16)]**

```math
\nu_t(H')
=
\sum_{\ell\in\mathcal K_{\text{str}}}
\mathbf 1
\left[
\ell\in\text{comp}(H')
\;\land\;
N_t(\ell)=0
\right].
```

**기호:** $\mathcal K_{\text{str}}$는 도구·스킬·메모리·하위 에이전트의 네 유형, $\text{comp}(H')$는 후보가 수정한 유형의 집합, $N_t(\ell)$은 이전에 채택된 수정 중 해당 유형의 기록 수입니다. $\nu_t(H')$는 이전 승리 후보에 등장하지 않았던 구조적 유형을 이번 후보가 몇 종류 건드렸는지 셉니다. **[p.21]** :chatgpt-content-reference{index="46"}

이는 새 코드의 양이나 새로운 기능의 유용성을 직접 측정하는 값이 아니라, **구성요소 유형 수준의 새로움**입니다.

**[저자 보고: 원문 식 (17)]**

$$
\Delta S\le\delta
\quad\Longrightarrow\quad
w_s\Delta S-w_c\Delta C+w_n\nu_t(H')>0.
$$

**기호:** $w_s,w_c,w_n\ge0$는 각각 점수 변화, 비용 변화, 구조적 새로움에 부여하는 가중치입니다. $\Delta C<0$이면 비용이 줄어 양의 기여를 합니다. 코딩은 $w_s=0$이므로 작은 점수 상승만으로는 채택되지 않지만, 업무·공학 설정은 양의 $w_s$를 사용합니다. **[p.22]** :chatgpt-content-reference{index="47"}

**중요한 재현성 문제:** 본문은 이 가중치들이 Table 5에 있다고 설명하지만, 실제 Table 5에는 없습니다. 이는 아래 6.2절에서 다룹니다.

#### E. 모든 조건을 통과한 후보 중 점수가 가장 높은 것을 고른다

**[저자 보고: 원문 식 (8)의 선택 규칙을 조건별로 풀어 쓴 표현]**

$$
\begin{aligned}
\mathcal H_t
&\sim
P_{\text{reg}}
\left(
\cdot\mid
H_t,\mathcal F_t,\mathcal L_t,b_t,\mathcal E_t,\mathcal B_t
\right)
\subseteq\Omega(H_t),\\[2mm]
H_{t+1}
&=
\begin{cases}
\displaystyle
\underset{H'\in\mathcal H_t\cap\mathcal A_t}
{\text{arg max}}
\;\widehat S(H'),
&
\mathcal H_t\cap\mathcal A_t\neq\varnothing,\\
H_t,
&
\mathcal H_t\cap\mathcal A_t=\varnothing.
\end{cases}
\end{aligned}
$$

**기호:** $\mathcal H_t$는 후보 집합, $P_{\text{reg}}$는 정규화된 제안 과정, $\mathcal F_t$는 현재 피드백, $\mathcal L_t$는 수정 이력, $\mathcal E_t$는 탐색 지시, $\mathcal B_t$는 가지치기 대상입니다. $\Omega(H_t)$는 소스 수정을 통해 도달할 수 있는 하네스 공간, $\mathcal A_t$는 채택 조건을 만족하는 후보 집합이며, $\text{arg max}$는 점수를 가장 크게 만드는 후보를 선택한다는 뜻입니다. **[Appendix C.1, p.19; Algorithm 2, p.20]** :chatgpt-content-reference{index="48"} :chatgpt-content-reference{index="49"}

공학 설계에는 추가로 유효 출력 비율이 3%포인트 넘게 감소하거나, 미제출 비율이 2%포인트 넘게 증가하는 후보를 거부하는 조건이 있습니다. **[Appendix C.3, p.22]** :chatgpt-content-reference{index="50"}

### 3.5 실제 설정과 정규화의 의미

| 설정 | 코딩 | 업무 에이전트 | 공학 설계 |
|---|---:|---:|---:|
| 진화 라운드 $T$ | 20 | 20 | 40 |
| 문제당 평가 반복 $k$ | 2 | 2 | 4 |
| 잡음 허용폭 $\delta$ | 0.017 | 0.004 | 0.020 |
| $b_{\max}$ / $b_{\min}$ | 4 / 1 | 3 / 1 | 4 / 1 |
| 정체 구간 $w$ | 3 | 3 | 3 |
| 탐색용 후보 수 $m_{\text{draft}}$ | 1 | 1 | 1 |
| 가지치기 구간 $n_{\text{prune}}$ | 4 | 4 | 5 |
| $\beta_0$ / $\beta_1$ | 0.10 / 44.5 | 0.10 / 35.4 | 0.15 / 24.4 |

**[저자 보고: Table 5, p.23]** :chatgpt-content-reference{index="51"}

**반드시 구분할 점:** 저자가 말하는 $L_0$ , Lasso/ $L_1$ , Ridge/ $L_2$는 대부분 **기능적 유사성에 대한 비유**입니다. 고정된 파라미터 벡터에 해당 노름의 벌점을 더해 최적화한 것이 아니며, 비용 제약도 제곱 노름 벌점이 아닙니다. **[§3.1, p.4; Appendix C, p.19; C.3, p.21]** :chatgpt-content-reference{index="52"} :chatgpt-content-reference{index="53"} :chatgpt-content-reference{index="54"}

> **용어 풀이 — $L_0$, $L_1$, $L_2$:** 보통 $L_0$는 활성 항목의 개수, $L_1$은 절댓값의 합, $L_2$는 제곱합에 기반한 크기와 관련됩니다. 이 논문에서는 이를 각각 수정 수 제한, 구조 제거, 전체 자원 증가 억제와 연결하지만, 고전적 정규화의 수학적 보장이 그대로 따라오는 것은 아닙니다.

---

## 4. 결과: 무엇이 얼마나 개선되었으며, 어떤 일반화를 보여주는가?

### 4.1 주 실험 결과

아래는 **Claude Opus 4.8을 사용하는 주 실험**입니다. 숫자는 모두 0–100 표기로 제시되지만 **서로 같은 의미의 지표는 아닙니다.**

| 벤치마크·분할 | 지표의 의미 | 초기 $H_0$ | RRSI | 절대 변화 |
|---|---|---:|---:|---:|
| Terminal-Bench 2.1 — 진화용 | 단위 테스트 기반 성공률 | 74.2 | 80.2 | +6.0점 |
| SWE-bench Verified — OOD | 이슈 해결률 | 82.0 | 83.8 | +1.8점 |
| Harvey LAB — 진화용 | 채점 기준 통과 비율 | 89.4 | 90.5 | +1.1점 |
| Harvey LAB — ID 보류 | 채점 기준 통과 비율 | 86.9 | 89.2 | +2.3점 |
| JobBench — OOD | 가중 루브릭 점수 | 36.0 | 40.7 | +4.7점 |
| GDPval — OOD | 인간 전문가 결과물 대비 승률 | 48.8 | 52.3 | +3.5점 |
| APEX-Agents — OOD | 단일 실행의 과제 성공률 | 34.2 | 37.9 | +3.7점 |
| EngDesign — 진화용 | 시뮬레이터·테스트 기반 통과율 | 50.0 | 54.9 | +4.9점 |
| Frontier-Eng — OOD | Medal Score | 17.7 | 22.0 | +4.3점 |

**수치:** Figure 3, p.8. **평가 정의:** Appendix A, pp.17–18. :chatgpt-content-reference{index="55"} :chatgpt-content-reference{index="56"} :chatgpt-content-reference{index="57"}

> **용어 풀이 — ID 보류·루브릭·pass@1:** ID 보류는 같은 데이터 분포에서 가져왔지만 진화에는 사용하지 않은 문제입니다. 루브릭은 결과물을 평가하는 세부 기준표이며, pass@1은 여러 답 중 가장 좋은 것을 고르는 대신 한 번 생성한 결과가 성공하는 비율입니다.

> **용어 풀이 — Medal Score:** Frontier-Eng에서는 고정된 기준 결과에 도달한 정도에 따라 1, 0.67, 0.33의 점수를 부여해 평균합니다. 따라서 22.0이라는 수치를 “문제의 22%를 완전히 해결했다”로 읽으면 안 됩니다. **[Appendix A.8, p.18]** :chatgpt-content-reference{index="58"}

**[검토·해석]** 가장 중요한 결과는 진화용 점수의 큰 상승이 아니라, **평가 방식이 다른 여러 OOD 환경에서 초기 하네스보다 개선된 점수가 관찰되었다는 것**입니다. 특히 공학 설계는 모델 채점기가 아니라 고정된 시뮬레이터로 평가하므로, 모든 개선이 채점 모델의 문체 선호를 공략한 결과라는 설명은 약해집니다. 다만 시뮬레이터가 결정적이라는 사실이 실행 모델의 확률적 변동까지 없애지는 않습니다. **[p.8; Appendix A.7–A.8, p.18]** :chatgpt-content-reference{index="59"} :chatgpt-content-reference{index="60"}

### 4.2 기존 방법과의 비교에서 실제로 말할 수 있는 것

업무 에이전트에서 Meta-Harness는 진화용 성적이 **93.0**으로 RRSI의 **90.5**보다 높지만, 세 OOD 벤치마크에서는 RRSI가 각각 더 높은 점수를 기록합니다. 그러나 Harvey LAB의 ID 보류 성적은 둘 다 **89.2**이므로, RRSI가 모든 보류 평가에서 단독으로 더 좋았다고 표현하면 부정확합니다. **[Table 1, p.8]** :chatgpt-content-reference{index="61"}

또한 Table 1은 업무 에이전트의 개별 비교값을 제공하지만, Figure 1의 코딩·공학 비교는 기존 방법의 **평균값**을 보여줍니다. 따라서 그 그림만으로 각 개별 기준선에 대한 차이까지 복원할 수는 없습니다. **[Figure 1, p.2]** :chatgpt-content-reference{index="62"}

### 4.3 정규화 제거 실험

| 변형 | 진화용 점수 | ID 보류 | OOD 평균 | 정책 토큰/실행 |
|---|---:|---:|---:|---:|
| 초기 하네스 | 89.4 | 86.9 | 39.7 | 1.56백만 |
| 비정규화 진화 | 92.8 | 88.9 | 40.3 | 3.80백만 |
| 제안 측 정규화 제거 | 90.7 | 88.8 | 41.9 | 2.69백만 |
| 선택 측 정규화 제거 | 91.5 | 88.7 | 41.0 | 3.59백만 |
| **RRSI** | **90.5** | **89.2** | **43.6** | **2.42백만** |

**[저자 보고: Table 2, p.9]** OOD 평균은 JobBench, GDPval, APEX-Agents의 평균입니다. :chatgpt-content-reference{index="63"}

> **용어 풀이 — 제거 실험:** 시스템의 일부를 빼고 성능을 비교해 해당 부분의 기여를 조사하는 실험입니다.

**[검토·해석]** 이 결과는 “진화용 성적을 조금 덜 올리더라도 전이와 효율이 좋아질 수 있다”는 논문의 주장을 잘 뒷받침합니다. 그러나 이 실험은 **제안 측 전체와 선택 측 전체**를 비교하므로, 누출 검사·성능 하한·비용 조건·가지치기 중 어느 것이 얼마나 중요한지는 별도로 알 수 없습니다.

### 4.4 모델 일반화: 세 종류의 증거를 구분해야 한다

| 실험 | 저자 보고 | 입증 범위 |
|---|---|---|
| Opus로 독립 진화 | Terminal-Bench 74.2→80.2, SWE-bench 82.0→83.8 | 같은 실행 모델에서 새로운 벤치마크로의 전이 |
| Gemini 3.5 Flash로 독립 진화 | Terminal-Bench 64.6→78.7, SWE-bench 76.8→79.0 | 다른 모델 계열에서도 RRSI 절차가 작동하는 사례 |
| Gemini 3.5 Flash에서 진화한 하네스를 Flash Lite에 그대로 적용 | Terminal-Bench 11.2→14.6 | 동일한 수정 결과물이 새로운 모델에서도 도움이 된 사례 |

**[Tables 3–4, pp.9–10]** :chatgpt-content-reference{index="64"} :chatgpt-content-reference{index="65"}

**[검토·해석]** 여기에서 아직 입증되지 않은 것은 다음과 같습니다.

- 하나의 하네스를 코딩에서 법률 업무나 공학 설계로 그대로 옮기는 **도메인 간 전이**.
- 새로운 모델과 새로운 벤치마크가 동시에 등장하는 **결합된 분포 변화**.
- 다양한 모델에서 동일한 하네스가 가장 좋다는 **보편적 최적성**.

이 구분은 중요합니다. 이 논문이 보여준 것은 **고정된 모델을 더 잘 활용하는 실행 구조의 일반화**이며, 모델 내부 표현이나 가중치 자체의 일반화 능력이 학습으로 개선되었다는 결과는 아닙니다.

---

## 5. 가장 중요한 그림의 선정과 해석

| 그림 | 무엇을 보여주는가? | 어떻게 읽어야 하는가? |
|---|---|---|
| **Figure 2, p.4 — 방법 이해에 가장 중요** | 후보 제안과 후보 선택 양쪽에 정규화를 배치한다. | 단순히 프롬프트를 짧게 만드는 방법이 아니라, **탐색이 어디로 이동하고 어떤 수정이 남는지**를 제어하는 시스템이다. 그림의 $L_1/L_0$ 명칭은 본문 설명과 일부 다르므로 아래 문서 불일치 항목을 함께 봐야 한다. :chatgpt-content-reference{index="66"} |
| **Figure 3, p.8 — 성과 판단에 가장 중요** | 진화용·ID·OOD 성적을 초기 하네스와 나란히 비교한다. | 8개 벤치마크에 Harvey LAB의 두 분할이 포함되어 9쌍의 막대가 나온다. 핵심은 6개 미사용 분할의 개선이며, 막대 높이만으로 서로 다른 지표의 개선 크기를 비교해서는 안 된다. 오차막대도 제시되지 않는다. :chatgpt-content-reference{index="67"} |
| **Figure 4, p.10 — 비용 주장 검증에 가장 중요** | 토큰 비용과 OOD 평균, 실행 단계 수를 비교한다. | RRSI는 비교한 진화 방법들보다 적은 토큰으로 높은 OOD 평균을 보이지만 초기 하네스보다 비싸다. 또한 가로축 비용은 **진화용 분할에서 측정한 비용**, 세로축은 **OOD 점수**이므로 OOD 환경 자체의 비용 우위까지 입증한 그림은 아니다. :chatgpt-content-reference{index="68"} |
| **Figure 1, p.2 — 문제의식 요약에 중요** | 진화용 상대 개선율과 OOD 상대 개선율의 관계를 보여준다. | 진화용 점수가 많이 오르는 방법이 OOD에서도 좋은 것은 아니라는 메시지이다. 두 축은 점수 차이가 아니라 **상대 개선율**이므로, 기준 성적의 차이와 천장 효과를 고려해야 한다. :chatgpt-content-reference{index="69"} |

> **용어 풀이 — 천장 효과:** 원래 점수가 이미 높으면 더 올라갈 공간이 작아지는 현상입니다. 예를 들어 89점에서의 1점 상승과 36점에서의 4점 상승을 그대로 학습 효과의 강도로 비교하기는 어렵습니다.

그림을 보완하는 가장 이해하기 쉬운 사례는 **Table 6, p.24**입니다. 공학 설계에서 반복되는 작업 디렉터리 오류에 대한 복구 안내를 추가하자 통과 횟수가 **122/244에서 128/244**로 증가하고 토큰은 **1.6%** 늘어 채택되었습니다. 반대로 비용을 **13.6%** 줄인 코딩 수정도 성능 하한을 위반해 거부되었습니다. 이는 점수 또는 비용 하나만으로 판단하지 않는다는 구체적 사례이지만, 선택된 사례 자체가 전체 효과의 인과적 증명은 아닙니다. :chatgpt-content-reference{index="70"}

---

## 6. 통계적으로 취약한 부분과 직접 비교할 수 없는 수치

### 6.1 통계·평가 설계상의 주요 한계

| 표시 | 취약점 | 왜 중요한가? |
|---|---|---|
| **[통계적 불확실성 미보고]** | 주요 표는 점추정값을 제공하며, 독립적인 전체 진화 실험의 반복수와 최종 차이의 신뢰구간을 제시하지 않는다. **Tables 1–5.** :chatgpt-content-reference{index="71"} :chatgpt-content-reference{index="72"} :chatgpt-content-reference{index="73"} | $k=2$ 또는 $4$는 문제별 실행 반복수이지, 하네스 진화 전체를 다른 난수 조건으로 반복했다는 뜻이 아니다. 작은 점수 차이의 안정성을 판정하기 어렵다. |
| **[독립 표본 수 주의]** | Harvey LAB은 다수의 채점 기준을 합산하지만 같은 문제의 기준들은 동일한 결과물에 의존한다. **Appendix A.3, p.17.** :chatgpt-content-reference{index="74"} | 약 1만 4천 개 기준 판정을 1만 4천 개의 완전히 독립적인 문제처럼 취급해서는 안 된다. 문제 단위의 상관을 고려해야 한다. |
| **[유의성 검정 아님]** | $\delta$는 초기 하네스 반복 평가에서 정한 경험적 허용폭이다. **pp.6, 22.** :chatgpt-content-reference{index="75"} :chatgpt-content-reference{index="76"} | 후보를 반복 탐색하면서 생기는 선택 편향이나 다중 비교 오류를 통제한다는 수학적 보장은 제시되지 않는다. |
| **[평가자 편향 잔존]** | Harvey·JobBench·APEX 등은 모델 채점을 사용한다. GDPval은 여러 모델과 양방향 제시를 사용한다. **Appendix A, pp.17–18.** :chatgpt-content-reference{index="77"} | 여러 채점 모델과 순서 교환은 도움이 되지만, 인간 평가와의 일치도나 채점 모델 교체에 대한 강건성을 대신하지는 못한다. |
| **[인과적 기여 미분리]** | 여러 수정을 묶은 후보의 개선량을 각 수정에 동일하게 기록한다. **p.20.** :chatgpt-content-reference{index="78"} | “이 구성요소가 개선의 원인”이라는 결론은 별도 제거·복원 실험 없이는 강하게 내리기 어렵다. |
| **[계산량 통제 불완전]** | 기준선은 초기 하네스·정책·진화용 데이터·후보 예산을 공유한다. **p.7.** :chatgpt-content-reference{index="79"} | 같은 후보 수는 같은 총 토큰, 실행 시간, 검증기 호출, 금전 비용을 뜻하지 않는다. |
| **[전체 비용 아님]** | 비용 표와 그림은 최종 하네스의 정책 토큰을 보고한다. **Table 2; Figure 4.** :chatgpt-content-reference{index="80"} :chatgpt-content-reference{index="81"} | 제안자·분석자·검사자와 탈락 후보 평가까지 포함한 진화 비용, 실제 배포에서의 비용 회수 시점은 알 수 없다. |
| **[일반화 범위 제한]** | 도메인별로 별도 진화하며 최대 20·40라운드를 실험한다. **pp.2, 23.** :chatgpt-content-reference{index="82"} :chatgpt-content-reference{index="83"} | 수백·수천 회 자기개선, 새로운 도구 생태계, 새로운 언어·조직으로의 장기 일반화는 별도 질문이다. |

> **용어 풀이 — 점추정값·신뢰구간·선택 편향:** 점추정값은 “43.6점”처럼 하나로 요약한 결과입니다. 신뢰구간은 추정의 불확실성을 나타내는 범위이고, 선택 편향은 여러 후보 중 우연히 높은 점수를 받은 후보를 선택하면서 실제보다 좋아 보이는 현상입니다.

### 6.2 문서 내부의 불일치와 재현성 확인 사항

| 항목 | 확인된 내용 | 판단 |
|---|---|---|
| **식 (17)의 가중치 누락** | p.22는 $w_s,w_c,w_n$을 Table 5에 보고한다고 하지만, p.23의 표에는 해당 행이 없다. :chatgpt-content-reference{index="84"} :chatgpt-content-reference{index="85"} | **논문만으로 해당 채택 규칙을 완전히 재구성하기 어렵다.** 코딩의 $w_s=0$은 본문에서 확인되지만 나머지 정확한 값은 표에 없다. |
| **마지막 수정 예산의 경계값** | 라운드는 $t=0,\dots,T-1$이고 식 (4)는 올림을 사용한다. 코딩 설정을 그대로 대입하면 마지막 $b_{19}=2$이며, Table 5의 “final-round edit budget 1” 설명과 다르다. :chatgpt-content-reference{index="86"} :chatgpt-content-reference{index="87"} :chatgpt-content-reference{index="88"} | **문서 수식 기준의 불일치**이다. 실제 코드가 어느 인덱스·반올림 규칙을 사용하는지 확인해야 한다. |
| **GDPval 평가 횟수** | 185개 문제를 평가하고 각 쌍을 두 제시 순서로 채점한다고 하면서, 심판당 비교 횟수를 204회로 적는다. **pp.17–18.** :chatgpt-content-reference{index="89"} | 서술대로라면 $185\times2=370$회이다. 204회의 근거나 별도 부분집합 사용 여부가 설명되지 않는다. |
| **정규화 명칭 불일치** | Figure 2는 비용 수용을 $L_1$-style, 구조 가지치기를 $L_0$-style로 표기한다. 본문 §3.1은 구조 가지치기를 $L_1$, 비용 수용을 $L_2$ 비유로 설명한다. **p.4.** :chatgpt-content-reference{index="90"} | 명칭보다 실제 수식과 알고리즘을 기준으로 이해해야 한다. |
| **‘30% 적은 토큰’의 정확한 기준** | 초록은 30% 감소라고 요약하지만 Table 2의 3.80→2.42는 약 36.3% 감소이다. 공식 프로젝트 페이지는 약 36%로 제시한다. :chatgpt-content-reference{index="91"} :chatgpt-content-reference{index="92"} :chatgpt-content-reference{index="93"} | 초록의 대략적 표현과 표에서 계산한 정확한 비율을 구분하는 것이 적절하다. |

마지막 수정 예산에 대한 **검토 계산**은 다음과 같습니다.

```math
b_{19}
=
\left\lceil
1+\frac{3}{2}
\left(
1+\cos\left(\frac{19\pi}{20}\right)
\right)
\right\rceil
=
2.
```

**기호:** 앞의 식 (4)에 $T=20$, $t=19$, $b_{\min}=1$, $b_{\max}=4$를 대입한 것입니다. 이는 **문서의 수식을 그대로 계산한 결과**이며, 실제 공개 코드의 동작을 확인했다는 뜻은 아닙니다.

이러한 사항은 실험 결과가 틀렸다는 증거와는 다릅니다. 다만 **논문의 설명만으로 실험을 재현하거나 결과의 정확한 의미를 확인할 때 해결해야 할 사항**입니다.

### 6.3 직접 비교하면 안 되는 수치

**첫째, 서로 다른 지표를 같은 정확도로 취급하면 안 됩니다.** JobBench의 가중 루브릭 점수, GDPval의 인간 대비 승률, APEX의 성공률은 의미가 다릅니다. 따라서 43.6이라는 OOD 평균은 저자가 정한 세 지표의 요약치이지, 전체 OOD 문제의 43.6%를 해결했다는 뜻이 아닙니다. **[Table 2; Appendix A.4–A.6]** :chatgpt-content-reference{index="94"} :chatgpt-content-reference{index="95"}

**둘째, 상대 개선율과 점수 차이를 혼동하면 안 됩니다.** Frontier-Eng의 17.7→22.0은 **4.3점 상승**이자 약 **24.3% 상대 상승**입니다. Flash Lite의 11.2→14.6 역시 **3.4점 상승**이지만 상대적으로 약 **30.4%**이므로, 큰 상대 수치에는 낮은 출발점의 영향이 있습니다. **[pp.7, 9–10]** :chatgpt-content-reference{index="96"} :chatgpt-content-reference{index="97"}

**셋째, Frontier-Eng의 외부 리더보드와 바로 비교하면 안 됩니다.** 논문은 진화용 데이터와 겹치는 EngDesign 도메인을 제외하고, 구축 불가능한 평가 환경에는 점수를 주지 않으며, 47개 중 38개 문제가 점수에 기여한다고 설명합니다. 외부 수치와 비교하려면 동일한 버전·제외 규칙·분모인지 확인해야 합니다. **[Appendix A.8, p.18]** :chatgpt-content-reference{index="98"}

---

## 7. 2020년 이후 관련 연구와의 비교

아래 표의 **기존 연구 결과는 각 연구 저자의 보고**이며, RRSI와의 연결은 검토자의 해석입니다. 서로 다른 연구의 절대 점수는 동일한 실험 조건으로 얻은 순위가 아닙니다.

### 7.1 연구 흐름과 RRSI의 위치

| 연구 | 방법·저자 보고 | RRSI와 비교한 의미 |
|---|---|---|
| **DomainBed, 2020 / WILDS, 2021** | DomainBed는 데이터·모델·선택 기준의 불일치가 일반화 방법 비교를 왜곡한다고 지적했고, WILDS는 실제 배포의 분포 변화를 평가 대상으로 구성했다. :chatgpt-content-reference{index="99"} | 하네스 연구는 아니지만, RRSI도 **선택에 사용한 데이터와 최종 평가를 구분하고 동일 조건으로 비교해야 한다**는 동일한 평가 원칙을 따른다. |
| **ReAct, 2022 / Reflexion, 2023** | ReAct는 추론·행동을 교차시키고, Reflexion은 가중치 대신 자연어 피드백과 기억을 이용해 후속 시도를 개선한다. :chatgpt-content-reference{index="100"} | 모델 가중치 밖에서 성능을 개선하는 선행 흐름이다. RRSI는 특정 실행 패턴이나 기억 사용에 그치지 않고, 주변 프로그램의 수정·선택 절차를 다룬다. |
| **DSPy, 2023** | 언어모델 호출을 모듈화된 프로그램으로 표현하고, 주어진 지표에 맞춰 파이프라인을 최적화한다. :chatgpt-content-reference{index="101"} | RRSI의 차이는 프로그램 최적화 자체보다 **반복 최적화 과정의 과적합·비용·채택 규칙을 전면에 놓는 것**이다. |
| **ADAS, 2024** | 메타 에이전트가 코드로 새로운 에이전트를 설계하고 이전 설계 기록을 활용한다. 원 논문은 도메인·모델 간 전이도 보고한다. :chatgpt-content-reference{index="102"} | 자동 설계와 전이 가능성은 RRSI 이전에도 연구되었다. RRSI의 기여를 “자동 에이전트 설계 또는 일반화의 최초 발견”으로 표현하면 안 된다. |
| **Darwin Gödel Machine, 2025** | 자체 코드를 수정하고 다양한 에이전트의 기록 집합을 유지하는 개방형 진화를 사용한다. 원 논문은 자체 평가에서 SWE-bench 20.0→50.0 등을 보고한다. :chatgpt-content-reference{index="103"} | 다양한 경로를 보존하는 탐색에 무게를 두는 반면, RRSI는 현재 후보가 남기 위한 규칙과 비용 제어에 무게를 둔다. 두 접근은 결합 가능한 방향이다. |
| **GEPA, 2025** | 실행 궤적을 자연어로 성찰하고, 파레토 기반 선택으로 상호 보완적인 프롬프트 개선을 결합한다. 원 논문은 GRPO 대비 최대 35배 적은 롤아웃 사용을 보고한다. :chatgpt-content-reference{index="104"} | RRSI보다 주된 최적화 대상이 프롬프트에 가깝다. 다만 이력·성찰·다양성을 활용한다는 공통점이 있어, 동일 예산에서의 비교가 유용하다. |
| **Meta-Harness, 2026** | 과거 후보의 코드·점수·실행 기록을 파일시스템으로 제공하며 하네스 코드를 탐색한다. 원 논문은 5개 보류 모델에 걸친 수학 문제 성능 전이도 보고한다. :chatgpt-content-reference{index="105"} | RRSI와 가장 직접적인 선행 방법 중 하나이다. RRSI의 업무 실험에서 전이가 작았다는 결과를 Meta-Harness 전체의 전이 실패로 일반화해서는 안 된다. |
| **AHE, 2026** | 구성요소·경험·결정의 관측 가능성을 높여 수정과 결과를 추적한다. 원 논문은 Terminal-Bench 69.7→77.0과 다른 모델·벤치마크로의 전이를 보고한다. :chatgpt-content-reference{index="106"} | RRSI의 이력 기반 설계와 상당한 공통점이 있다. RRSI Table 1의 업무 환경 결과와 AHE 원 논문의 코딩 결과는 서로 다른 실험이다. |
| **TTHE, 2026** | 정답 없이 테스트 입력의 실행 기록을 이용해 하네스를 온라인으로 수정하며, 선택된 프로그램을 후속 입력에 유지한다. :chatgpt-content-reference{index="107"} | 원래 목적이 **비지도 테스트 시점 적응**이므로, 점수 피드백을 쓰는 공통 진화 환경에서의 비교와 원래 배포 목적을 구분해야 한다. |
| **HarnessX, 2026** | 프롬프트·도구·메모리·제어 흐름을 유형화된 구성요소로 조합하고, 실행 기록을 하네스 개선 및 모델 학습 신호와 연결한다. :chatgpt-content-reference{index="108"} | RRSI의 고정 가중치 실험은 HarnessX의 전체 하네스–모델 개선 체계를 모두 평가한 것은 아니다. |
| **HarnessCompass, 2026** | 과제 비특화 수정 제약, 에이전트의 능동 피드백, 구성요소별 최적화를 결합한다. 원 논문은 GPT-5.4로 SWE-bench Verified 54→66을 5회 진화에서 보고한다. :chatgpt-content-reference{index="109"} | 일반화 지향이라는 점에서 특히 가까운 비교 대상이다. RRSI의 주 비교표에는 포함되지 않아 두 방법의 상대 우위는 알 수 없다. |
| **HarnessBank, 2026, v2** | 다양한 의미적 유형의 하네스를 보관·재조합하고 단계적으로 선별한다. 7개 벤치마크 개선과 함께, 범용 최적 하네스보다 모델별 진화 과정의 중요성을 보고한다. :chatgpt-content-reference{index="110"} | RRSI의 한 쌍의 교차 모델 전이를 모든 모델에 통하는 최적 하네스로 확대하지 않아야 함을 시사한다. |
| **Rethinking the Evaluation of Harness Evolution for Agents, 2026** | 동일한 피드백·추론 예산에서 단순 재시도와 하네스 진화를 비교하고, 진화 방법이 항상 더 좋지는 않다고 보고한다. :chatgpt-content-reference{index="111"} | RRSI의 후속 검증에서 가장 중요한 기준선은 다른 진화 방법뿐 아니라 **같은 비용으로 더 오래 실행하거나 여러 번 시도하는 초기 하네스**이다. |

> **용어 풀이 — 파레토 선택·기록 집합:** 파레토 선택은 한 지표만 가장 좋은 후보가 아니라, 다른 후보가 모든 지표에서 동시에 이기지는 못하는 후보들을 남기는 방식입니다. 기록 집합은 다양한 과거 후보를 보존해 나중에 다시 출발점이나 조합 재료로 사용하는 저장소입니다.

**버전 주의:** RRSI 참고문헌의 *Self-Evolving Agent Harnesses via Gated Semantic Quality-Diversity*는 arXiv:2607.13683의 초기 제목이며, 확인한 v2의 제목은 *HarnessBank: Semantic Gene-Bank Search with Gated Verification for Agent-Harness Self-Evolution*입니다. 두 버전의 제목과 결과를 섞어 인용하지 않는 것이 좋습니다. **[RRSI p.13; 외부 원문 v2]** :chatgpt-content-reference{index="112"} :chatgpt-content-reference{index="113"}

### 7.2 RRSI의 차별성은 어디에 있는가?

**[검토·해석]** RRSI의 가장 설득력 있는 차별성은 자동 코드 수정 그 자체가 아니라, **수정 수 제한·부정적 실험 이력·누출 검사·성능 하한·비용 조건·구조 제거를 하나의 진화 절차로 결합하고, 여러 OOD 환경에서 검증한 점**입니다. 반면 일반화 지향의 제약은 HarnessCompass와, 다양성과 검증 중심의 탐색은 HarnessBank·GEPA와 개념적으로 겹칩니다. 따라서 신규성은 완전히 독립적인 원리의 발명보다 **구체적인 규칙의 결합과 평가 설계**에서 찾는 편이 타당합니다. **[RRSI §3, pp.4–6; §5, p.11]** :chatgpt-content-reference{index="114"} :chatgpt-content-reference{index="115"} :chatgpt-content-reference{index="116"}

**[직접 비교 불가]** RRSI의 SWE-bench 83.8, HarnessCompass의 66, DGM의 50을 그대로 순위로 나열해서는 안 됩니다. 모델, 초기 하네스, 평가 분할, 검색 예산과 피드백 접근 조건이 다르기 때문입니다. 또한 GEPA의 “롤아웃 수 절감”과 RRSI의 “최종 하네스 정책 토큰 절감”은 단위와 측정 대상이 달라 직접 비교할 수 없습니다. :chatgpt-content-reference{index="117"} :chatgpt-content-reference{index="118"} :chatgpt-content-reference{index="119"}

### 7.3 앞으로의 연구에 미칠 수 있는 영향

이 연구가 제시하는 중요한 방향은 **하네스 진화의 평가 단위를 바꾸는 것**입니다. 가장 높은 진화용 점수 대신, **보류 환경에서 남는 개선량, 추가 추론 비용, 전체 탐색 비용, 실패한 전이의 비율**을 함께 보고하도록 연구 관행을 발전시킬 수 있습니다. 이는 RRSI의 결론과 최근 평가 비판 연구가 만나는 지점입니다. **[RRSI §6, p.11]** :chatgpt-content-reference{index="120"} :chatgpt-content-reference{index="121"}

---

## 8. 문서가 답하지 않는 질문

| 아직 답하지 않은 질문 | 현재 증거의 경계 |
|---|---|
| **어떤 수정이 OOD 개선의 실제 원인인가?** | 묶음 제거 실험과 대표 사례는 있지만, 최종 수정 각각의 제거·복원에 따른 OOD 변화는 제시되지 않는다. **Table 2; Table 6.** :chatgpt-content-reference{index="122"} :chatgpt-content-reference{index="123"} |
| **하나의 하네스가 새 모델·새 도메인·새 도구를 동시에 견디는가?** | 도메인별로 별도 진화하며, 교차 모델 실험은 진화에 사용한 Terminal-Bench에서 수행된다. **p.2; Table 4.** :chatgpt-content-reference{index="124"} :chatgpt-content-reference{index="125"} |
| **‘독립적인 수정 하나’는 어떻게 판정하고 검증하는가?** | 이진 수정 표시와 유형 태그는 정의하지만, 큰 의미적 변경을 한 수정으로 묶는 것을 어떻게 방지하는지는 충분히 설명하지 않는다. **Appendix C.2, pp.19–20.** :chatgpt-content-reference{index="126"} :chatgpt-content-reference{index="127"} |
| **누출 검사자는 교묘하게 표현된 벤치마크 특화 규칙을 얼마나 잘 찾는가?** | 검사 대상은 설명하지만 탐지율·오탐률 평가가 없다. **§3.3, p.5.** :chatgpt-content-reference{index="128"} |
| **장기간 진화하면 성능과 구조가 안정되는가?** | 실험은 20·40라운드이며, 더 긴 자기개선은 저자도 추가 검증이 필요하다고 인정한다. **Table 5; Limitations.** :chatgpt-content-reference{index="129"} :chatgpt-content-reference{index="130"} |
| **총비용을 포함해도 단순 재시도보다 유리한가?** | 최종 실행 비용은 제공하지만, 진화 전체 비용을 포함한 손익분기점과 동일 총예산 재시도 비교는 제시하지 않는다. **Table 2; Figure 4.** :chatgpt-content-reference{index="131"} :chatgpt-content-reference{index="132"} |
| **새 모델이 기존 기능을 잘못 활용하거나 새로운 실패 유형을 만들면 어떻게 되는가?** | 교차 모델 전이의 긍정적 사례는 있지만, 모델 교체에 따른 실패 유형 변화와 하네스 재적응 조건은 체계적으로 분석하지 않는다. **Tables 3–4.** :chatgpt-content-reference{index="133"} |
| **가중치 학습과 하네스 진화를 함께 하면 일반화가 더 좋아지는가?** | 가중치가 바뀌는 설정은 연구 범위 밖이라고 명시한다. **Limitations, p.11.** :chatgpt-content-reference{index="134"} |

---

## 9. 결론과 후속 연구 방향

### 9.1 저자가 제시한 시사점과 남겨둔 과제

**[저자 보고]** 핵심 시사점은 자기개선에서 **무엇을 바꿀 수 있는가뿐 아니라, 반복되는 피드백이 어떻게 영구적인 변경으로 전환되는가를 통제해야 한다**는 것입니다. 저자는 고정 가중치 설정, 유한한 진화용 데이터와 정규화 하이퍼파라미터 의존성을 한계로 인정하고, 다른 에이전트 구조·도구 생태계·더 긴 자기개선 과정에서의 추가 검증이 필요하다고 적습니다. **[Conclusion 및 Limitations, p.11]** :chatgpt-content-reference{index="135"}

다만 논문에는 구체적인 일정이나 확정된 후속 실험 계획이 제시되어 있지 않습니다. 따라서 “저자가 앞으로 특정 방법을 개발할 계획”이라고 단정하기보다는, **추가 검증이 필요한 범위를 명시했다**고 요약하는 것이 정확합니다.

### 9.2 일반화 성능 향상을 위한 우선 후속 연구

#### ① ‘새 문제’와 ‘새 모델’을 동시에 바꾸는 평가

**[후속 연구 제안]** 가장 우선할 실험은 **모델 × 벤치마크 × 도구 인터페이스**의 교차 평가입니다. 예를 들어 모델 A와 코딩 환경에서 만든 하네스를 모델 B의 새로운 코딩 환경, 모델 B의 새로운 업무 환경에 각각 적용해야 합니다. 그래야 모델 전이, 과제 전이, 도구 전이가 어느 정도 독립적으로 또는 결합되어 작동하는지 알 수 있습니다.

진화용·설정 선택용·최종 테스트용 환경을 분리하고, 최종 테스트는 탐색이 끝난 뒤 한 번만 사용하는 설계가 필요합니다. 이 방향은 RRSI가 남긴 검증 범위를 확장하면서, DomainBed가 강조한 모델 선택의 공정성 문제도 반영합니다. :chatgpt-content-reference{index="136"} :chatgpt-content-reference{index="137"}

#### ② 통계적 불확실성을 ‘문제’와 ‘진화 실행’ 두 수준에서 측정

**[후속 연구 제안]** 전체 진화 과정을 여러 독립 조건으로 반복하고, 같은 문제에서 초기·개선 하네스의 결과를 짝지어 비교해야 합니다. 업무 벤치마크는 채점 기준을 독립 표본으로 취급하기보다 문제 단위로 묶어 분석하고, 채점 모델을 바꾼 결과도 함께 보고하는 것이 좋습니다.

> **용어 풀이 — 짝지은 비교·부트스트랩:** 짝지은 비교는 같은 문제의 개선 전후 결과를 직접 비교하는 것입니다. 부트스트랩은 관찰한 문제들을 다시 뽑아 여러 가상 표본을 만들고, 성능 차이가 얼마나 흔들리는지 추정하는 방법입니다.

이때 필요한 반복 수는 임의로 정하기보다, 검출하려는 최소 개선량과 관측 변동을 기준으로 정하는 편이 타당합니다. 현재의 $k=2,4$ 설정만으로는 전체 탐색의 재현성을 평가할 수 없습니다. **[Table 5, p.23]** :chatgpt-content-reference{index="138"}

#### ③ 최종 하네스의 각 수정을 제거·복원해 일반화 원인을 확인

**[후속 연구 제안]** 최종 변경 사항을 하나씩 제거하고, 다시 조합해 어느 기능이 어떤 OOD 환경에 도움이 되는지 측정해야 합니다. 특히 “종료 전 검증”, “도구 오류 복구”, “문맥 관리”처럼 과제와 무관하게 보이는 수정이 실제로 여러 환경에서 효과를 내는지 확인하는 것이 중요합니다.

이 실험은 RRSI가 제시한 **재사용 가능한 메커니즘을 선택했다는 해석**을 단순한 전체 점수 비교보다 강하게 검증할 수 있습니다. **[Table 6, p.24]** :chatgpt-content-reference{index="139"}

#### ④ 단순 재시도와 동일한 총예산으로 비교

**[후속 연구 제안]** 비교군에는 다른 진화 알고리즘뿐 아니라, 초기 하네스에 더 많은 토큰을 주거나 여러 번 실행해 결과를 고르는 방법도 포함해야 합니다. 예산에는 제안·분석·검사·탈락 후보 평가·최종 실행 비용을 모두 포함해야 합니다.

특히 배포 횟수가 적으면 진화 비용을 회수하기 어렵고, 배포 횟수가 많으면 재사용 가능한 하네스 개선의 가치가 커질 수 있으므로, **배포 규모에 따른 비용–성능 곡선**이 필요합니다. 이는 최근 하네스 평가 연구의 핵심 요구와도 일치합니다. :chatgpt-content-reference{index="140"}

#### ⑤ 정규화와 탐색 다양성의 균형을 검증

**[후속 연구 제안]** 수정량 제한과 가지치기를 너무 강하게 하면 복잡하지만 유용한 개선 경로를 일찍 버릴 수 있습니다. RRSI의 채택 조건을 GEPA의 파레토 선택이나 DGM·HarnessBank의 다양한 후보 보존 방식과 결합하고, **평균 성능뿐 아니라 최악의 환경에서의 성능 저하**도 측정하는 연구가 유망합니다. :chatgpt-content-reference{index="141"}

핵심은 “모든 환경에서 같은 하네스가 최고”라는 목표를 먼저 가정하지 않는 것입니다. 공통으로 재사용할 부분과 모델·환경에 맞춰 바뀌어야 할 부분을 분리하는 연구가 더 현실적인 출발점입니다.

### 9.3 최종 평가

**RRSI의 가장 중요한 성과는 ‘더 많이 자기개선했다’가 아니라, ‘진화용 점수를 덜 올리더라도 새로운 환경에 남는 개선을 선택할 수 있다’는 가능성을 보여준 것입니다.** 5개 OOD 벤치마크의 일관된 초기 하네스 대비 개선, 두 종류의 실행 정책에서의 결과, 비정규화 진화 대비 토큰 절감은 이 가능성을 뒷받침합니다. **[Figure 3; Tables 2–4]** :chatgpt-content-reference{index="142"} :chatgpt-content-reference{index="143"}

그러나 현재 증거로 가장 정확하게 말할 수 있는 결론은 **“조건이 제한된 하네스 수준의 일반화 개선을 관찰했다”**입니다. 모델 가중치 자체의 일반화 향상, 모든 모델에 통하는 범용 하네스, 장기간의 안정적인 재귀적 발전, 통계적으로 확정된 우월성은 후속 연구가 검증해야 할 별도의 주장입니다.

---

## 참고자료: 사용한 원문과 공식 자료

### 분석 대상 및 공식 자료

| 자료 | 출처 |
|---|---|
| Peng Xia et al. (2026). **RRSI: Regularized Recursive Self-Improvement of Agent Harnesses.** arXiv:2609.24972v2. | 업로드된 24쪽 PDF 및 arXiv 원문 정보. :chatgpt-content-reference{index="144"} |
| **RRSI: Regularized Recursive Self-Improvement of Agent Harnesses — Official Project Page.** | 저자 공식 프로젝트 사이트, regularized-rsi.com. :chatgpt-content-reference{index="145"} |
| **google-research/rrsi — Official Repository and README.** | Google Research의 공개 GitHub 저장소. :chatgpt-content-reference{index="146"} |

### 외부 비교 연구

| 연도 | 참고자료의 전체 제목 | 1차 출처 |
|---|---|---|
| 2020 | Ishaan Gulrajani and David Lopez-Paz. **In Search of Lost Domain Generalization.** | arXiv:2007.01434. :chatgpt-content-reference{index="147"} |
| 2021 | Pang Wei Koh et al. **WILDS: A Benchmark of in-the-Wild Distribution Shifts.** | ICML, PMLR 139. :chatgpt-content-reference{index="148"} |
| 2022 | Shunyu Yao et al. **ReAct: Synergizing Reasoning and Acting in Language Models.** | arXiv:2210.03629. :chatgpt-content-reference{index="149"} |
| 2023 | Noah Shinn et al. **Reflexion: Language Agents with Verbal Reinforcement Learning.** | arXiv:2303.11366. :chatgpt-content-reference{index="150"} |
| 2023 | Omar Khattab et al. **DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines.** | arXiv:2310.03714. :chatgpt-content-reference{index="151"} |
| 2024 | Shengran Hu, Cong Lu, and Jeff Clune. **Automated Design of Agentic Systems.** | arXiv:2408.08435. :chatgpt-content-reference{index="152"} |
| 2025 | Jenny Zhang et al. **Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents.** | arXiv:2505.22954. :chatgpt-content-reference{index="153"} |
| 2025 | Lakshya A. Agrawal et al. **GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning.** | arXiv:2507.19457. :chatgpt-content-reference{index="154"} |
| 2026 | Yoonho Lee et al. **Meta-Harness: End-to-End Optimization of Model Harnesses.** | arXiv:2603.28052. :chatgpt-content-reference{index="155"} |
| 2026 | Jiahang Lin et al. **Agentic Harness Engineering: Observability-Driven Automatic Evolution of Coding-Agent Harnesses.** | arXiv:2604.25850. :chatgpt-content-reference{index="156"} |
| 2026 | Jun Nie et al. **TTHE: Test-Time Harness Evolution.** | arXiv:2607.08124. :chatgpt-content-reference{index="157"} |
| 2026 | Tingyang Chen et al. **HarnessX: A Composable, Adaptive, and Evolvable Agent Harness Foundry.** | arXiv:2606.14249. :chatgpt-content-reference{index="158"} |
| 2026 | Luan Zhang et al. **HarnessCompass: Guiding Automatic Harness Evolution toward Generalizable and Effective Agent Harnesses.** | arXiv:2608.01918. :chatgpt-content-reference{index="159"} |
| 2026 | Xiaotian Luo et al. **HarnessBank: Semantic Gene-Bank Search with Gated Verification for Agent-Harness Self-Evolution.** 초기 제목: **Self-Evolving Agent Harnesses via Gated Semantic Quality-Diversity.** | arXiv:2607.13683v2 및 초기 버전 정보. :chatgpt-content-reference{index="160"} |
| 2026 | Yike Wang et al. **Rethinking the Evaluation of Harness Evolution for Agents.** | arXiv:2607.12227. :chatgpt-content-reference{index="161"} |

RRSI의 후속 논문·정정·독립 재현 결과를 매주 확인하는 알림을 설정할까요?
