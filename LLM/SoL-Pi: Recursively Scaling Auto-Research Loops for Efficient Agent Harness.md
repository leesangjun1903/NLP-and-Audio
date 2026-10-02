# SoL-Pi: Recursively Scaling Auto-Research Loops for Efficient Agent Harness

## 1. Executive summary — 8문장

SoL-Pi의 목적은 기반 언어모델의 가중치를 다시 학습하지 않고, 도구 실행·문맥 관리·관측 처리·작업 위임을 담당하는 **하네스**를 자동 개선하여 장시간 에이전트 작업의 토큰 비용을 줄이는 것입니다〔p.2, §1〕.  
*용어 — 하네스(harness): 모델이 무엇을 보고, 어떤 도구를 쓰며, 언제 작업을 계속하거나 종료할지 중개하는 실행 소프트웨어.* :chatgpt-content-reference{index="1"}

저자들은 152개 개선 방향과 535개 실행 환경을 활용하는 넓은 탐색과 개별 후보의 반복 개선을 결합하고, 평가 기준과 최종 평가를 탐색 과정에서 분리합니다〔p.3, §2.1–2.3〕. :chatgpt-content-reference{index="2"}  
그 결과 선택된 네 메커니즘은 **Action Fusion, Online Context Compact, ObservationPack, Evidence-Preserving Reducer**입니다〔pp.4–5, Figure 4〕. :chatgpt-content-reference{index="3"} :chatgpt-content-reference{index="4"}  
네 메커니즘을 모두 적용한 구성은 EdgeBench에서 Pi 대비 기록된 토큰 트래픽을 44.7–49.0%, API 비용을 약 3분의 1 줄입니다〔pp.6–7, Tables 1–2〕. :chatgpt-content-reference{index="5"}  
다만 평균 점수는 GPT-5.6 Sol에서 44.833→42.003, Opus 5에서 44.756→42.224로 낮아지므로, 이를 **성능 저하 없는 개선**이라고 표현하는 것은 부정확합니다〔p.7, Table 2〕. :chatgpt-content-reference{index="6"}  
논문의 5.3–12.8% 성능 향상은 전체 구성이 아니라, 각 모델에서 가장 높은 점수를 낸 **서로 다른 단일 메커니즘 구성**의 결과입니다〔p.7, Table 2; p.9, Table 4〕. :chatgpt-content-reference{index="7"} :chatgpt-content-reference{index="8"}  
고정된 전체 구성이 다른 모델에서도 비용을 줄였다는 점은 유의미하지만, 별도 최종 평가용 40개 과제의 분리 결과와 반복실험 불확실성이 충분히 제시되지 않아 일반화의 강도를 확정하기는 어렵습니다〔p.6, §2.5; pp.6–9, Tables 1–4〕. :chatgpt-content-reference{index="9"}  
따라서 이 연구는 **“기반 모델의 보편적 능력이 향상되었다”기보다 “재사용 가능한 실행 효율화 규칙이 일부 새로운 과제·모델로 이전될 수 있다”는 초기 증거**로 읽는 것이 타당하며, 지속적인 재귀적 개선과 스케일링 법칙은 후속 연구 과제로 남습니다〔p.12, §5.1〕. :chatgpt-content-reference{index="10"}

---

## 2. 연구의 목적·필요성과 핵심 주장

### 2.1 무엇을 해결하려는가?

**[저자 보고]** 장시간 코딩 에이전트에서는 코드 생성 자체뿐 아니라, 이전 기록을 반복해서 입력하는 비용, 대용량 로그를 여러 번 읽는 비용, 편집 직후 실행할 명령을 별도 모델 호출로 결정하는 비용이 누적됩니다. 하네스는 이러한 낭비를 줄일 수 있지만, 문맥 압축·검증·복구·위임이 서로 얽혀 있어 한 부분의 최적화가 이후 실패나 추가 비용을 유발할 수 있습니다〔p.2, §1〕. :chatgpt-content-reference{index="11"}

**[해석]** 연구 질문은 “토큰을 무조건 적게 쓰게 만들 수 있는가?”가 아닙니다. 더 정확하게는 **“완료해야 할 작업과 필요한 증거를 유지하면서, 특정 문제의 정답에 의존하지 않는 반복 비용을 자동으로 찾아 제거할 수 있는가?”**입니다. 성공한다면 모델 교체나 추가 학습 없이도 적용 가능한 개선이 됩니다. 다만 ‘필요한 작업을 유지했다’는 판단은 선언이 아니라 실제 평가로 검증해야 합니다〔pp.2–3, §2.1〕. :chatgpt-content-reference{index="12"} :chatgpt-content-reference{index="13"}

### 2.2 핵심 주장과 근거

| 핵심 주장 | 저자가 제시한 근거 | 해석 및 주의점 | 원문 위치 |
|---|---|---|---|
| 하네스 개선을 자동 연구 대상으로 삼을 수 있다 | 152개 방향, 535개 환경, 3,000회 이상 실행, 60,000회 이상 상호작용 | 상당한 탐색 규모이지만, 자동 탐색이 수동 설계보다 우수하다는 대조실험은 아님 | p.2 §1; p.3 §2.2–2.3. :chatgpt-content-reference{index="14"} :chatgpt-content-reference{index="15"} |
| 전체 구성은 실행 비용을 크게 낮춘다 | Pi 대비 토큰 44.7–49.0%, API 비용 약 33% 감소 | 가장 직접적으로 지지되는 주장이나, 점수 감소를 동반 | pp.6–7, Tables 1–2. :chatgpt-content-reference{index="16"} |
| 일부 구성은 점수도 높인다 | Sol에서는 ObservationPack, Opus에서는 Action Fusion이 최고 점수 | 모델별 평가 결과에서 선택된 구성으로, 하나의 고정된 ‘성능 최적 구성’이 아님 | p.7 Table 2; p.9 Table 4. :chatgpt-content-reference{index="17"} |
| 다른 모델로 이전된다 | Sol에서 개발한 전체 구성을 추가 탐색 없이 Opus에 적용 | **효율성 이전의 초기 증거**이며, 모델 일반화 전반의 보장은 아님 | p.6 §3.1; p.7 Table 2. :chatgpt-content-reference{index="18"} |
| 다른 벤치마크에서도 효율적이다 | Terminal-Bench 4와 IMO 2026에서 해결당 비용 감소 | Terminal-Bench 해결 수는 18→15, IMO는 Pi와 같은 3/6 | p.7 Table 3. :chatgpt-content-reference{index="19"} |
| 네 메커니즘은 상호 보완적이다 | 단독 적용·전체 적용의 활성화와 효율 비교 | 서로 다른 활성화 과제 부분집합을 비교하므로 인과적 상호작용을 분리하지 못함 | p.8 §3.4; p.10 Figure 7. :chatgpt-content-reference{index="20"} :chatgpt-content-reference{index="21"} |
| 더 효율적인 하네스가 다음 연구 주기를 가속할 수 있다 | 차세대 연구 루프의 출발점으로 SoL-Pi를 사용할 계획 | 현재 실증 결과가 아니라 장기 연구 가설 | p.12 §5.1. :chatgpt-content-reference{index="22"} |

*용어 — 재귀적 자기개선(RSI): 개선된 시스템이 다음 개선 과정에도 참여하는 구조; 전이(transfer): 개발에 쓰지 않은 과제나 모델에서도 개선이 유지되는 현상; 상호작용 효과: 두 기능을 함께 썼을 때의 효과가 각 기능의 개별 효과만으로 설명되지 않는 부분.*

---

## 3. 제안 방법과 시스템 구조 — 수식 포함

### 먼저 구분해야 할 점: 원문에는 제안법을 규정하는 명시적 수식이 없습니다

**첨부 논문의 방법론은 주로 절차와 구현 설명으로 제시됩니다. 아래 ‘설명식’은 그 서술을 이해하기 위해 제가 수학적으로 재구성한 것이며, 논문에 실린 공식이나 실제 소스코드의 정확한 계산식을 옮긴 것이 아닙니다.** 특히 허용오차의 수치와 문맥 압축 게이트의 세부 상수는 PDF에 제시되지 않습니다〔pp.3–5, §2〕. :chatgpt-content-reference{index="23"} :chatgpt-content-reference{index="24"} :chatgpt-content-reference{index="25"}

### 3.1 ‘모델 구조’가 아니라 ‘모델을 감싸는 실행 구조’의 변경

**[저자 보고]** SoL-Pi는 새로운 신경망을 제안하거나 모델 가중치를 업데이트하지 않습니다. Pi의 확장 기능으로 네 메커니즘을 구현하고, 실행 과정에서 모델에 전달할 정보와 도구 호출 방식을 변경합니다. 로그 발췌에는 별도의 저비용 모델인 GPT-5.6 Luna를 사용합니다〔p.5, §2.5; p.11, §4.1〕. :chatgpt-content-reference{index="26"} :chatgpt-content-reference{index="27"}

**[설명식 1: 실행 구조의 개념적 표현]**

$$
z_t=P_\phi(\tau_t),\qquad
a_t\sim\pi_\theta(\cdot\mid z_t,\mathcal{T}_\phi),\qquad
o_{t+1}=E\!\left(X_\phi(a_t)\right).
$$

여기서 $t$는 실행 단계, $\tau_t$는 현재까지의 실행 기록, $z_t$는 모델에 실제로 제시되는 문맥입니다. $\pi_\theta$는 가중치 $\theta$가 고정된 기반 모델, $\phi$는 변경 가능한 하네스 코드·설정, $\mathcal{T}\_\phi$는 도구 인터페이스입니다. $P_\phi$는 문맥 구성, $X_\phi$는 도구 실행 처리, $E$는 작업 환경, $a_t$는 행동, $o_{t+1}$은 환경이 반환한 관측을 뜻합니다.

핵심은 **$\theta$가 아니라 $\phi$를 바꾼다**는 점입니다. 따라서 여기서 ‘학습된 메커니즘’은 새로운 신경망 파라미터가 아니라, 탐색으로 선택된 코드·규칙·프롬프트를 의미합니다.

*용어 — 실행 궤적 또는 트래젝터리(trajectory): 모델의 판단, 도구 호출, 환경 응답이 시간 순서로 쌓인 기록; 도구 인터페이스: 모델이 사용할 수 있는 명령과 입력·출력 형식.*

### 3.2 외부 연구 루프: 넓게 제안하고, 유망한 후보를 깊게 개선

**[저자 보고]** 탐색 방향은 문맥, 진행 관리, 도구, 위임, 프롬프트·정책, 개선·평가의 여섯 계열로 분류됩니다. 기존 실행 기록에서 낭비를 찾는 Oracle Analysis 이후, 각 후보를 독립적으로 구현·검토·검증하고, 실패하면 수정하거나 폐기합니다〔p.3, §2.2〕. :chatgpt-content-reference{index="28"}

*용어 — Oracle Analysis: 이 논문에서는 기존 기록을 분석해 제거 가능한 작업량을 추정하는 진단 단계; 모든 정답을 아는 실제 모델을 뜻하지 않습니다.*

각 실행 기록은 별도 분석자가 읽고, 종합자가 결과를 합치는 **Map–Reduce 분석**을 거칩니다. 구현은 완료 조건을 만족할 때까지 수정하는 Ralph Loop 방식으로 진행되며, 별도 검토자가 확인합니다. 각 탐색 계열의 조정 코드는 실험 후 폐기하고, 후보와 증거를 보존합니다〔p.3, §2.2, Figure 2〕. :chatgpt-content-reference{index="29"}

*용어 — Map–Reduce: 여러 자료를 각각 분석한 뒤 결과를 합치는 처리 방식; Ralph Loop: 명시된 완료 조건을 충족할 때까지 구현과 검토를 반복하는 작업 루프.*

#### 후보 수용 기준

**[저자 보고]** 먼저 모든 능력 지표가 사전 허용 범위를 만족해야 하며, 그다음 적어도 하나의 효율 지표가 개선되어야 합니다. 통과 후보 중에서는 다른 후보에 일방적으로 뒤처지지 않는 결과를 남깁니다〔p.3, §2.1〕. :chatgpt-content-reference{index="30"}

**[설명식 2: 절대 허용오차를 가정한 단순화]**

$$
\text{Accept}(h)=
\left[
\bigwedge_j Q_j(h)\ge Q_j(h_0)-\epsilon_j
\right]
\land
\left[
\bigvee_k E_k(h) < E_k(h_0)
\right].
$$

$h$는 후보 하네스, $h_0$는 비교 기준 하네스입니다. $Q_j$는 클수록 좋은 $j$번째 능력 지표, $\epsilon_j$는 허용 감소량, $E_k$는 작을수록 좋은 $k$번째 효율 지표입니다. $\bigwedge$는 ‘모든 조건’, $\bigvee$는 ‘하나 이상의 조건’, $\land$는 ‘그리고’를 뜻합니다. 원문이 실제로 절대오차와 상대오차 중 무엇을 사용했는지는 이 식만으로 확정할 수 없습니다.

*용어 — 파레토 비지배 후보: 다른 후보가 모든 평가 항목에서 같거나 더 좋으면서 일부에서는 더 좋은 경우가 존재하지 않는 후보.*

**[해석]** 이 기준은 “정확히 같은 성능”이 아니라 “허용된 성능 감소 이내”를 요구합니다. 따라서 최종 점수 감소 자체가 프로토콜 위반이라고 단정할 수는 없지만, 허용오차가 공개되지 않으면 독자는 그 판단을 재검증하기 어렵습니다.

### 3.3 연구 환경과 최종 평가의 분리

**[저자 보고]** 535개 개발 환경은 다음 두 종류입니다.

| 환경 | 구성과 검증 | 주의점 |
|---|---|---|
| 저장소 기반 495개 과제 | GitHub issue–PR 쌍에서 수정 전 저장소를 복원하고, 수정 전 실패·수정 후 성공하는 회귀 테스트로 검증 | 495개는 **과제 수**이며, 고유 저장소 수라고 해석하면 안 됨 |
| 합성 40개 과제 | 성공 판정 프로그램을 먼저 만들고, 여러 해결 경로가 가능한 환경을 구성 | 검증 프로그램이 포착하지 못하는 실패 가능성은 별도 문제 |

PR과 회귀 테스트는 작업을 수행하는 에이전트에게 숨깁니다〔pp.3–4, §2.3, Figure 3〕. :chatgpt-content-reference{index="31"} :chatgpt-content-reference{index="32"}

*용어 — 회귀 테스트: 수정으로 문제가 해결되고 기존 기능이 다시 망가지지 않는지 확인하는 테스트; 검증기(verifier): 결과가 정해진 성공 조건을 만족하는지 판정하는 프로그램.*

EdgeBench의 공개 51개 과제 중 11개는 고정 후보의 일방향 수용 판단에, 나머지 40개는 최종 일반화 평가에 사용한다고 명시합니다. 그러나 주된 결과 표는 51개 집계이며, 두 부분의 결과를 따로 제시하지 않습니다〔p.6, §2.5–3.1〕. :chatgpt-content-reference{index="33"}

**[해석]** 11개에서 얻은 결과를 코드 수정에 돌려주지 않더라도, 수용 판단에 사용한 집합과 최종 시험 집합은 역할이 다릅니다. 따라서 **51개 전체 집계와 완전히 독립적인 40개 시험 결과를 동일시하면 안 됩니다.**

### 3.4 네 메커니즘의 작동 원리

#### A. Action Fusion: 중간 판단이 필요 없는 편집과 실행을 결합

**[저자 보고]** 파일 편집 후 테스트·빌드·실행을 별도로 요청하는 대신, 편집과 후속 명령을 하나의 도구 요청으로 묶습니다. 편집 결과를 보고 명령을 결정해야 하는 경우에는 분리합니다〔p.4, §2.4; p.5, Figure 4(a)〕. :chatgpt-content-reference{index="34"} :chatgpt-content-reference{index="35"}

그림의 국소 실행 예시에서는 다음과 같습니다.

$$
n_{\text{API}}:3\rightarrow2.
$$

$n_{\text{API}}$는 그림에 제시된 해당 실행 구간의 API 호출 수입니다. **전체 작업의 호출 수가 항상 3분의 1 감소한다는 뜻은 아닙니다.**

**[해석]** 일반화 가능성이 비교적 높은 이유는 특정 정답이 아니라 “편집→예정된 검증”이라는 반복적인 작업 구조를 겨냥하기 때문입니다. 반대로 중간 결과에 따라 분기해야 하는 작업까지 묶으면 판단 기회를 제거할 수 있습니다.

#### B. Online Context Compact: 압축 비용을 회수할 수 있을 때 문맥 압축

**[저자 보고]** `update_plan`으로 관리되는 계획 단계의 완료 시점에 남은 요청 수를 추정하고, 향후 입력 절감액이 프롬프트 캐시 재작성 비용을 보상할지 판단합니다. 이후 압축에서는 아직 회수하지 못한 비용과 더 큰 절감 여유도 고려합니다〔p.4, §2.4〕. :chatgpt-content-reference{index="36"}

*용어 — 문맥 압축: 긴 실행 기록을 더 짧은 요약·상태 표현으로 바꾸는 것; 프롬프트 캐시: 반복 입력의 일부를 재사용하는 기능으로, 읽기와 새로 쓰기의 비용이 다를 수 있습니다.*

**[설명식 3: 비용 게이트의 개념적 근사]**

$$
\widehat{R}_t=
\min\left(
\bar r_t u_t,\,
\frac{W-L_t}{g_t}
\right),
\qquad
\widehat{R}_t(L_t-L'_t)p_r>
\widehat K_t+D_t+M_t.
$$

$t$는 계획 단계 완료 시점, $\widehat R_t$는 남은 요청 수 추정치, $\bar r_t$는 완료된 단계당 관측 요청 수, $u_t$는 미완료 단계 수입니다. $W$는 문맥 한도, $L_t$와 $L'_t$는 압축 전후 문맥 길이, $g_t$는 요청당 문맥 증가량, $p_r$는 캐시 읽기 단가입니다. $\widehat K_t$는 예상 재작성 비용, $D_t$는 미회수 비용, $M_t$는 추가로 요구하는 절감 여유입니다. 이 근사는 $g_t>0$, $L_t<W$인 경우를 상정하며, 세부 계산과 여유 적용 방식은 원문에 명시되지 않습니다.

문맥이 한도에 가까워질 때의 압축 경로도 별도로 존재합니다. 또한 저자들은 **게이트가 요약 호출 자체의 비용을 별도로 가격화하지 않는다**고 밝힙니다〔p.5, §2.5〕. :chatgpt-content-reference{index="37"}

**[해석]** 따라서 이는 완전한 미래 비용 최적화가 아니라 관측 기록에 의존하는 비용 추정 규칙입니다. 계획이 부정확하거나 남은 작업량이 갑자기 바뀌면 추정이 빗나갈 수 있습니다.

#### C. ObservationPack: 원문 접근은 유지하고 반복 전송만 줄임

**[저자 보고]** 10 KiB를 초과하는 도구 출력을 로컬에 보관하고, 이후 첫 두 번의 모델 제공자 요청에는 원문을 보냅니다. 세 번째 요청부터는 고정 식별자, 원래 크기, 앞뒤의 완전한 줄을 담은 짧은 발췌로 대체하며, 필요한 원문 구간은 다시 가져올 수 있습니다〔p.4, §2.4; p.5, Figure 4(c)〕. :chatgpt-content-reference{index="38"} :chatgpt-content-reference{index="39"}

**[설명식 4]**

$$
P_k(o)=
\begin{cases}
o,
& \text{size}(o)\le10\,\text{KiB}\ \text{or}\ k\le2,\\
\bigl(\text{handle}(o),\text{size}(o),\text{excerpt}(o)\bigr),
& \text{otherwise}.
\end{cases}
$$

$o$는 도구 출력, $k$는 그 출력이 생성된 뒤의 모델 제공자 요청 순번, $P_k(o)$는 해당 요청에 포함할 표현입니다. $\text{size}$는 크기, $\text{handle}$은 저장 원문을 다시 찾는 식별자, $\text{excerpt}$는 짧은 발췌입니다.

*용어 — KiB: 1,024바이트 단위; 핸들(handle): 전체 내용을 담는 대신 저장 위치를 가리키는 식별자.*

**[해석]** 중요한 차이는 “원문을 삭제한다”가 아니라 “매번 전송하지 않는다”입니다. 그러나 원문이 저장되어 있다는 사실만으로, 모델이 나중에 필요한 정보를 정확히 다시 찾아본다는 보장까지 생기지는 않습니다.

#### D. Evidence-Preserving Reducer: 저비용 모델이 발췌하고 규칙으로 대조

**[저자 보고]** 사전에 정한 빌드·테스트 명령의 4 KiB 이상 로그를 대상으로 저비용 모델이 증거를 발췌합니다. 파일 읽기와 검색 결과는 대상에서 제외합니다. 별도 검증기가 형식, 원문 해시, 종료 상태, 정확한 인용, 크기를 확인하고, 실패하거나 축소 효과가 없으면 원문을 사용합니다〔p.5, §2.4〕. :chatgpt-content-reference{index="40"}

*용어 — 증거 영수증(receipt): 원문과 대조할 수 있는 짧은 증거 발췌 묶음; 해시(hash): 내용의 동일성을 확인하기 위한 지문; 결정적 검증: 같은 입력에 항상 같은 결과를 내는 규칙 기반 검사.*

**[설명식 5]**

$$
R(o)=
\begin{cases}
r,
& \text{Eligible}(o)\land V(r,o)=1
  \land \text{size}(r)<\text{size}(o),\\
o,
& \text{otherwise}.
\end{cases}
$$

$r$은 보조 모델이 만든 발췌, $R(o)$는 주 모델에 전달할 결과, $\text{Eligible}(o)$는 명령 종류·크기 등 적용 조건, $V(r,o)$는 발췌와 원문을 대조하는 검증 함수입니다.

이 처리는 ObservationPack보다 먼저 수행되며, ObservationPack은 검증된 receipt를 다시 축약하지 않습니다〔p.5, §2.4, Figure 4(d)〕. :chatgpt-content-reference{index="41"}

**[해석 — 중요한 한계]** 정확한 인용과 해시는 **발췌가 원문에서 왔다는 사실**을 확인하지만, **해결에 필요한 모든 정보를 포함했다는 사실**은 확인하지 못합니다. 이는 출처 보존과 의미적 충분성의 차이입니다.

### 3.5 비용 지표는 어떻게 읽어야 하는가?

**[저자 보고]** 논문은 토큰 트래픽을 입력·캐시 읽기·캐시 쓰기·출력으로 나누며, 효율은 전체 과제 점수 합계당 API 비용으로 측정합니다. 비용은 2026년 8월 17일 가격을 사용합니다〔p.6, §3, Table 1〕. :chatgpt-content-reference{index="42"} :chatgpt-content-reference{index="43"}

**[설명식 6: 지표 정의의 수학적 표현]**

$$
C(h)=\sum_b\sum_{q\in\mathcal Q_b}p_{bq}T_{bq}(h),
\qquad
E(h)=\frac{C(h)}{\sum_{i=1}^{N}s_i(h)}
=\frac{C(h)}{N\bar s(h)}.
$$

$C(h)$는 총 API 비용, $b$는 모델·제공자, $\mathcal Q_b$는 해당 제공자의 과금 범주, $p_{bq}$는 토큰당 단가, $T_{bq}$는 범주별 사용량입니다. $s_i$는 과제 $i$의 점수, $N$은 과제 수, $\bar s$는 평균 점수, $E$는 점수당 비용으로 **작을수록 좋습니다**.

예를 들어 전체 Sol 구성은 표의 반올림 수치로 다음과 같이 근사됩니다.

$$
E\approx\frac{894}{51\times42.003}\approx0.4173.
$$

표의 0.4174와의 작은 차이는 표시된 비용·점수의 반올림과 양립합니다. 여기서 894는 달러 비용이며, 평균 점수로만 나누는 것이 아니라 **51개 점수의 합계로 나눈다**는 점이 중요합니다〔p.6, Table 1〕. :chatgpt-content-reference{index="44"}

---

## 4. 성능 향상과 비용 절감: 어떤 구성이 무엇을 개선했는가?

### 4.1 EdgeBench 주요 결과

아래는 **저자 보고 수치**입니다. B는 10억 토큰이며, 점수는 해결률과 동일한 지표가 아닙니다〔p.7, Table 2〕. :chatgpt-content-reference{index="45"}

| 모델 | 구성 | 평균 점수 ↑ | 토큰 트래픽(B) ↓ | API 비용(USD) ↓ | 점수당 비용 ↓ |
|---|---|---:|---:|---:|---:|
| GPT-5.6 Sol | Codex | 34.738 | 3.0537 | 1,787 | 1.0086 |
| GPT-5.6 Sol | Pi | 44.833 | 2.1538 | 1,339 | 0.5855 |
| GPT-5.6 Sol | SoL-Pi 전체 구성 | 42.003 | 1.0990 | 894 | 0.4174 |
| GPT-5.6 Sol | SoL-Pi 성능 구성: ObservationPack | 47.208 | 2.0224 | 1,271 | 0.5280 |
| Opus 5 | Claude Code | 43.689 | 2.0045 | 2,535 | 1.1377 |
| Opus 5 | Pi | 44.756 | 2.3697 | 1,741 | 0.7625 |
| Opus 5 | SoL-Pi 전체 구성 | 42.224 | 1.3101 | 1,158 | 0.5376 |
| Opus 5 | SoL-Pi 성능 구성: Action Fusion | 50.482 | 2.1016 | 1,605 | 0.6235 |

**[재계산: 같은 모델의 Pi 대비]**

| 구성 | 평균 점수 상대 변화 | 토큰 감소 | API 비용 감소 | 점수당 비용 감소 |
|---|---:|---:|---:|---:|
| Sol 전체 | **−6.31%** | 48.97% | 33.23% | 28.71% |
| Opus 전체 | **−5.66%** | 44.71% | 33.49% | 29.50% |
| Sol 성능 구성 | +5.30% | 6.10% | 5.08% | 9.82% |
| Opus 성능 구성 | +12.79% | 11.31% | 7.81% | 18.23% |

**[해석]** 전체 구성은 ‘최고 점수’보다 ‘낮은 비용’을 선택한 운영 지점입니다. 반면 성능 구성은 평가된 단일 메커니즘 중 최고 점수를 선택한 결과이므로, **“점수 12.8% 향상과 토큰 49% 절감”을 한 구성의 동시 성과처럼 합쳐 말하면 안 됩니다.**

### 4.2 개별 메커니즘의 효과와 중요한 예외

아래 점수는 각 메커니즘을 Pi에 하나씩 추가한 결과입니다〔p.9, Table 4〕. :chatgpt-content-reference{index="46"}

| 구성 | Sol 평균 점수 | Opus 평균 점수 |
|---|---:|---:|
| Pi | 44.833 | 44.756 |
| + Action Fusion | 46.664 | **50.482** |
| + Online Context Compact | 41.993 | 49.155 |
| + Evidence-Preserving Reducer | 44.630 | 43.405 |
| + ObservationPack | **47.208** | 47.047 |
| 네 메커니즘 전체 | 42.003 | 42.224 |

**[해석]** Online Context Compact의 점수 변화 방향이 모델에 따라 반대이고, 전체 구성은 일부 단일 구성보다 낮습니다. 따라서 “좋은 메커니즘을 더 많이 합치면 더 좋은 에이전트가 된다”는 결론은 성립하지 않습니다.

특히 **Opus에서 전체 구성이 점수당 비용까지 최선은 아닙니다.** ObservationPack 단독은 비용 1,176달러·점수 47.047·점수당 비용 0.4899이고, 전체 구성은 1,158달러·42.224·0.5376입니다. 즉 단독 구성이 전체보다 18달러 더 들지만 점수는 약 11.42% 높고, 점수당 비용도 더 낮습니다〔p.9, Table 4〕. :chatgpt-content-reference{index="47"}

따라서 논문의 ‘Efficiency’라는 명칭은 **평가된 모든 효율 정의에서 최적이라는 뜻이 아니라, 전체 스택의 저비용 운영 지점**으로 읽어야 합니다.

### 4.3 다른 벤치마크: 비용 개선과 능력 개선을 구분

다음은 저자 보고 결과입니다〔p.7, Table 3〕. :chatgpt-content-reference{index="48"}

| 평가 | 구성 | 해결 수 | 총 모델 비용(USD) | 해결당 비용(USD) |
|---|---|---:|---:|---:|
| Terminal-Bench 4, CPU 전용 63개 | Codex | 18/63 | 272.35 | 15.13 |
| 동일 | Pi | 18/63 | 286.45 | 15.91 |
| 동일 | SoL-Pi | **15/63** | **211.12** | **14.07** |
| IMO 2026, 6문제 | Codex | **5/6** | 114.47 | 22.89 |
| 동일 | Pi | 3/6 | 75.95 | 25.32 |
| 동일 | SoL-Pi | 3/6 | **62.69** | **20.90** |

**[재계산·해석]** Terminal-Bench의 해결률은 Pi의 28.57%에서 23.81%로 **4.76퍼센트포인트 감소**합니다. 총비용과 해결당 비용은 개선되지만, 문제 해결 능력이 유지되었다고 단정할 수 없습니다. IMO에서는 Pi와 해결 개수가 같고 더 저렴하지만, 6문제만으로 수학적 일반화 성능을 평가하기에는 증거가 제한적입니다.

*용어 — 퍼센트포인트: 두 비율의 단순 차이; 28.57%→23.81%는 −4.76퍼센트포인트이며, 상대 감소율은 약 16.67%입니다.*

IMO 결과는 Lean 4로 형식화·검증하며 문제당 150분 한도를 적용합니다. 따라서 자연어 수학 풀이 평가와도 그대로 비교할 수 없습니다〔p.7, Table 3 주석〕. :chatgpt-content-reference{index="49"}

*용어 — Lean 4: 증명이 정해진 논리 규칙을 따르는지 컴퓨터로 검사하는 정리 증명 도구.*

### 4.4 에이전트 군집 실험

**[저자 보고]** 각 조건에서 한 번씩, 두 시간 동안 수행한 커널 최적화 실험의 최종 결과는 SoL-Pi 작업자 20개 군집이 **1,127 cycles·60.11달러**, Pi 작업자 군집이 **1,366 cycles·82.12달러**, 단일 Codex 에이전트가 **1,333 cycles·39.20달러**입니다〔pp.7–8, §3.3, Figure 5〕. :chatgpt-content-reference{index="50"} :chatgpt-content-reference{index="51"}

*용어 — 군집(swarm): 여러 에이전트가 병렬로 탐색하고 일부 발견을 공유하는 구조; cycles: 여기서는 최적화된 코드의 시뮬레이터 실행 주기이며, 에이전트의 응답 시간 자체가 아닙니다.*

**[재계산·해석]** SoL-Pi 군집은 Pi 군집보다 cycles가 약 17.5%, API 비용이 26.8% 낮습니다. 그러나 **같은 두 시간은 같은 달러 예산이 아니며**, 단일 에이전트가 가장 저렴합니다. 반복 없는 한 과제 실험이므로 군집 일반화나 일관된 우월성의 증명보다는 사례 연구에 가깝습니다.

---

## 5. 일반화 성능 향상 가능성: 무엇이 입증되고 무엇이 남았는가?

### 5.1 일반화를 네 수준으로 나누어야 합니다

| 일반화의 종류 | 논문이 제공하는 증거 | 판단 |
|---|---|---|
| 개발 과제 → 새 평가 과제 | 535개 개발 환경과 EdgeBench의 분리 | 긍정적 설계이지만 40개 최종 시험 집계가 별도로 필요 |
| 개발 모델 → 다른 모델 | Sol에서 개발한 전체 구성을 Opus에 추가 탐색 없이 적용 | **효율성 이전을 뒷받침하는 초기 증거** |
| 코딩 중심 환경 → 다른 과제 유형 | Terminal-Bench, Lean 기반 IMO, 커널 최적화 | 평가 범위를 넓혔으나 해결 성능 향상은 일관되지 않음 |
| 기반 모델 자체의 일반 능력 | 가중치 학습이나 모델 단독 평가 없음 | **입증하지 않음** |

근거: p.6 §2.5·§3.1, p.7 Table 3, p.12 §5.1. :chatgpt-content-reference{index="52"} :chatgpt-content-reference{index="53"} :chatgpt-content-reference{index="54"} :chatgpt-content-reference{index="55"}

### 5.2 왜 일반화될 가능성이 있는가?

**[해석]** 가장 설득력 있는 이유는 개선 대상이 특정 문제의 정답보다 **작업 간 공통적인 실행 낭비**라는 점입니다. 예정된 편집·검증을 결합하고, 오래된 대형 관측의 반복 전송을 줄이고, 원문을 다시 찾을 수 있게 만드는 규칙은 여러 저장소와 모델에 적용될 여지가 있습니다. 저자들도 재사용 가능한 효율화 메커니즘을 탐색 목표로 제시합니다〔pp.2–4, §2.1–2.4〕. :chatgpt-content-reference{index="56"} :chatgpt-content-reference{index="57"}

그러나 이 구조적 가능성이 곧 일반화의 실증은 아닙니다. 계획을 얼마나 충실히 사용하는지, 로그에서 어떤 증거를 필요로 하는지, 압축 이후 정보를 얼마나 잘 회수하는지는 모델과 과제에 따라 달라질 수 있으며, Figure 6의 활성화 차이는 이러한 의존성이 실제로 존재함을 보여줍니다〔p.9, Figure 6〕. :chatgpt-content-reference{index="58"}

### 5.3 효율 개선이 능력 개선으로 이어지는 경로는 아직 가설입니다

**[해석]** 평균 실행비가 약 3분의 1 감소한다면, 같은 예산으로 계산상 약 1.5배의 실행을 배정할 여지가 있습니다. 하지만 추가 실행이 서로 다른 정보를 만들고, 올바른 후보를 선택할 수 있어야 실제 성능 향상으로 이어집니다.

저자들이 제안한 “더 저렴한 하네스로 더 많은 연구를 수행해 다음 하네스를 개선한다”는 경로도 이러한 조건에 의존하며, 현재 논문은 그 누적 효과를 입증하지 않았다고 명시합니다〔p.12, §5.1〕. :chatgpt-content-reference{index="59"}

---

## 6. 통계적으로 취약한 부분과 직접 비교하면 안 되는 수치

### 6.1 통계·평가 설계의 취약점

| 표시 | 취약한 부분 | 왜 중요한가 |
|---|---|---|
| **[통계 취약]** | 주요 표에 반복 횟수·시드·분산·신뢰구간이 제시되지 않음 | 점수 증가·감소의 재현성과 불확실성을 추정하기 어려움〔pp.6–9, Tables 1–4〕. :chatgpt-content-reference{index="60"} :chatgpt-content-reference{index="61"} |
| **[미검증]** | ‘비슷한 성능’에 대한 비열등성 검정 없음 | 점수가 비슷해 보이는 것과 허용 감소폭 이내임을 통계적으로 보이는 것은 다름〔p.3 수용 기준; p.7 Table 2〕. :chatgpt-content-reference{index="62"} :chatgpt-content-reference{index="63"} |
| **[선택 편향 위험]** | Performance 구성을 모델별 최고 점수로 선택 | 여러 후보 중 우연히 높게 나온 결과가 포함될 수 있어 독립 재평가가 필요〔p.6 §3.1; p.9 Table 4〕. :chatgpt-content-reference{index="64"} |
| **[보고 부족]** | 11개 수용 집합과 40개 최종 시험 집합의 결과 미분리 | 완전한 최종 시험 일반화 성능을 집계표에서 확인할 수 없음〔p.6 §2.5〕. :chatgpt-content-reference{index="65"} |
| **[표본 한계]** | IMO 6문제, 군집은 조건당 1회 | 개별 문제·실행의 영향이 크며 결과 분포를 알 수 없음〔p.7 Table 3, §3.3〕. :chatgpt-content-reference{index="66"} :chatgpt-content-reference{index="67"} |
| **[인과 해석 제한]** | Figure 7은 구성별로 다른 활성화 과제 집합을 사용 | 관측된 차이가 모듈 결합 때문인지 과제 구성 차이 때문인지 분리되지 않음〔p.10 Figure 7〕. :chatgpt-content-reference{index="68"} |
| **[스케일링 미검증]** | 탐색 폭·깊이를 동일 예산으로 비교하지 않음 | 많은 탐색을 했다는 사실만으로 탐색량 증가의 인과적 이득이나 스케일링 법칙을 도출할 수 없음〔p.3 §2.2; p.12 §5.1〕. :chatgpt-content-reference{index="69"} :chatgpt-content-reference{index="70"} |
| **[비용 범위 제한]** | 결과 표는 API 비용 중심 | 전체 탐색비·사람의 작업·환경 실행·저장·검증 비용을 포함한 총경제성은 별도 분석 필요〔p.6 §3; p.12 §5.1〕. :chatgpt-content-reference{index="71"} :chatgpt-content-reference{index="72"} |

*용어 — 신뢰구간: 관측 결과의 불확실성을 나타내는 범위; 비열등성 검정: 새 방법의 성능 감소가 사전에 정한 허용폭보다 작다는 것을 평가하는 절차; 선택 편향: 여러 결과 중 좋은 것만 고르면서 기대 성능을 과대평가할 위험.*

또한 **60,000회 상호작용은 60,000개의 독립 평가 표본이 아닙니다.** 동일 과제·실행 안의 상호작용은 연결되어 있기 때문에, 통계적 독립성의 단위를 과제·저장소·실행 반복 수준에서 따져야 합니다〔p.2 탐색 규모〕. :chatgpt-content-reference{index="73"}

### 6.2 직접 비교하면 안 되는 수치

| 비교 | 판정 |
|---|---|
| 전체 구성의 토큰 −49%와 성능 구성의 점수 +12.8% | **다른 구성·다른 모델의 결과이므로 결합 불가**〔Tables 2·4〕 |
| Figure 1의 비용 −50.0%·−54.3%와 Pi 대비 약 −33% | **기준 하네스가 다름**: 전자는 Codex·Claude Code 대비〔p.1 Figure 1; p.7 Table 2〕 |
| Figure 8의 예상 토큰 −11.5%와 실제 EdgeBench 절감률 | **반사실적 추정과 실측 결과의 차이**〔p.10 Figure 8〕 |
| EdgeBench 공식 GPT-5.5 점수 31.2와 Sol 기반 하네스 비교 | 모델이 다르고 비용·토큰이 미보고되어 논문도 순위 밖 참고값으로 처리〔p.6 Table 1〕 |
| Terminal-Bench 4의 CPU 전용 63개 결과와 전체 벤치마크 | GPU 의존 과제가 제외된 부분집합 평가〔p.7 Table 3 주석〕 |
| 총 토큰 트래픽 감소와 GPU 연산·실행시간 감소 | 캐시 재전송량을 포함하므로 같은 양이 아님〔p.9 Table 4, Cache Reuse and Total Cost〕 |

각 구분의 근거는 Figure 1 설명, Tables 1–3 주석, Figure 8 설명에 명시되어 있습니다. :chatgpt-content-reference{index="74"} :chatgpt-content-reference{index="75"} :chatgpt-content-reference{index="76"} :chatgpt-content-reference{index="77"} :chatgpt-content-reference{index="78"}

*용어 — 반사실적 추정: 실제로 다시 실행한 결과가 아니라, 관측 기록을 바탕으로 “모든 가능한 지점에서 기능이 작동했다면”을 계산한 값.*

---

## 7. 가장 중요한 그림의 선정과 해석

### Figure 2·3 — 일반화 주장을 이해하는 출발점〔pp.3–4〕

**[그림이 보여주는 것]** 개발 기록→분석→제안→구현→검토→개발 검증의 순환과, 그 밖에 놓인 고정 후 평가를 구분합니다. 개발 환경도 저장소 기반과 검증기 기반의 두 경로로 나눕니다. :chatgpt-content-reference{index="79"} :chatgpt-content-reference{index="80"}

**[해석]** 중요한 기여는 평가 실패를 특정 과제용 수정으로 되돌리지 않으려는 설계입니다. 다만 그림은 설계 의도를 보여주는 것이므로, 실제 데이터 중복 검사와 40개 시험 집합의 분리 보고를 대신하지는 못합니다.

### Figure 4 — 네 메커니즘의 책임 경계〔p.5〕

**[그림이 보여주는 것]** 호출 결합, 문맥 압축, 대형 관측의 지연 축약, 검증된 증거 발췌가 각각 다른 위치에서 작동합니다. 특히 reducer의 검증 실패 시 원문 복귀 경로와 ObservationPack의 원문 회수 경로가 중요합니다. :chatgpt-content-reference{index="81"}

**[해석]** 단순히 “요약을 더 많이 한다”가 아니라, **중간 판단이 필요한가, 원문에 다시 접근할 수 있는가, 발췌가 원문과 일치하는가**를 분리한 구조입니다. 그러나 이 그림의 문맥 길이 배수와 호출 수는 작동 예시이지 전체 평가 평균이 아닙니다.

### Figure 6 — 모델에 따라 같은 기능이 다르게 사용됨〔p.9〕

**[그림이 보여주는 것]** Online Context Compact의 과제 활성화율은 Sol 92.2%, Opus 33.3%이고, Action Fusion의 활성화 과제당 평균 횟수는 70.58 대 13.54입니다. 횟수 축은 로그 스케일입니다. :chatgpt-content-reference{index="82"}

*용어 — 활성화율: 해당 기능이 한 번 이상 작동한 과제 비율; 활성화 강도: 작동한 과제에서의 평균 작동 횟수; 로그 스케일: 같은 간격이 같은 증가량이 아니라 같은 배율을 나타내는 축.*

**[해석]** 다른 모델에서도 효율 개선은 가능하지만, 동일 코드가 동일 행동을 유발하지는 않습니다. 저자들은 단일 모델에서의 개발을 가능한 원인으로 제안하지만, 그림만으로 원인을 확정할 수는 없습니다〔p.8, §3.4〕. :chatgpt-content-reference{index="83"}

### Figure 7 — 상호 보완성의 단서이지 증명은 아님〔p.10〕

**[그림이 보여주는 것]** ObservationPack의 활성화율은 단독 80.4%에서 전체 56.9%로, 활성화 과제당 횟수는 8.59에서 1.66으로 감소합니다. 반면 해당 부분집합의 점수당 비용 개선은 더 크게 나타납니다. :chatgpt-content-reference{index="84"}

**[해석]** 다른 메커니즘이 먼저 관측량을 줄여 ObservationPack의 필요성이 감소했을 가능성이 있습니다. 하지만 비교 대상 과제가 같지 않아, 각 모듈의 기여를 더하거나 ‘시너지 크기’를 추정하면 안 됩니다.

### Figure 8 — 자동 연구는 단조로운 상승 과정이 아님〔p.10〕

**[그림이 보여주는 것]** Action Fusion은 27개 기록된 반복을 거쳐 개발되며, 표시된 10개 탐색 배치의 활성화율과 점수는 오르내립니다. 9번째 배치 점수 88.9보다 선택된 10번째 점수 87.0이 낮지만, 10번째 활성화율은 100%입니다. 토큰 11.5% 감소는 모든 후보 지점이 작동한다고 가정한 추정입니다. :chatgpt-content-reference{index="85"}

**[해석]** 이 그림은 단일 점수 최고값보다 기능 사용과 과제 점수를 함께 고려했다는 사례입니다. 동시에 탐색 반복 횟수가 증가하면 성능이 계속 향상된다는 해석을 부정합니다.

---

## 8. 문서가 답하지 않는 질문

| 아직 답하지 않는 질문 | 필요한 추가 정보·실험 |
|---|---|
| 성능 감소를 어디까지 허용했는가? | 능력 지표별 허용오차 수치, 절대·상대오차 구분, 전체 구성의 최종 수용 기준〔p.3 §2.1〕 |
| 40개 최종 시험 과제만 보면 결과가 유지되는가? | 11개·40개 집합의 별도 점수·비용·과제별 결과〔p.6 §2.5〕 |
| 저렴해진 이유 중 얼마나 ‘필요한 일을 덜 함’이 차지하는가? | 성공·실패별 비용, 검증 수행량, 조기 종료·재시도·진행량 분석〔p.7 Table 3〕 |
| 발췌에서 중요한 증거가 누락되는 비율은 얼마인가? | 의미적 누락 검사, 원문 재조회 성공률, 누락에 따른 오진 분석〔p.5 §2.4〕 |
| 탐색 규모의 어느 부분이 일반화를 개선했는가? | 동일 예산에서 환경 수·다양성·아이디어 수·계열별 깊이를 분리한 비교〔p.12 §5.1〕 |
| 네 메커니즘이 수동 설계나 단순 탐색보다 나은가? | 같은 예산의 수동 설계, 무작위 탐색, 간단한 반복 개선 대조군 |
| 개발 비용을 언제 회수하는가? | 총 탐색비, 유지보수비, 과제당 절감액, 배포 실행량 |
| 실행 기록으로 기반 모델을 학습하면 일반화가 개선되는가? | 동일 학습량·연산량 조건의 모델 재학습 및 독립 평가 |

위 질문은 원문의 수용 규칙, 평가 분리, 로그 검증, 향후 연구 설명에서 남는 공백을 정리한 것입니다. :chatgpt-content-reference{index="86"} :chatgpt-content-reference{index="87"} :chatgpt-content-reference{index="88"} :chatgpt-content-reference{index="89"}

**[외부 자료 보충]** 공식 프로젝트 글은 인간이 초기 원칙을 제공하고 탐색 방향을 걸렀으며, 살아남은 후보의 코드를 이해하고 정리하는 단계에 다시 참여했다고 설명합니다. 따라서 이 연구를 **처음부터 끝까지 인간 개입이 전혀 없는 연구 자동화**로 표현해서도 안 됩니다. 이는 첨부 논문보다 공식 글에서 더 명시적으로 설명된 부분입니다. :chatgpt-content-reference{index="90"}

---

## 9. 2020년 이후 관련 연구와의 비교

### 9.1 역사적 위치

ReAct는 추론과 행동을 교차시키는 에이전트 실행 방식을 정리했고, Reflexion은 가중치를 바꾸지 않고 언어적 피드백을 기억해 다음 시도를 개선하는 접근을 제시했습니다. SoL-Pi는 이와 달리 **개별 문제의 다음 시도보다, 여러 문제에서 재사용할 실행 규칙을 개선하는 것**에 초점을 둡니다〔ReAct p.1; Reflexion p.1〕. :chatgpt-content-reference{index="91"}

아래 비교는 **각 연구의 저자 보고를 요약한 뒤 SoL-Pi와의 차이를 해석한 것**입니다. 기반 모델, 과제, 계산 예산, 비용 정의가 다르므로 연구 간 숫자를 직접 순위화할 수 없습니다.

### 9.2 방법·결과·일반화 비교

| 연구 | 저자가 보고한 방법·결과 | SoL-Pi와의 차이 및 일반화 관점 |
|---|---|---|
| **SWE-agent, 2024** | 파일 탐색·편집·도구 피드백을 모델에 맞게 설계한 ACI를 제안〔Figure 1〕. :chatgpt-content-reference{index="92"} | 모델 밖 인터페이스가 성능을 좌우한다는 선행 근거. SoL-Pi의 차이는 이를 **자동 탐색과 비용 제약**의 대상으로 삼는 데 있음 |
| **ADAS, 2024/2025** | 에이전트를 코드로 표현하고, 축적된 후보 아카이브를 이용해 새로운 설계를 탐색하며 과제·모델 전이를 평가〔Figure 2, Figure 3〕. :chatgpt-content-reference{index="93"} | 코드 기반 자동 설계와 전이 자체는 SoL-Pi 이전에도 존재. SoL-Pi는 장시간 하네스의 반복 비용에 더 특화 |
| **GEPA, 2025/2026** | 실행 기록을 자연어로 반성하고 프롬프트를 수정하며, 파레토 후보의 보완적인 개선을 결합〔p.1〕. :chatgpt-content-reference{index="94"} | 주된 탐색 대상이 프롬프트인 반면 SoL-Pi는 도구 실행·보관·검증 코드까지 변경. 프롬프트 최적화만으로 같은 효과가 가능한지도 대조할 필요 |
| **AgentDiet, 2025/2026** | 불필요·중복·만료된 실행 기록을 제거하여 입력 토큰 39.9–59.7%, 논문 정의 비용 21.1–35.9% 감소를 보고〔초록; Figure 2〕. :chatgpt-content-reference{index="95"} | 문맥 낭비 제거 자체는 새로운 개념이 아님. SoL-Pi는 여러 실행 경계에서 개선을 자동 발견·결합한다는 점이 차이 |
| **ACON, 2025** | 관측·이력 압축 지침을 최적화하고 작은 압축 모델로 증류; 초판은 최대 문맥 토큰 26–54% 감소를 보고〔Figures 1–2, Tables 1–2〕. :chatgpt-content-reference{index="96"} | ‘최대 문맥 길이’는 SoL-Pi의 ‘누적 토큰 트래픽’과 다른 지표. 작은 모델에서의 능력 개선과 압축기 증류는 SoL-Pi의 유용한 확장 방향 |
| **Darwin Gödel Machine, 2025; Hyperagents, 2026** | 전자는 자기 코드 수정과 평가를 반복하며 후보 아카이브를 확장하고, 후자는 개선을 생성하는 메타 수준 코드까지 수정 대상으로 확장〔각 Figure 1〕. :chatgpt-content-reference{index="97"} | SoL-Pi는 효율적인 하네스 발견에 집중하며, 그 효율이 다음 세대 연구를 누적 가속하는 효과는 아직 미실증 |
| **Meta-Harness, 2026** | 과거 코드·점수·실행 기록을 파일시스템으로 탐색하는 외부 루프; 200개 수학 문제의 5개 모델 평가에서 검색기 없는 기준 대비 평균 +4.7점〔Table 6〕. :chatgpt-content-reference{index="98"} | 자동 하네스 탐색과 다중 모델 전이의 직접 선행 연구. 다만 코딩 실험은 동일 89개 과제에서 탐색·평가했다고 명시하므로, 평가 종류별 프로토콜을 나눠 읽어야 함 |
| **AHE, 2026** | 구성요소·실행 경험·수정 결정을 관측 가능한 기록으로 연결〔Figure 2〕; Terminal-Bench 69.7→77.0%, 다른 벤치마크인 SWE-bench에서는 75.2→75.6%〔Tables 1–2〕. :chatgpt-content-reference{index="99"} | 개발 환경의 큰 향상이 새 환경에서는 작아질 수 있음을 보여줌. SoL-Pi 역시 비용 전이와 점수 전이를 별도로 검증해야 함 |
| **RHI, 2026** | 과제별 에이전트 루프를 프롬프트로 명세하고, 수정 이력에 대한 쌍대 피드백으로 개선; 30개 합성 연구 과제에서 비용 최대 60% 감소를 보고〔Figure 1〕. :chatgpt-content-reference{index="100"} | 과제별 특화와 재사용 가능한 범용 하네스는 다른 목표. LLM 심사 기반 선호 평가도 SoL-Pi의 실행 검증 결과와 직접 비교 불가 |
| **Rethinking the Evaluation of Harness Evolution for Agents, 2026** | 피드백·추론 예산을 맞춰 단순 재시도와 비교하고, 45/10/34개 학습·검증·시험 분리를 적용; 시험 평균은 67.7→68.3〔Tables 2–3〕. :chatgpt-content-reference{index="101"} | SoL-Pi를 평가할 때 핵심적인 비판 기준. 하네스 개선의 효과와 추가 탐색 예산의 효과를 구분해야 함 |
| **ModularRSI, 2026년 9월** | 성공·실패 궤적을 대조하고 다섯 기능 모듈을 독립 개선한 뒤 통합; 평가 벤치마크와 분리한 2,000개 개발 과제 및 동결 하네스 전이를 보고〔Figure 1, Table 3〕. :chatgpt-content-reference{index="102"} | SoL-Pi의 환경 분리·독립 개발과 가장 가까운 동시기 접근. 대조 궤적 분석과 명시적 수정 범위 제한을 결합할 가치가 있음 |
| **RRSI, 2026년 9월, v2** | 후보 수정량을 제한하고 누출·평가 잡음·불필요한 복잡성을 제어; 8개 벤치마크에서 분포 밖 성능 개선을 보고〔Figures 1–3〕. :chatgpt-content-reference{index="103"} | ‘탐색을 크게 하는 것’ 외에 ‘탐색이 과적합하지 않도록 제한하는 것’도 중요함. SoL-Pi의 미공개 허용오차와 후보 선택 안정성을 보완할 방향 |

*용어 — ACI: 에이전트와 컴퓨터 사이의 명령·피드백 인터페이스; 아카이브: 이전 후보와 결과를 보존한 저장소; 증류(distillation): 큰 모델의 처리 방식을 작은 모델에 학습시키는 과정; 메타 수준: 문제를 푸는 절차가 아니라 그 절차를 개선하는 절차; 정규화: 평가 자료의 우연한 특징에 과도하게 맞추지 않도록 변경량·복잡성 등을 제어하는 원리; 분포 밖(OOD): 개발 자료와 과제 특성이나 환경이 달라진 평가 조건.*

### 9.3 이 비교에서 도출되는 평가

**[해석]** SoL-Pi의 차별성을 “최초의 자동 하네스 개선” 또는 “최초의 모델 간 전이”라고 두기는 어렵습니다. 더 타당한 기여는 **다양한 실행 환경에서 비용 낭비를 탐색하고, 서로 다른 실행 경계를 다루는 네 메커니즘을 선택·통합하여 장시간 작업의 비용–품질 관계를 측정한 것**입니다. 이는 ADAS·Meta-Harness의 자동 설계 방향과 AgentDiet·ACON의 문맥 효율화 방향을 연결합니다. :chatgpt-content-reference{index="104"} :chatgpt-content-reference{index="105"}

최근 ModularRSI와 RRSI까지 함께 보면, 향후 경쟁의 중심은 단순한 탐색 횟수보다 **개발·평가 분리, 재사용 가능한 실패 원인 식별, 모듈 간 간섭 제어, 잡음에 강한 선택**으로 이동할 가능성이 있습니다. 이는 제 전망이며, 동시기에 발표된 연구들이 SoL-Pi의 영향을 받아 나왔다고 주장하는 것은 아닙니다. :chatgpt-content-reference{index="106"}

---

## 10. 결론과 후속 연구 방향

### 10.1 저자들이 제시한 시사점과 계획

**[저자 보고]** 저자들은 다음 네 방향을 제시합니다. 첫째, 많은 실행 환경과 연구 아이디어에 하네스를 노출시키는 **‘하네스 사전학습’**입니다. 둘째, 여러 기반 모델의 실행 기록을 이용하는 다중 백엔드 개발입니다. 셋째, SoL-Pi를 다음 자동 연구 주기의 시작점으로 삼는 재귀적 효율 개선입니다. 넷째, 동일 예산에서 탐색 폭·깊이의 효과와 스케일링 관계를 체계적으로 조사하는 것입니다〔p.12, §5.1〕. :chatgpt-content-reference{index="107"}

*용어 — 하네스 사전학습: 이 논문에서는 여러 과제의 실행 경험으로 하네스 코드·규칙을 개선한다는 비유이며, 언어모델 가중치의 사전학습과 동일하지 않습니다; 백엔드: 하네스가 호출하는 기반 모델·제공자.*

### 10.2 추가 제안 1: ‘성능 유지’를 사전 정의된 비열등성으로 검증

**[제안]** 동일 과제에서 기준·후보를 여러 번 실행하고, 과제별 점수 차이의 불확실성을 평가해야 합니다. 후보 선택은 개발·검증 집합에서 끝내고, 최종 시험 집합은 한 번의 최종 판단에만 사용해야 합니다.

설명적으로 다음 조건을 사용할 수 있습니다.

```math
\text{LCB}_{95\%}(\Delta Q) > -\epsilon.
```

$\Delta Q$는 후보와 기준의 평균 성능 차이, $\epsilon$은 사전에 정한 허용 감소폭, $\text{LCB}_{95} % $는 그 차이에 대한 95% 신뢰 하한입니다. **이는 제가 제안하는 평가 조건이지, SoL-Pi가 수행한 검정이 아닙니다.**

저장소 내부 과제들의 유사성을 고려하여 저장소 단위로 재표집하고, 11개 수용 집합과 40개 최종 시험 집합을 분리 보고하는 것이 우선입니다.

*용어 — 재표집: 관측 자료에서 과제나 저장소를 반복 추출해 결과의 변동성을 추정하는 방법.*

### 10.3 추가 제안 2: 일반화를 모델·과제·시간 축에서 동시에 평가

**[제안]** 새로운 과제만 남겨 두는 것으로는 충분하지 않습니다. 개발에 쓰지 않은 모델 계열, 저장소, 언어, 검증기 종류, 이후 시점에 만들어진 문제를 각각 남겨 두고 평가하는 설계가 필요합니다.

이를 설명하는 개발 목표는 다음과 같습니다.

$$
\min_h\;
\mathbb{E}_{(b,d)\sim\mathcal D_{\text{dev}}}
\left[C(h;b,d)\right]
\quad
\text{subject to}\quad
Q(h;b,d)\ge Q(h_0;b,d)-\epsilon_{bd}.
$$

$b$는 기반 모델, $d$는 과제 영역, $\mathcal D_{\text{dev}}$는 개발용 모델–영역 분포, $C$는 비용, $Q$는 품질, $\epsilon_{bd}$는 해당 조합의 허용 감소폭입니다. 최종 일반화 평가는 이 최적화에 사용하지 않은 모델·영역에서 수행해야 합니다.

이 접근은 평균 비용만 낮추면서 특정 언어나 작업군의 성능을 크게 희생하는 후보를 걸러내는 데 유용할 것입니다.

### 10.4 추가 제안 3: 전체 결합보다 ‘언제 어떤 기능을 쓸 것인가’를 연구

**[제안]** 네 메커니즘의 사용·미사용을 모두 조합하면 16개 구성이 됩니다. 이를 같은 과제·반복 조건에서 비교하면, 단독 효과와 결합 효과를 지금보다 명확히 분리할 수 있습니다.

특히 Table 4의 Opus 결과는 **항상 전체 구성을 켜는 방식보다 과제·모델에 맞춘 선택적 사용이 유리할 수 있음**을 시사합니다. 다만 선택 규칙도 시험 점수를 보고 정하는 것이 아니라 개발 자료에서 확정해야 합니다〔p.9, Table 4〕. :chatgpt-content-reference{index="108"}

### 10.5 추가 제안 4: ‘증거의 정확성’에서 ‘증거의 충분성’으로

**[제안]** reducer 평가에는 인용 일치율뿐 아니라 중요한 오류 줄의 누락률, 원문 재조회 성공률, 발췌를 보고 내린 진단의 정확성, 재조회 지연을 포함해야 합니다. ObservationPack 역시 저장된 정보를 얼마나 자주, 얼마나 정확하게 회수하는지 측정해야 합니다.

이는 일반화에 직접 연결됩니다. 익숙하지 않은 테스트 출력이나 새로운 도구에서는, 개발 환경에서 중요하지 않았던 정보가 핵심 단서가 될 수 있기 때문입니다. 현재 검증은 원문과의 일치에 강점을 두지만, 모든 중요 증거의 포함 여부까지 판정하지는 않습니다〔p.5, §2.4〕. :chatgpt-content-reference{index="109"}

### 10.6 추가 제안 5: 탐색의 경제성과 재귀적 효과를 직접 측정

**[제안]** 배포 비용 절감만 아니라 전체 연구 비용을 포함해야 합니다. 단순한 손익분기 실행 횟수는 다음과 같이 정의할 수 있습니다.

```math
n_{\text{break-even}}
=
\frac{C_{\text{search}}}{c_0-c_1},
\qquad c_0 > c_1.
```

$C_{\text{search}}$는 개선안을 찾는 데 든 총 추가 비용, $c_0$와 $c_1$은 기준·개선 하네스의 동일 품질 조건에서의 평균 실행비, $n_{\text{break-even}}$은 탐색비를 회수하는 데 필요한 실행 횟수입니다. 유지보수비와 환경 운영비를 제외한 단순화이며, **논문은 필요한 총 탐색비를 제시하지 않아 이 값을 계산할 수 없습니다**〔p.12, Search Coverage and Cost〕. :chatgpt-content-reference{index="110"}

재귀적 개선을 검증하려면 여러 세대에 걸쳐 같은 총 연구 예산을 주고, 새로 발견한 유효 메커니즘 수·독립 시험 성능·전체 연구비가 어떻게 변하는지 보여줘야 합니다.

### 최종 판단

**SoL-Pi의 가장 강한 성과는 기반 모델의 지능 향상 자체가 아니라, 재사용 가능한 실행 규칙을 통해 일부 환경에서 유용한 작업당 비용을 낮춘 점입니다.** 반면 전체 구성의 점수 감소, 모델별 성능 구성의 사후 선택, 최종 시험 부분집합의 미분리 보고 때문에 ‘일반화 성능이 향상되었다’는 표현은 적용 범위를 제한해야 합니다〔Tables 2–4〕. :chatgpt-content-reference{index="111"} :chatgpt-content-reference{index="112"}

앞으로 가장 중요한 질문은 **“얼마나 많은 자동 연구를 실행했는가?”가 아니라 “같은 총예산과 검증된 품질 조건에서, 얼마나 많은 새로운 과제·모델로 개선이 이전되는가?”**입니다.

---

## 참고자료 — 본문 분석에 사용한 자료의 전체 제목

| 자료 | 제목·출처 |
|---|---|
| 분석 대상 | **Haozhe Liu et al. — SoL-Pi: Recursively Scaling Auto-Research Loops for Efficient Agent Harness.** arXiv:2609.20519v1, 2026; 첨부 PDF. :chatgpt-content-reference{index="113"} |
| 공식 보충 자료 | **SoL-Pi: Scaling Auto-Research Loops for Efficient Agent Harnesses.** NVIDIA/NVlabs 공식 프로젝트 글. :chatgpt-content-reference{index="114"} |
| ReAct | **Shunyu Yao et al. — ReAct: Synergizing Reasoning and Acting in Language Models.** arXiv:2210.03629. :chatgpt-content-reference{index="115"} |
| Reflexion | **Noah Shinn et al. — Reflexion: Language Agents with Verbal Reinforcement Learning.** arXiv:2303.11366. :chatgpt-content-reference{index="116"} |
| SWE-agent | **John Yang et al. — SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering.** arXiv:2405.15793. :chatgpt-content-reference{index="117"} |
| ADAS | **Shengran Hu, Cong Lu, Jeff Clune — Automated Design of Agentic Systems.** arXiv:2408.08435. :chatgpt-content-reference{index="118"} |
| GEPA | **Lakshya A. Agrawal et al. — GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning.** arXiv:2507.19457. :chatgpt-content-reference{index="119"} |
| AgentDiet | **Yuan-An Xiao et al. — Reducing Cost of LLM Agents with Trajectory Reduction.** arXiv:2509.23586; Proceedings of the ACM on Software Engineering/FSE 2026. :chatgpt-content-reference{index="120"} |
| ACON | **Minki Kang et al. — ACON: Optimizing Context Compression for Long-horizon LLM Agents.** arXiv:2510.00615; 본문에서 인용한 절감 범위는 v1 기준. :chatgpt-content-reference{index="121"} |
| DGM | **Jenny Zhang et al. — Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents.** arXiv:2505.22954. :chatgpt-content-reference{index="122"} |
| Hyperagents | **Jenny Zhang et al. — Hyperagents.** arXiv:2603.19461. :chatgpt-content-reference{index="123"} |
| Meta-Harness | **Yoonho Lee et al. — Meta-Harness: End-to-End Optimization of Model Harnesses.** arXiv:2603.28052. :chatgpt-content-reference{index="124"} |
| AHE | **Jiahang Lin et al. — Agentic Harness Engineering: Observability-Driven Automatic Evolution of Coding-Agent Harnesses.** arXiv:2604.25850. :chatgpt-content-reference{index="125"} |
| RHI | **Hyunin Lee et al. — Recursive Harness Self-Improvement.** arXiv:2607.15524. :chatgpt-content-reference{index="126"} |
| 평가 비판 연구 | **Yike Wang et al. — Rethinking the Evaluation of Harness Evolution for Agents.** arXiv:2607.12227. :chatgpt-content-reference{index="127"} |
| ModularRSI | **Siwei Wu et al. — ModularRSI: Modular and Generalizable Recursive Harness Self-Improvement.** arXiv:2609.14857. :chatgpt-content-reference{index="128"} |
| RRSI | **Peng Xia et al. — RRSI: Regularized Recursive Self-Improvement of Agent Harnesses.** arXiv:2609.24972v2. :chatgpt-content-reference{index="129"} |
