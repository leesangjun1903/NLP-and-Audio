# ORCA-bench: How Ready Are Language Model Agents for Oncall?

> **⚠️ 주의**: 본 논문은 arXiv:2607.28545v1 (2026년 7월 30일 제출, Preprint)으로, 아직 동료 심사(peer review)를 통과하지 않은 상태입니다. 모든 수치는 논문 원문에 근거하며, 불확실한 내용은 명시적으로 표시합니다.

---

## 1. Executive Summary (10문장 이내)

ORCA-bench는 LLM 에이전트의 온콜(oncall) 근본 원인 분석(RCA) 능력을 평가하는 최초의 프로덕션 충실도(production-fidelity) 벤치마크이다.  
OpenTelemetry로 계측된 19개 마이크로서비스(Astronomy Shop)를 6일간 운영하며, Prometheus·Jaeger·OpenSearch를 Grafana API를 통해 실제 원격 측정 인터페이스로 제공한다.  
총 1,079개의 RCA 태스크는 보고 특이성(Easy/Medium/Hard), 탐지 시간(TTD: 15분~24시간), 동시 장애 시나리오를 체계적으로 변화시켜 구성되었다.  
정답 증상은 전문 SRE 2인이 수작업으로 검증하였고, LLM-as-judge는 인간 재채점과 Cohen's $\kappa_w = 0.90$의 높은 일치도를 달성하였다.  
5개 프론티어 에이전트(Claude Opus 4.7, Claude Sonnet 4.6, GPT-5.5, GLM-5, DeepSeek-V4-Pro) 평가 결과, Medium 난이도에서 최고 RCA Accuracy는 **25.3%**, Hard에서는 **10.0%**에 불과하다.  
GLM-5는 40%의 incident report에서 근거 없는 루트 코즈를 환각(hallucination)하였다.  
소스 코드 접근 제거 시 모든 모델의 모든 지표가 하락하며, 이는 코드 접근이 RCA에 필수적임을 보여준다.  
실제 프로덕션 시스템은 본 벤치마크보다 수 배 이상 복잡하므로, 이 결과는 격차의 하한(lower bound)에 해당한다.  
결론적으로 현재의 프론티어 코딩 에이전트는 프로덕션 온콜 업무를 독립적으로 수행할 준비가 되어 있지 않다.

### 1-1. 연구의 목적과 필요성

**목적**: 실제 프로덕션 온콜 환경에서 LLM 에이전트의 RCA 수행 능력을 측정하는 공개 벤치마크 구축.

**필요성**:
- 기존 SWE 벤치마크(SWE-bench 등)는 정적 저장소(frozen repo) + 명확한 버그 리포트 기반이라, "실행 중인 분산 시스템 + 모호한 자연어 사용자 불만" 환경과 근본적으로 다름
- 기존 SRE 벤치마크(AIOpsLab, ITBench, OpenRCA, SREGym)는 원격 측정 인터페이스, 소스 코드 접근, 보고 특이성 변화, TTD 변화, 인간 검증 중 **적어도 하나**가 결여됨 (Table 1, p.2)
- 실제 온콜 SRE는 사건 발생 수 시간 후, "checkout이 망가졌어요" 같은 모호한 보고를 받고 메트릭·로그·트레이스·소스코드를 동시에 분석해야 함
- 이 도메인별 격차를 측정하지 않으면, 에이전트의 실제 배포 위험성을 과소평가할 수 있음

---

## 2. 핵심 주장과 근거 표

| 핵심 주장 | 근거 / 수치 | 출처 위치 |
|-----------|------------|----------|
| 프론티어 에이전트는 온콜에 준비되지 않음 | Medium RCA Accuracy 최고 25.3%, Hard 10.0% | Fig. 1, p.1; Sec. 5, p.7 |
| Hard 태스크에서 성능이 급격히 하락 | Easy→Hard 간 RCA Accuracy 19~50%p 하락 | Fig. 5, p.8; Sec. 5 Finding #2 |
| 환각(hallucination)이 심각한 수준 | GLM-5 40.2%, DeepSeek-V4-Pro 7.2% | Fig. 1, p.1 |
| 소스 코드 접근이 성능에 필수적 | 코드 제거 시 RCA Accuracy 9~16%p 하락, 환각 대폭 증가 | Fig. 6, p.9; Sec. 5 Finding #3 |
| 원격 측정 호출의 26~40%가 실패 | 모델별 실패율 25.8%~40.1% | Fig. 8(b), p.10 |
| ORCA-bench 결과는 격차의 하한 | 50GB/6일 고정 코드베이스, 공개 시스템, 태스크 격리 조건 | Sec. 6 Conclusion, p.10 |
| Claude Fable 5가 가장 높은 성능 | Verified 서브셋에서 RCA depth 58.2%, RCA accuracy 40.6% | Sec. 5, p.7; Fig. K.1, p.33 |
| LLM-as-judge의 신뢰성 검증 완료 | Cohen's $\kappa_w = 0.90$, Spearman $\rho = 0.92$ | Table 3, p.7 |
| 동시 발생 장애가 가장 어려운 시나리오 | Day 6: 3개 모델이 각각 하나씩 다른 루트 코즈만 식별 | Sec. 5 Finding #2, p.8 |
| GPT-5.5가 원격 측정 효율 최고 | 가장 적은 명령(28.2개), 가장 낮은 실패율(25.8%) | Fig. 8(a)(b), p.10 |

### 2-1. 해결하고자 하는 문제, 제안하는 방법, 모델 구조, 성능 향상 및 한계

#### 해결하고자 하는 문제

LLM 에이전트의 온콜 RCA 능력을 측정할 수 있는 **프로덕션 충실도 벤치마크의 부재**. 기존 벤치마크의 공통 한계:
- 소스 코드 접근 없음
- 실제 원격 측정 인터페이스(Grafana API 등) 미제공
- 사용자 보고 특이성(issue specificity) 변화 미고려
- TTD(Time-to-Detection) 오프셋 미고려
- 인간 수준의 정답 검증 부재

#### 제안하는 방법 (벤치마크 구성 6단계)

**B3 태스크 매개변수화**:

$$\text{report time} = t_{\text{incident start}} + \text{offset}$$

$$\text{detection time} = t_{\text{incident start}} + \text{TTD}, \quad \text{TTD} \in \{15\text{min}, 1\text{h}, 8\text{h}, 24\text{h}\}$$

**B5 정답 루트 코즈 생성**:

$$\text{candidates} = \text{events active in } [\text{start of report day}, \min\{\text{detection time}, \text{end of report day}\})$$

전처리: 프론트엔드 증상 없는 이벤트 제거 → GPT-5.4로 각 후보의 plausible/implausible 판정

**E2 Per-rubric 점수**:

$$s_i \in \{0, 1, 2, 3\}$$

- $s = 0$: 루트 코즈를 완전히 놓침
- $s = 1$: 증상 확인만, 더 이상 탐색 없음
- $s = 2$: 관련 텔레메트리 일부 식별, 루트 코즈 불완전
- $s = 3$: 루트 코즈, 메커니즘, 지지 신호 모두 정확히 식별

**E3 태스크 레벨 메트릭**:

$$\text{RCA Accuracy} = \frac{|\{\text{tasks where all plausible root causes named}\}|}{|\text{incident tasks}|}$$

$$\text{RCA Depth} = \frac{1}{N}\sum_{i=1}^{N}\bar{s}_i \times \frac{100}{3} \%$$

여기서 $\bar{s}_i$는 태스크 $i$의 per-rubric 점수 평균

$$\text{Hallucination Rate} = \frac{|\{\text{non-empty reports naming no plausible root cause}\}|}{|\text{non-empty incident reports}|}$$

**판사 신뢰도**:

$$\kappa_w = \frac{\bar{p}_o - \bar{p}_e}{1 - \bar{p}_e} = 0.90 \text{ (quadratic-weighted Cohen's } \kappa)$$

#### 모델 구조

본 논문은 새로운 모델 아키텍처를 제안하지 않음. 평가 대상:
- **에이전트 하네스**: Terminus-2 (Merrill et al., 2026) — bash 터미널만 제공, 컨텍스트 압축 지원
- **평가 모델**: Claude Opus 4.7, Claude Sonnet 4.6, GPT-5.5, GLM-5, DeepSeek-V4-Pro (+ Verified 서브셋의 Claude Fable 5)
- **판사 모델**: GPT-5.4
- **환경**: OpenTelemetry Astronomy Shop (19개 마이크로서비스, 13개 언어)

#### 성능 향상 및 한계

**성능 (Code + Telemetry, N=884, Table I.1)**:

| 모델 | RCA Depth | RCA Accuracy | Hallucination |
|------|-----------|--------------|---------------|
| Claude Opus 4.7 | 48.5±1.1% | 28.6±1.5% | 12.6±1.1% |
| Claude Sonnet 4.6 | 46.7±1.1% | 30.9±1.6% | 14.8±1.2% |
| GPT-5.5 | 48.8±1.1% | 24.3±1.4% | 25.0±1.5% |
| DeepSeek-V4-Pro | 19.7±1.0% | 15.0±1.2% | 7.2±0.9% |
| GLM-5 | 27.7±1.0% | 17.6±1.3% | 40.2±1.6% |

**한계**:
- 11개 feature flag만 사용한 고정 장애 라이브러리 (실제 환경은 무한히 다양한 장애 유형)
- OpenTelemetry와 Astronomy Shop이 사전 학습 데이터에 포함되었을 가능성 (데이터 오염)
- 태스크 격리 — 이전 사건 기억 없음
- 읽기 전용 조사 — 수정 → 증상 해소 확인 루프 없음
- 단일 프롬프트 템플릿 + 단일 에이전트 하네스만 평가

---

## 3. 각 주장에 페이지/Figure/Table 번호 표시

| 주장 | 근거 위치 |
|------|----------|
| ORCA-bench가 유일하게 모든 특성을 지원 | **Table 1**, p.2 |
| Medium RCA Accuracy 최고 25.3% | **Fig. 1**, p.1; **Table I.3(b)**, p.28 |
| Hard RCA Accuracy 최고 10.0% | **Fig. 1**, p.1; **Table I.3(a)**, p.28 |
| GLM-5 hallucination 40.2% | **Fig. 1**, p.1; **Table I.1**, p.27 |
| LLM judge κw=0.90 | **Table 3**, p.7; **Fig. G.1**, p.25 |
| 소스 코드 제거 시 RCA Accuracy 9~16%p 하락 | **Fig. 6**, p.9; Finding #3, p.9 |
| 원격 측정 실패율 26~40% | **Fig. 8(b)**, p.10 |
| 동시 6개 장애 Day 6 시나리오 결과 | **Finding #2**, p.8; **App. C.2**, p.20 |
| Claude Fable 5 RCA depth 58.2% | **Fig. K.1**, p.33; Sec. 5, p.7 |
| 에이전트가 배경 노이즈에 산만해지는 실패 모드 | **Fig. 4**, p.8; Finding #1, p.7-8 |
| 코드 접근 사용 비율 Opus 4.7: 16%, GLM-5: 20% | **Fig. 7**, p.9 |

---

## 4. 저자 직접 보고 결과 vs. 내 해석

### 저자가 직접 보고한 결과

| 결과 | 수치 | 위치 |
|------|------|------|
| Medium 최고 RCA Accuracy | 25.3% (Claude Sonnet 4.6) | Fig. 1, p.1 |
| Hard 최고 RCA Accuracy | 10.0% (Claude Opus 4.7) | Fig. 1, p.1 |
| Hallucination 최악 모델 | GLM-5 40.2% | Fig. 1, p.1 |
| 코드 제거 RCA Accuracy 하락폭 | 9~16%p (모든 모델) | Fig. 6, Finding #3 |
| LLM judge 인간 일치도 | κw=0.90, ρ=0.92 (All) | Table 3 |
| Claude Fable 5 Verified 성능 | RCA depth 58.2±5.8%, accuracy 40.6±8.8% | Sec. 5, p.7 |
| GPT-5.5 텔레메트리 명령 수 | 28.2개 (최소) | Fig. 8(a) |
| DeepSeek-V4-Pro 텔레메트리 명령 수 | 75.5개 (최다) | Fig. 8(a) |

### 나의 해석 (저자 주장과 분리)

1. **모델 간 전략적 차이**: GPT-5.5는 명령 수가 가장 적으면서도(28.2개) RCA depth가 가장 높은(48.8%) 모델이다. 이는 "적게, 정확하게" 쿼리하는 전략이 "많이, 탐색적으로" 쿼리하는 전략보다 효과적임을 시사한다. 반면 DeepSeek-V4-Pro는 명령 수가 가장 많지만(75.5개) 성능이 낮아, 쿼리 효율성과 RCA 품질이 양의 상관관계를 가질 수 있다. (저자는 이 인과관계를 명시적으로 주장하지 않음)

2. **환각과 정확도의 역관계**: GLM-5는 환각률(40.2%)이 가장 높지만, DeepSeek-V4-Pro는 환각률(7.2%)이 낮으면서도 RCA Accuracy(15.0%)도 낮다. 즉 낮은 환각률이 반드시 높은 정확도를 의미하지 않으며, DeepSeek-V4-Pro는 "틀리게 말하기"보다 "아무것도 찾지 못하기"에 가깝다.

3. **벤치마크 자체의 데이터 오염 위험**: Astronomy Shop과 OpenTelemetry 문서가 공개되어 있어 사전 학습 데이터에 포함되었을 가능성이 높다. 저자는 이를 한계로 인정하지만, 이 오염이 각 모델별로 얼마나 다르게 영향을 미쳤는지는 측정되지 않았다.

4. **코드 접근의 비대칭 효과**: 코드 제거 시 GPT-5.5의 환각률이 25.0%→56.0%로 급증(+31%p)하는 반면, Incident Time Accuracy는 72.4%→72.4%로 거의 변화 없다(Table I.4). 이는 소스 코드가 시간 정보가 아닌 메커니즘 이해에 핵심적으로 기여함을 보여준다.

---

## 5. 통계적으로 취약한 부분과 비교 불가능한 수치

### ⚠️ 통계적으로 취약한 부분

| 항목 | 이유 |
|------|------|
| **Claude Fable 5 결과** | Verified 서브셋 32개 incident tasks만 평가 (n이 매우 작음). 95% CI가 매우 넓음: RCA accuracy $40.6 \pm 8.8\%$, depth $58.2 \pm 5.8\%$ | 
| **Hard 태스크 DeepSeek-V4-Pro RCA Accuracy** | $1.1 \pm 0.6\%$ (n=280) — 분산이 기준값의 54%에 달함 |
| **각 조건별 per-model 비교** | 다중 비교 보정(multiple comparison correction) 미실시. 20개 이상의 지표를 동시 비교 |
| **ORCA-bench Verified 서브셋** | 40개 태스크 (8 control, 8 easy, 12 medium, 12 hard) — stratified sampling이지만 절대 크기가 작아 난이도별 κw 신뢰구간이 넓음 |
| **Fig. G.1 일부 셀** | GLM-5 Easy n=7, DeepSeek-V4-Pro Easy n=7 — 비교 신뢰도 낮음 |

### 🚫 비교 불가능한 수치

| 비교 | 이유 |
|------|------|
| **ORCA-bench vs. AIOpsLab/ITBench/OpenRCA/SREGym 성능 수치** | 시스템, 태스크 구성, 난이도, 평가 방식이 완전히 다름 (Table 1). 직접 수치 비교 불가 |
| **모델별 절대 RCA Accuracy 비교** | 각 모델의 reasoning effort 설정이 다름 (Claude 계열: "medium", GLM-5/DeepSeek: 설정 불가) |
| **Telemetry-only vs. Code+Telemetry 비교** | 코드 접근 제거가 RCA Accuracy에 미치는 영향이 모델별로 상이한 이유를 인과적으로 분리 불가 |
| **Claude Fable 5 vs. 나머지 모델** | 서로 다른 태스크 세트(Verified 32개 vs. 전체 884개)에서 평가 |

---

## 6. 문서가 답하지 않는 질문

1. **왜 특정 모델이 다른 모델보다 더 효율적으로 텔레메트리를 쿼리하는가?** — 프롬프트 구조, 파인튜닝 데이터, 또는 추론 방식의 차이인지 분석되지 않음

2. **Astronomy Shop의 사전 학습 데이터 포함 여부와 그 영향량** — 모델별로 데이터 오염 정도가 다를 수 있으나 측정 불가

3. **체계적 프롬프트 엔지니어링, 구조화된 워크플로우, 또는 인과추론 RCA와의 하이브리드 접근이 성능을 얼마나 회복시킬 수 있는가?** — "Methods we did not study"로 명시만 하고 실험 없음

4. **지속적 학습(persistent memory) 에이전트가 SRE 업무에서 인간을 능가할 수 있는 조건은 무엇인가?** — 저자가 방향으로 제시하나 실험 없음

5. **태스크 격리가 아닌 연속 조사(sequential investigation)에서 성능은 어떻게 변하는가?** — 모든 태스크는 cold-start로 설계됨

6. **실제 프로덕션 시스템(비공개 코드베이스)에서 에이전트 성능은 얼마나 더 낮은가?** — 저자는 "하한"이라 주장하지만, 실제 격차는 측정되지 않음

7. **6일치 50GB 텔레메트리 중 에이전트가 실제로 조회한 비율** — 조회 효율성과 성능의 정량적 관계 불명확

8. **다국어 소스 코드(Go, Java, Python, Node.js, C# 등 13개)에 대한 모델별 이해 능력 차이** — 언어별 성능 분석 없음

9. **인간 SRE의 동일 벤치마크 성능** — 인간 기준점(human baseline) 부재로 격차의 절대적 의미 파악 불가

10. **feature flag 이외의 실제 장애 유형(예: 네트워크 파티션, 디스크 고갈)에 대한 일반화 가능성** — 11개 feature flag로만 구성

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.1): RCA Accuracy and Hallucination Rate

```
RCA Accuracy (Medium) ↑    RCA Accuracy (Hard) ↑    Hallucination Rate ↓
24.7  25.3  25.0            10.0   8.6   8.6          12.6  14.8  25.0  40.2  7.2
[Opus][Son.][GPT] [GLM][DS]  [Opus][Son.][GPT] [GLM][DS]  [Opus][Son.][GPT][GLM][DS]
```

**해석**: 
- Medium(현실적 입력) 기준 최고 성능이 25.3%에 불과 — 4개 중 3개는 놓침
- Hard에서는 10.0%로 급락 — 모호한 보고("사이트에 문제가 있어요") 환경의 처참한 성능
- 모델 간 성능 편차가 큼: Claude 계열이 RCA Accuracy 우위, GPT-5.5는 높은 환각률(25.0%)이 주목
- GLM-5는 환각률 40.2%로 신뢰성 측면에서 가장 위험 — 온콜 맥락에서 잘못된 근본 원인 진단은 장애 해결을 지연시킬 수 있음
- DeepSeek-V4-Pro는 환각률(7.2%)이 낮지만 Hard RCA Accuracy(5.4%)도 낮아 "안전하게 틀림"에 가까움

### Figure 5 (p.8): RCA Quality by Issue Specificity

```
RCA Accuracy ↑                    RCA Depth ↑
Control: ~80-90%                  Easy > Medium > Hard
Easy: ~25-60%                     모든 모델에서 동일 패턴
Medium: ~15-25%  
Hard: ~1-10%
```

**해석**:
- **Control 태스크(장애 없음 식별)**에서 모든 모델이 높은 성능 — "장애 없음"을 올바르게 보고하는 것은 상대적으로 쉬움
- Easy→Hard 방향으로 단조 감소하는 성능: 보고 특이성이 낮아질수록 가설 공간이 폭발적으로 증가($4.41 \pm 0.11$개의 평균 루트 코즈 vs. Easy의 $2.00 \pm 0.06$개)
- **Easy 태스크조차** 최고 모델이 58.7%에 불과 — 완전히 해결된 문제가 하나도 없음
- RCA Depth는 모든 난이도에서 RCA Accuracy보다 높음 — 에이전트가 부분적 진전은 하지만 완전한 식별에는 실패함을 시사

### Figure 6 (p.9): RCA Quality with and without Code Access

```
모델           RCA Acc. 변화   Inc. Time 변화   Hall. 변화
Opus 4.7:      +9             +3               -19
Sonnet 4.6:    +10            +4               -14
GPT-5.5:       +16            +7               -31
GLM-5:         +11            +4               -33
DeepSeek:      +10            +9               -6
```

**해석**:
- 소스 코드 접근이 **모든** 모델에서 RCA Accuracy와 환각률 모두에 명확하고 일관된 효과
- GPT-5.5는 코드 접근 제거 시 환각률이 25.0%→56.0%로 가장 크게 증가(+31%p) — 코드가 "확인 도구"로 가장 많이 활용됨을 시사
- Incident Time Accuracy는 상대적으로 변화 작음(Opus 4.7: +3%p) — 시간 정보는 텔레메트리에 내재, 코드와 무관
- DeepSeek-V4-Pro의 환각률 변화 폭이 가장 작음(-6%p) — 코드 접근 제거의 영향을 가장 덜 받지만, 기본 성능 자체가 낮음

### Figure 7 (p.9): Agent Behavior during Investigation (Opus 4.7 vs. GLM-5)

```
Opus 4.7: 텔레메트리 70% → 소스 코드 16% → 기타 12%
GLM-5:    텔레메트리 57% → 소스 코드 20% → 기타 61% (비율이 다른 노드)
```

**해석**:
- 두 모델 모두 70% 전후를 텔레메트리 관련 명령에 사용 — 소스 코드는 보조적 역할
- Opus 4.7은 **텔레메트리 쿼리(Telemetry query)**에 집중하는 선형 흐름(87% 자기 루프), GLM-5는 **기타(Other)** 카테고리 비율이 높아 더 산만한 탐색 패턴
- 에이전트는 소스 코드를 두 가지 방식으로 활용: ① 조사 초기 시스템 탐색, ② 텔레메트리 쿼리 후 가설 확인
- 소스 코드가 성능에 크게 기여함에도 실제 사용 비율이 낮다는 점은 **미활용 잠재력**을 시사

### Figure 8 (p.10): Telemetry Retrieval Efficiency and Accuracy

```
(a) 명령 수    (b) 실패율    (c) Any-match 정확도
DS:  75.5      38.1%        metrics/logs/traces 모두 낮음
GLM: 68.6      40.1%
Son: 52.3      32.2%
Opu: 41.5      28.1%
GPT: 28.2      25.8%        metrics/logs/traces 모두 최고
```

**해석**:
- **GPT-5.5**: 명령 수 최소(28.2) + 실패율 최저(25.8%) + 실제 증상 포착률 최고 → 가장 효율적인 텔레메트리 전략
- **DeepSeek-V4-Pro**: 명령 수 최다(75.5) + 실패율 38.1% → 많이 시도하지만 대부분 빈 결과 반환
- 실패의 주요 원인이 "error"가 아닌 "empty"(쿼리 성공했으나 데이터 없음) — 즉 쿼리 설계 능력의 문제
- Prometheus의 10K+ 타임 시리즈를 효과적으로 탐색하는 능력이 모델 간 핵심 차별 요인임을 시사

---

## 8. 결론: 시사점, 후속 연구 계획, 추가 연구 방향

### 8-1. 모델의 일반화 성능 향상 가능성

**저자가 제시한 일반화 관련 한계**:

1. **시스템 사전 지식 오염**: OpenTelemetry·Astronomy Shop이 공개 문서로 사전 학습에 포함되었을 가능성 높음. 실제 비공개 프로덕션 시스템에서는 성능이 더 낮을 것임 (Sec. 6, p.10)

2. **스케일과 동적 변화**: 50GB/6일 고정 코드베이스는 일별 테라바이트·지속 진화하는 실제 시스템과 근본적으로 다름 — 에이전트가 만나는 첫 실제 시스템에서 즉각 out-of-distribution 장애 직면

3. **태스크 격리 문제**: 모든 태스크는 cold-start. 인간 SRE는 수개월의 시스템 직관(어떤 서비스가 불안정한지, 어떤 알림이 노이즈인지) 축적 — 이를 재현할 persistent memory 메커니즘 부재

**일반화 성능 향상 가능성이 높은 방향** (내 해석):

| 방향 | 기대 효과 | 근거 |
|------|---------|------|
| **Persistent memory / 에피소드 메모리** | Cold-start 한계 극복. 저자가 "언젠가 인간 초과 가능" 방향으로 명시 | Sec. 6 "Single-task isolation" 한계 |
| **Action loop (수정 → 검증 루프)** | RCA 정확도 직접 향상 가능. 수정 성공/실패 신호가 가설 정제에 핵심 | Sec. 6 "Read-only investigation" 한계 |
| **구조화 워크플로우 + 인과추론 하이브리드** | LLM의 언어 이해력 + 그래프 기반 인과추론의 정밀성 결합 | Sec. 6 "Methods we did not study"; Pham et al., 2025 |
| **Retrieval-Augmented RCA** | 이전 사건 데이터베이스 활용으로 유사 장애 패턴 인식 향상 | 현재 에이전트에 없는 기능 |
| **쿼리 전략 훈련** | GPT-5.5의 고효율 쿼리 패턴을 다른 모델에 이식 가능성 | Fig. 8 분석 |

**Easy 태스크(58.7% 최고)조차 개선 여지**: Claude Fable 5가 Verified 서브셋에서 Easy는 크게 개선하지 못하고 Medium에서 주로 향상을 보임(Fig. K.1) — 이는 단순히 모델 크기를 키우는 접근이 아니라 RCA 특화 훈련이 필요함을 시사

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> **⚠️ 주의**: 아래 비교는 논문 내 인용 문헌과 일반적 연구 동향에 기반합니다. 논문에 인용되지 않은 연구에 대해서는 확실한 수치를 제시하지 않습니다.

#### 논문 내 인용된 관련 연구 비교

| 벤치마크 | 발표 | 핵심 차이점 | ORCA-bench 대비 |
|---------|------|------------|----------------|
| **SWE-bench** (Jimenez et al., 2024) | ICLR 2024 | 정적 저장소 + 실패 테스트 → 코드 패치 | RCA와 근본적으로 다른 작업 구조 |
| **AIOpsLab** (Shetty et al., 2024) | ACM SoCC 2024 | 커스텀 툴, 소스코드·TTD·issue specificity 없음 | 모든 핵심 차원 결여 |
| **ITBench** (Jha et al., 2025) | ICML 2025 | 텔레메트리 인터페이스 있으나 소스코드·issue specificity 없음 | 부분적 개선 |
| **OpenRCA** (Xu et al., 2025) | ICLR 2025 | 수동 루트 코즈 검증, CSV 원시 데이터 제공 | Grafana API 대신 CSV, TTD 미고려 |
| **SREGym** (Clark et al., 2026) | arXiv 2026 | 실시간 진단(live diagnosis), 소스코드·issue specificity 없음 | 역사적 텔레메트리 + specificity 변화 없음 |
| **RCAEval** (Pham et al., 2025) | ACM WebConf 2025 | LLM-free 인과추론 기반 RCA | 자연어 보고 처리 불가, 보완적 관계 |
| **TerminalBench** (Merrill et al., 2026) | arXiv 2026 | CLI 환경 일반 에이전트 평가 | ORCA-bench의 에이전트 하네스로 활용 |

#### ORCA-bench가 앞으로의 연구에 미치는 영향

1. **벤치마크 표준화**: 처음으로 원격 측정 인터페이스 + 소스 코드 + issue specificity + TTD + 인간 검증을 모두 갖춘 기준점 제공. 향후 SRE 에이전트 연구의 공통 평가 플랫폼이 될 가능성

2. **LLM-as-judge 방법론**: κw=0.90의 인간-LLM 일치도 달성 방법론이 다른 복잡한 작업 평가에 적용 가능한 사례 제공

3. **하한 프레임워크**: "벤치마크 결과 = 실제 격차의 하한"이라는 프레임이 AI 안전성 논의에서 방법론적 기여

4. **비공개 공개 데이터 공개**: https://hub.harborframework.com/datasets/orca-bench/ORCA-bench 를 통한 공개 재현 가능성

#### 앞으로 연구 시 고려할 점

**① 데이터 오염 제어 설계**: 
- 비공개 내부 시스템 기반 벤치마크 구축 필요
- 또는 동적으로 생성되는 시스템 구성으로 사전 학습 오염 최소화

**② 인간 기준선 확립**:
- 동일 태스크에서 숙련 SRE의 성능 측정 필요 (현재 논문에 없음)
- "인간 대비 X%" 형태의 의미 있는 비교 가능

**③ 장애 다양성 확대**:
- 11개 feature flag → 실제 운영 환경의 다양한 장애 유형(네트워크 파티션, 메모리 누수, 설정 드리프트 등)
- 외부 의존성 장애(클라우드 서비스, CDN 등) 포함

**④ 지속 학습 설계**:
- 순차적 사건 조사에서의 에이전트 성능 평가
- 이전 사건에서 학습하는 메모리 메커니즘 효과 측정

**⑤ 다중 에이전트 협력**:
- 실제 온콜은 종종 팀 단위로 진행. 다중 에이전트 협력이 단일 에이전트 대비 성능 향상 여부

**⑥ 비용-성능 트레이드오프**:
- GPT-5.5는 토큰 사용량(0.54M/trial)이 Claude Opus 4.7(1.67M/trial)의 1/3 — 동등한 성능에 훨씬 적은 비용. 실제 배포 시 경제성 분석 필요

**⑦ 행동 루프(action loop) 연구**:
- 읽기 전용 RCA → 수정 배포 → 결과 관찰의 전체 루프 평가
- 이는 RCA 정확도와 MTTR(Mean Time To Recover) 모두에 영향

---

## 참고문헌 (논문 내 인용 기준)

- **본 논문**: Gong, A., Choi, K., Agarwal, A., Schechner, J., Huang, R., Agrawal, R., Agarwal, A., & Dwivedi, R. (2026). *ORCA-bench: How Ready Are Language Model Agents for Oncall?* arXiv:2607.28545v1
- Shetty, M., et al. (2024). *Building AI Agents for Autonomous Clouds*. ACM SoCC 2024.
- Jha, S., et al. (2025). *ITBench: Evaluating AI Agents Across Diverse Real-World IT Automation Tasks*. ICML 2025.
- Xu, J., et al. (2025). *OpenRCA: Can Large Language Models Locate the Root Cause of Software Failures?* ICLR 2025.
- Clark, J., et al. (2026). *SREGym: A Live Benchmark for AI SRE Agents*. arXiv:2605.07161.
- Jimenez, C. E., et al. (2024). *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR 2024.
- Pham, L., et al. (2025). *RCAEval: A Benchmark for Root Cause Analysis of Microservice Systems*. ACM WebConf 2025.
- Merrill, M. A., et al. (2026). *TerminalBench: Benchmarking Agents on Hard, Realistic Tasks in CLI*. arXiv:2601.11868.
- Zheng, L., et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. NeurIPS 2023.
- Zhou, J. P., et al. (2025). *Graders Should Cheat: Privileged Information Enables Expert-Level Automated Evaluations*. EMNLP 2025.
- Yang, J., et al. (2025). *SWE-bench Multimodal*. ICLR 2025.
- Yang, J., et al. (2026). *SWE-smith: Scaling Data for Software Engineering Agents*. NeurIPS 2026.
- Bogomolov, E., et al. (2024). *Long Code Arena*. arXiv:2406.11612.
- **데이터셋**: https://hub.harborframework.com/datasets/orca-bench/ORCA-bench
- **시스템**: https://github.com/open-telemetry/opentelemetry-demo
