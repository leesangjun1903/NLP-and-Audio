
# Fara-7B: An Efficient Agentic Model for Computer Use

## 1. 논문의 핵심 주장과 주요 기여

**Fara-7B** 논문은 컴퓨터 사용 에이전트(CUA)의 **데이터 부족 문제**를 해결하기 위한 종합적인 솔루션을 제시합니다. 핵심 주장은 **고품질의 합성 데이터만으로도 소형 모델이 대형 모델과 경쟁 수준의 성능을 달성할 수 있다**는 것입니다.[1]

주요 기여는 다음과 같습니다:

**FaraGen 엔진**: 웹 기반 다단계 작업을 위한 스케일러블한 합성 데이터 생성 시스템으로, 약 **$1 당 검증된 궤적을 생산**합니다. 이는 기존의 인간 주석(human annotation) 기반 데이터 수집의 비용을 획기적으로 절감했습니다.[1]

**Fara-7B 모델**: 오직 스크린샷으로만 컴퓨터를 인식하며, 액세서빌리티 트리나 DOM 파싱 없이 직접 클릭 좌표를 예측하는 **순수 시각 기반 접근법**을 채택합니다.[1]

**WebTailBench 벤치마크**: 기존 벤치마크에서 과소 대표되는 현실 세계 작업(부동산 검색, 일자리 지원, 다중 항목 쇼핑 등)을 포함하는 새로운 평가 세트입니다.[1]

***

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

### 2.1 주요 문제점

컴퓨터 사용 에이전트 개발의 가장 큰 병목은 **고품질의 인간-컴퓨터 상호작용 데이터의 부족**입니다. LLM은 인터넷의 풍부한 텍스트 데이터로 학습했지만, CUA 궤적(trajectory)에 대한 비교 가능한 대규모 데이터셋이 존재하지 않습니다. 인간 주석가를 통한 수집은 각 작업이 수십 단계를 포함하므로 극도로 비용이 높습니다.[1]

### 2.2 제안 방법: FaraGen 파이프라인

**FaraGen**은 3단계 폐쇄형 루프 시스템입니다:[1]

#### 2.2.1 작업 제안 (Task Proposal)

세 가지 전략으로 현실적인 작업을 생성합니다:

1. **목표 URL 기반 작업 제안**: ClueWeb22 및 Tranco URL 인덱스를 활용하여 전자상거래, 여행, 예약 등 특정 카테고리의 작업을 생성합니다.[1]

2. **에이전트 기반 URL 탐색**: 무작위로 샘플링된 URL에서 다중모달 LLM 에이전트가 실시간으로 웹사이트를 탐색하며 작업을 생성합니다. 이 방식은 약 67%의 학습 작업을 생성하며, 일반적으로 복잡도가 낮습니다.[1]

3. **범례 작업 제안**: 기존 작업을 템플릿으로 변환하여 엔티티를 변경하고 다양한 재소 및 소매업체에서 유사한 작업을 생성합니다.[1]

#### 2.2.2 작업 해결 (Task Solving)

**Magentic-One** 멀티에이전트 프레임워크를 확장하여 작업을 해결합니다:[1]

**오케스트레이터 에이전트**: 계획 수립, WebSurfer 에이전트의 진행 상황 모니터링, 그리고 **임계점(critical points)** 감시를 담당합니다. 임계점은 다음과 같은 상황을 포함합니다:[1]

- 민감한 정보(로그인, 결제 정보) 입력
- 인간 수준의 소통(이메일, 채용 지원)
- 되돌리기 어려운 조치(항공편 예약, 테이블 예약)

오케스트레이터는 매 단계마다 다음과 같은 5가지 진단 필드를 유지합니다:

| 필드 | 설명 |
|------|------|
| `is_at_critical_point` | 민감/불가역적 조치 직전 여부 |
| `is_satisfied` | 작업 완료 여부 |
| `last_action_successful` | 의도된 작업이 예상 결과로 이어졌는지 |
| `is_in_loop` | 반복 행동 감지 |
| `next_steps` | WebSurfer를 위한 자연어 지시사항 |

**WebSurfer 에이전트**: Set-of-Marks(SoM) 에이전트로서 접근성 트리(Ax Tree) 기반으로 작동하여, 오케스트레이터의 지시에 따라 구체적인 브라우저 조작을 수행합니다.[1]

**사용자 시뮬레이터**: 임계점에서 다중 턴 상호작용을 가능하게 하며, 사용자 응답을 시뮬레이션하여 궤적의 복잡성과 현실성을 증가시킵니다.[1]

**최적화 기법** (표 4 참고): 더 강력한 모델 사용(o3, GPT-5)이 약 50%의 성능 향상을, 맥락 구성 개선 및 오류 허용성이 나머지 50%를 기여합니다.[1]

#### 2.2.3 궤적 검증 (Trajectory Verification)

단일 검증자로는 충분하지 않으므로, 세 가지 상호 보완적 LLM 기반 검증자를 사용합니다:[1]

$$\text{Verification Score} = \max(f_{\text{align}}, f_{\text{rubric}}, f_{\text{multimodal}})$$

1. **정렬 검증자 (Alignment Verifier)**: 수행된 조치와 최종 응답이 작업 의도와 일치하는지 평가합니다.

2. **루브릭 검증자 (Rubric Verifier)**: 작업의 하위 목표에 대한 점수 기반 루브릭을 생성합니다. 정상화된 점수가 0.8 이상인 경우 성공으로 표시합니다:

$$f_{\text{rubric}} = \frac{\text{earned points}}{\text{total points}} \geq 0.8$$

3. **다중모달 검증자 (Multimodal Verifier)**: 스크린샷과 최종 응답을 검사하여 환각(hallucination)을 감지합니다.

이들 검증자는 인간 판단과 **83.3% 일치율**을 보이며, 위양성률 16.7%, 위음성률 18.4%입니다.[1]

### 2.3 모델 구조

#### 2.3.1 기본 공식화

Fara-7B는 다음과 같이 수식화됩니다:[1]

$$T = (q_0, \{o_0, r_0, a_0\}, \ldots, \{o_T, r_T, a_T\})$$

여기서:
- $$q_0$$: 초기 사용자 쿼리
- $$o_t$$: 스크린샷 관찰
- $$r_t$$: 사고/추론 텍스트
- $$a_t$$: 원자적 조치

모델은 다음을 학습합니다:

$$P(r_t, a_t | q_0, \{o_0, r_0, a_0\}, \ldots, \{o_{t-1}, r_{t-1}, a_{t-1}\})$$

다중 턴 상호작용을 지원하기 위해:

$$P(r_{t+k}, a_{t+k} | q_0, \{o_0, r_0, a_0\}, \ldots, q_1, \{o_{t+1}, r_{t+1}, a_{t+1}\}, \ldots, \{o_{t+k-1}, r_{t+k-1}, a_{t+k-1}\})$$

#### 2.3.2 관찰 공간

**픽셀 입력 중심 접근**: Fara-7B는 DOM 파싱이나 접근성 트리에 의존하지 않고 순수 스크린샷만 사용합니다. 이는 다음과 같은 이점을 제공합니다:[1]

- 동적 웹사이트에 대한 더 강력한 일반화
- 불완전한 또는 유지되지 않는 접근성 트리 우회
- 실제 인간 컴퓨터 사용과의 일관성

#### 2.3.3 행동 공간

표 7에 제시된 11개의 원자적 행동:

| 행동 | 설명 |
|------|------|
| 키 입력 | 지정된 순서대로 키 누르기 (예: CTRL+C) |
| 타입 | 좌표 (x, y)에 입력 문자열 입력 |
| 마우스 이동 | 커서를 좌표로 이동 |
| 좌클릭 | 좌표에서 좌측 마우스 버튼 클릭 |
| 스크롤 | 마우스 휠 스크롤 |
| URL 방문 | 지정된 URL 방문 |
| 웹 검색 | 지정된 쿼리로 웹 검색 |
| 뒤로 가기 | 이전 페이지로 이동 |
| 기억 | 나중 참조용 정보 저장 |
| 대기 | 지정된 시간만큼 대기 |
| 종료 | 현재 작업 종료 |

#### 2.3.4 컨텍스트 관리

토큰 효율성을 위해 **최근 N=3개의 관찰만 유지**하면서, 모든 이전의 생각과 행동은 보존합니다. 이는 다음과 같은 트레이드오프를 구현합니다:[1]

$$\text{Context Efficiency} = \frac{\text{Accuracy}}{\text{Token Usage}} \propto \frac{1}{N}$$

### 2.4 모델 훈련

**기본 모델**: Qwen2.5-VL-7B를 기초로 사용합니다.[1]

**데이터 혼합** (약 1.8백만 샘플):

1. **궤적 데이터**: FaraGen으로부터 생성된 다중에이전트 궤적을 관찰-생각-행동 단계로 분해
2. **접지(Grounding) 데이터**: 스크린샷에서 UI 요소 식별
3. **거절 데이터**: 유해한 작업 거부 학습
4. **UI 스크린샷 QA 및 캡셔닝**: 웹페이지 정보 추출 능력 향상

**손실 함수**: 표준 교차 엔트로피 손실

$$\mathcal{L} = -\sum_{t=1}^{T} \log P(a_t | \text{history}_t)$$

### 2.5 성능 향상

#### 2.5.1 벤치마크 성과 (표 9, 10)[1]

**WebVoyager** (가장 성숙한 벤치마크):
- Fara-7B: 73.5% ± 1.0
- GPT-4o SoM 에이전트: 65.1% ± 0.6
- UI-TARS-1.5-7B: 66.4% ± 0.8
- OpenAI computer-use-preview: 70.9% ± 1.9

**Online-Mind2Web**:
- Fara-7B: 34.1% ± 3.7
- GPT-4o SoM 에이전트: 34.6% ± 1.5

**DeepShop** (쇼핑 작업):
- Fara-7B: 26.2% ± 2.0
- GPT-4o SoM 에이전트: 16.0% ± 2.3
- UI-TARS-1.5-7B: 11.6% ± 1.4

**WebTailBench** (새로운 벤치마크):
- Fara-7B: 38.4% ± 0.7
- GPT-4o SoM 에이전트: 30.8% ± 3.0
- OpenAI computer-use-preview: 25.7% ± 1.7

#### 2.5.2 비용 효율성 분석 (표 10)[1]

$$\text{Cost Efficiency Ratio} = \frac{\text{Task Accuracy}}{\text{Cost per Task}}$$

**Fara-7B의 비용 우위**:

$$\text{평균 작업당 비용} = \$0.025$$

이는 다음과 같이 비교됩니다:

- GPT-5 SoM 에이전트: $0.316 (약 12.6배 비쌈)
- o3 SoM 에이전트: $0.514 (약 20.5배 비쌈)
- OpenAI computer-use-preview: $0.913 (약 36.5배 비쌈)

출력 토큰 측면에서 Fara-7B는 약 1.1k 토큰을 사용하는 반면, o3은 20.9k 토큰을 사용합니다.

#### 2.5.3 추론 단계 효율성

작업당 평균 단계: 16.5 ± 21.1 (GPT-5와 비슷)

이는 다음을 의미합니다:

$$\text{단계 효율성} = \frac{\text{입력 토큰 수 / 단계}}{1.2 \times 10^5 / 16.5} \approx \text{매우 높음}$$

#### 2.5.4 데이터 및 추론 스케일링 (그림 7)[1]

**데이터 스케일링**:
- 20K 단계: ~40% 정확도
- 200K 단계: ~60% 정확도
- 2M 단계 (전체): ~73.5% 정확도

이는 강한 양의 스케일링 트렌드를 시사하며, Fara-7B는 추가 데이터로부터 계속 이득을 얻을 수 있습니다:

$$\text{Accuracy} \propto \log(\text{Training Data Size})$$

**추론 단계 스케일링**:
- 15 단계 제한: ~40% 정확도
- 50 단계 제한: ~60% 정확도
- 100 단계 제한: ~73.5% 정확도

#### 2.5.5 접지 성능 (표 13)[1]

ScreenSpot-V2 벤치마크에서 Fara-7B는 기본 모델 Qwen2.5-VL을 능가합니다:

- Fara-7B: 89.3%
- Qwen2.5-VL: 86.6%

텍스트 요소(Tx) 접지에서 특히 우수:
- 모바일: 97.5%
- 데스크톱: 95.3%
- 웹: 92.7%

### 2.6 한계

1. **드래그 앤 드롭 불가능**: 네이티브 지원 부재[1]

2. **미디어 처리 제한**: 비디오, 오디오 콘텐츠 감시 또는 청취 불가[1]

3. **저지연 작업 부적절**: 게임 플레이 같은 초저지연 상황에 대응 불가[1]

4. **복잡한 작업에서 감소된 정확도**: 더 복잡한 작업에서 성능 저하[1]

5. **임계점 이후의 데이터 부족**: 임계점 이후의 행동에 대한 학습 데이터 없음으로, 사용자 확인 후 모델 행동이 예측 불가능할 수 있음[1]

6. **환각 취약성**: 지원되지 않는 소스를 잘못 귀인하거나 오도하는 콘텐츠에 의해 오도될 수 있음[1]

7. **환경 변화에 대한 제한된 강건성**: 웹사이트 레이아웃의 변경에 대한 적응력 제한[1]

***

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재 일반화 성능

#### 3.1.1 교차 도메인 일반화

**도메인 다양성 지표**:
- 학습 데이터: 70,117개의 고유 도메인 방문[1]
- 평균 궤적당 고유 도메인: 약 0.5개

이는 **궤적의 약 50%가 데이터셋의 다른 곳에 나타나지 않는 웹사이트를 방문**함을 의미하며, 내재적인 도메인 일반화 능력을 시사합니다.

#### 3.1.2 작업 복잡도 일반화

표 11에 따르면 WebTailBench의 단일 기술 작업 vs. 다중 단계 작업 성능:

**단일 기술 작업** (7개 카테고리):
- 평균 성능: 약 42.2%

**다중 단계/교차 사이트 작업** (3개 카테고리):
- 쇼핑 목록 (2개 항목): 49.0%
- 가격 비교: 32.7%
- 구성 작업: 23.0%

이는 **더 복잡한 작업에서 성능이 감소**함을 보여주지만, 여전히 SoM GPT-4o (각각 17.0%, 27.5%, 26.7%)를 능가합니다.[1]

#### 3.1.3 시각 일반화

표 13의 ScreenSpot 성능은 **접지 능력의 강한 일반화**를 나타냅니다:

$$\text{Visual Generalization Score} = \frac{\text{Tx Accuracy} + \text{Icon Accuracy}}{2}$$

- 모바일 + 데스크톱 + 웹 평균: 89.3%

이는 다양한 화면 형식과 레이아웃에 걸친 일반화를 시사합니다.

### 3.2 향상 가능성

#### 3.2.1 데이터 스케일링의 한계

Figure 7 (좌측)의 데이터 스케일링 곡선은 다음과 같이 근사될 수 있습니다:

$$\text{Accuracy}(D) = A_{\max} \left(1 - e^{-\beta D}\right)$$

여기서:
- $$D$$: 훈련 데이터 크기
- $$\beta$$: 수렴 속도 상수
- $$A_{\max}$$: 점근적 최대 정확도

**추정**: 현재 2M 단계로 ~73.5%, 추가 데이터로 잠재적으로 75-80%에 도달 가능[1]

$$\text{Headroom} = A_{\max} - \text{Current Accuracy} \approx 6-7\%$$

#### 3.2.2 강화 학습(RL)의 미활용 잠재력

UI-TARS-1.5-7B는 광범위한 RL을 사용하지만, **Fara-7B는 순수 감독 미세조정(SFT)만 사용**합니다. 이는 다음을 의미합니다:[1]

$$\text{SFT vs RL Performance Gap} \propto \text{Potential RL Gains}$$

비교 데이터:
- Fara-7B (SFT만): 73.5% 정확도
- UI-TARS-1.5-7B (SFT + RL): 66.4% 정확도

따라서 Fara-7B가 RL을 채택하면 잠재적으로 **75-80% 범위에 도달** 가능합니다.

#### 3.2.3 더 큰 모델 크기로의 스케일링

**현재**: Qwen2.5-VL-7B (70억 파라미터)

Fara-7B를 더 큰 모델로 스케일링하면 (13B, 32B):

$$\text{Performance Improvement} \propto \log(\text{Model Size})$$

일반적으로 모델 크기 2배는 약 **5-10% 정확도 향상**을 야기합니다.

#### 3.2.4 멀티모달 입력 강화

**현재**: 스크린샷 + 브라우저 메타데이터만 사용

잠재적 개선:
- **구조적 정보**: 접근성 트리의 선택적 활용
- **OCR 개선**: 고급 텍스트 인식
- **의미론적 임베딩**: 페이지 요소의 의미론적 관계 명시

이러한 개선은 **5-15% 추가 성능 향상**을 가져올 수 있습니다.

#### 3.2.5 도메인 적응 기법

**제로샷 학습**: Fara-7B는 이미 제로샷 도메인 일반화에서 우수하지만, **도메인별 프롬프팅** 기법은 다음을 가능하게 합니다:

$$P(\text{action} | \text{domain-specific context}) > P(\text{action} | \text{generic context})$$

결과적으로 **3-8% 향상** 예상됩니다.

### 3.3 일반화의 구조적 요인

#### 3.3.1 FaraGen 데이터 세트의 역할

**일반화 성능 = 데이터 다양성 함수**

$$\text{Gen}_{\text{performance}} = f(\text{Domain Diversity}, \text{Task Complexity}, \text{Data Quality})$$

FaraGen의 강점:
1. **도메인 다양성**: 70,117개 고유 도메인 (직전 SOTA 벤치마크 대비 10배 이상)
2. **작업 복잡도**: 3-84 단계 범위, 평균 6.9 단계
3. **검증 품질**: 83.3% 인간 판단 일치율

#### 3.3.2 "픽셀 입력" 접근법의 일반화 우위

접근성 트리 기반 SoM 에이전트와 비교:

**SoM 에이전트의 일반화 문제**:
- 불완전한 또는 동적으로 생성되는 접근성 트리
- 웹사이트별 DOM 구조의 변동성
- 마크업 불일치로 인한 행동 선택 오류

**Fara-7B의 우위**:
- 인간이 인식하는 것과 동일한 입력 사용
- 시각적 패턴에 대한 더 강건한 학습
- 호스팅 또는 마크업 변경에 대한 낮은 민감도

실증적 증거: Table 9에서 Fara-7B가 모든 벤치마크에서 동일 크기의 SoM 에이전트(UI-TARS-1.5-7B)를 능가합니다.

#### 3.3.3 다단계 추론의 일반화

표 7 (중간)에서:

$$\text{Performance}(\text{step budget}) = a + b \cdot \log(\text{step budget})$$

- 15 단계: ~40%
- 100 단계: ~73.5%

이는 추론 예산을 증가시키면 **추가 5-10% 향상** 가능함을 시사합니다.

#### 3.3.4 메모리 메커니즘의 역할

Fara-7B의 "Memorize" 행동은 **교차 사이트 비교 작업**의 일반화를 향상시킵니다:

$$\text{Accuracy}_{\text{comparison}} = P(\text{retrieve remembered info}) \times P(\text{correct comparison})$$

WebTailBench 가격 비교 작업에서 이미 **32.7% 달성** (UI-TARS: 8.8%)

### 3.4 미래 일반화 개선 방향

#### 3.4.1 하이브리드 입력 접근법

$$P(\text{action} | \text{screenshot}, \text{selectively-used Ax-Tree}, \text{OCR})$$

- 정상 상황: 스크린샷만 사용
- 모호한 상황: 접근성 트리 선택적 활용
- 텍스트 집약적 페이지: OCR 강화

**예상 성능 향상**: 7-12%

#### 3.4.2 작업별 특화 모듈

$$f_{\text{specialized}}(x) = \alpha \cdot f_{\text{general}}(x) + (1-\alpha) \cdot f_{\text{specialized domain}}(x)$$

작업 카테고리별 경량 어댑터:
- 쇼핑 작업용
- 여행 예약용
- 정보 검색용

**예상 성능 향상**: 5-10%

#### 3.4.3 지속적 학습 프레임워크

온라인으로 새로운 도메인과 작업 패턴으로부터 학습:

$$\text{Accuracy}_{\text{time}}(t) = \text{Accuracy}_{\text{base}} + \int_0^t \Delta_{\text{online}}(\tau) d\tau$$

**예상 성능 향상**: 시간 경과에 따라 누적적 5-15%

***

## 4. 논문이 앞으로의 연구에 미치는 영향과 고려사항

### 4.1 학문적 영향

#### 4.1.1 컴퓨터 사용 에이전트 연구의 패러다임 전환

**기존 패러다임 (이전)**:
- 대형 모델 중심 (GPT-4, Claude 등)
- 외부 구조화된 정보 의존 (HTML, 접근성 트리)
- 소수의 대규모 벤치마크 사용

**새로운 패러다임 (Fara-7B 이후)**:
- 소형 효율적 모델의 가능성 증명
- 순수 시각 기반 접근법의 우월성
- 고품질 합성 데이터의 가치 입증

이는 **"작지만 강력한" 에이전트**라는 새로운 연구 방향을 개방했습니다.[1]

#### 4.1.2 합성 데이터 생성의 새로운 표준

**이전**: 제한된 규모, 높은 오류율
**현재**: 145K 궤적, 83.3% 검증 정확도, $1 당 생성 비용

이는 합성 데이터 생성 연구에서 **"FaraGen 스타일" 멀티에이전트 파이프라인**을 새로운 표준으로 설정했습니다.[1]

#### 4.1.3 평가 벤치마크 표준의 진화

**WebTailBench의 기여**:
- 11개 카테고리, 609개 작업
- 현실 세계 작업의 더 나은 표현
- 검증 시스템의 공개 릴리스

이는 **더 현실적이고 공정한 벤치마킹**의 새로운 기준을 제시합니다.[1]

### 4.2 산업 적용에 미치는 영향

#### 4.2.1 온디바이스 배포 가능성

**비용-효율 비교**:

| 모델 | 파라미터 | 비용/작업 | 배포 가능성 |
|------|---------|----------|-----------|
| Fara-7B | 7B | $0.025 | ✅ 높음 |
| GPT-4o SoM | N/A | $0.302 | ❌ 낮음 |
| GPT-5 SoM | N/A | $0.316 | ❌ 낮음 |

Fara-7B는 모바일, IoT, 엣지 디바이스에서의 실시간 배포를 가능하게 합니다.[1]

#### 4.2.2 기업 자동화의 개인화

작은 모델이 실행 가능하므로:

$$\text{Enterprise Agents} = \sum_{i=1}^{n} \text{Specialized Fara-7B}_i$$

각 부서별, 프로세스별 맞춤형 에이전트 배포:
- 영업 자동화
- HR 프로세싱
- 재무 보고서 생성

#### 4.2.3 프라이버시 강화 시스템

온디바이스 실행으로:
- 사용자 데이터가 외부 서버로 전송 안 됨
- 민감한 비즈니스 프로세스 보호
- GDPR, HIPAA 등 규정 준수 용이

### 4.3 앞으로의 연구 시 고려할 점

#### 4.3.1 안전성 및 거버넌스

**현재 상황**:
- AgentHarm-Chat에서 94.2% 거절율[1]
- WebTailBench-Refusals에서 81.9% 거절율[1]
- 임계점 감지에서 82.6% (19/23 작업) 성공[1]

**개선 필요 영역**:
1. **더 견고한 프롬프트 주입 방어**: 현재 13개 대적 테스트 중 9개만 통과[1]
2. **사기성 웹사이트 감지**: 온라인 사기 패턴 학습
3. **다중 레이어 검증**: 단일 모델 신뢰도 감소

#### 4.3.2 인간-에이전트 상호작용 설계

**현재 한계**: 임계점에서 에이전트가 완전히 정지

**향후 연구 방향**:

$$P(\text{user-approved action}) = P(\text{action success}) \times P(\text{user confirms})$$

- 인간이 이해 가능한 설명 생성
- 적응형 거절 수준 (위험도 기반)
- 실시간 모니터링 및 개입 인터페이스

#### 4.3.3 교차 도메인 일반화 강화

**현재**: 70K 도메인에서 학습한 일반화

**미래 연구**:
1. **도메인 시프트 감지**: 미지의 도메인 자동 인식
2. **적응형 프롬프팅**: 도메인 특성에 따른 프롬프트 조정
3. **전이 학습**: 새로운 도메인에 최소한의 데이터로 적응

#### 4.3.4 장기 작업 수행 능력

**현재**: 평균 6.9 단계, 최대 84 단계

**미래 요구사항**: 수백 단계의 복잡한 작업

**개선 방법**:

$$\text{Long-Horizon Success} = \sum_{t=1}^{T} P(\text{correct action}_t | \text{state}_t)$$

여기서 $$\sum P \leq P^T$$이므로 오류가 누적됩니다.

해결책:
1. **계층적 작업 분해**: 큰 작업을 하위 작업으로 분해
2. **체크포인트 메커니즘**: 중요 지점에서의 상태 저장 및 복구
3. **오류 복구 정책**: 실패로부터의 자동 복구

#### 4.3.5 다양한 행동 공간 확장

**현재**: 11개의 기본 행동

**미래 확장**:
- 파일 시스템 조작 (로컬 파일 관리)
- 애플리케이션 전환 (멀티 앱 워크플로우)
- 마우스 드래그 앤 드롭
- 음성/비디오 상호작용
- API 호출 (웹 스크래핑 대신)

#### 4.3.6 데이터 효율성의 한계 탐색

**현재**: 145K 궤적으로 SOTA 성능

**더 효율적인 학습**:

$$\text{Data Efficiency} = \frac{\text{Performance}^2}{\text{Data Volume}}$$

- **자기 개선**: 모델이 이전 실패로부터 학습
- **메타 학습**: 새로운 작업 유형에 빠르게 적응
- **지식 증류**: 소형 모델이 더 큰 모델의 추론 학습

#### 4.3.7 모델 편향과 공정성

**고려사항**:
- ClueWeb22/Tranco 코퍼스의 지리적 편향 (주로 영어, 서구 중심)
- 특정 웹사이트 유형 과대/과소 표현
- 특정 사용자 프로필에 대한 편향

**미래 연구**:

$$\text{Fairness Score} = \frac{1}{n} \sum_{i=1}^{n} P(\text{success}_i)$$

모든 인구 통계학적 그룹 $$i$$에서 균형잡힌 성공률 목표

### 4.4 2020년 이후 관련 최신 연구와의 연관성

#### 4.4.1 웹 에이전트 연구의 진화 경로

**2020-2021 초기 단계**:
- Mind2Web (2023): 웹 네비게이션 데이터셋 소개[2]
- WebShop, WebArena: 제한된 환경에서의 벤치마크

**2022-2023 중기 발전**:
- CogAgent (2024): 18B GUI 특화 모델[3]
- UI-TARS (2025): 7B 규모 컴퓨터 사용 모델[1]
- VisualWebArena: 현실적 웹 환경 벤치마크

**2024-2025 최근 동향**:
- **Fara-7B (2025)**: 합성 데이터 중심 소형 모델[1]
- PC Agent-E: 고품질 데이터로 효율적 훈련[4]
- OpenAI computer-use-preview, Claude 3.5 Sonnet: 프로토타입 배포[1]

**Fara-7B의 위치**: 합성 데이터 생성의 효율성과 소형 모델의 가능성을 연결하는 **중요한 전환점**

#### 4.4.2 합성 데이터 생성의 최신 동향과의 연계

**관련 연구**:
- MAG-V (2024): 멀티에이전트 검증 프레임워크[5]
- MetaSynth (2025): 메타 프롬팅 기반 다양한 데이터 생성[6]
- Matrix (2025): 탈중앙화 멀티에이전트 합성 데이터 생성[7]
- OptiTrust (2025): 검증 가능한 합성 데이터 파이프라인[8]

**Fara-7B의 기여**: FaraGen은 이러한 최신 기법들을 **웹 기반 작업에 처음 적용한 체계적 시스템**입니다.

#### 4.4.3 멀티모달 모델 일반화 연구와의 연계

**최신 조사 결과**:
- VLM 도메인 적응 서베이 (2025): 제로샷 일반화의 한계와 미세조정의 중요성[9]
- Enhancing Generalization in Vision-Language-Action Models (2025): 40% 일반화 개선 달성[10]

**Fara-7B의 시사점**: 
- 스크린샷만 사용하는 "픽셀 입력" 접근이 **실제로 더 나은 일반화**를 제공
- 이는 기존 VLM 일반화 연구와 **대조적인 발견**으로, 과도한 구조화가 오히려 일반화를 해친다는 증거

#### 4.4.4 안전 및 거버넌스 연구와의 연계

**관련 연구**:
- AdvCUA (2025): 컴퓨터 사용 에이전트의 보안 위협 벤치마크[11]
- OS-Harm (2025): 컴퓨터 사용 에이전트 안전 벤치마크[12]
- AgentHarm (2024): LLM 에이전트 해로움 측정

**Fara-7B의 위치**: 안전성을 **설계 초반부터 고려** (임계점, 다중 검증자)

***

## 5. 결론

### 5.1 핵심 성과

Fara-7B는 세 가지 핵심 성과를 달성했습니다:

1. **데이터 부족 문제의 체계적 해결**: FaraGen을 통해 145,603개의 고품질 궤적을 약 $1/작업의 비용으로 생성

2. **효율성과 성능의 새로운 균형**: 7B 파라미터 모델이 훨씬 더 큰 모델들과 경쟁하면서 비용은 12-36배 낮음

3. **시각 기반 접근의 우월성 입증**: 접근성 트리 의존 없이 순수 스크린샷으로 더 나은 일반화 달성

### 5.2 미래 전망

**단기 (1-2년)**:
- 엣지 디바이스 배포 및 온디바이스 에이전트 상용화
- 도메인별 특화 모델 개발
- 안전성 및 거버넌스 프레임워크 강화

**중기 (2-5년)**:
- 멀티 에이전트 시스템으로의 확장
- 장기 작업 수행 능력 향상
- 인간-에이전트 협업 모델 정착

**장기 (5년 이상)**:
- AGI 수준의 일반적 컴퓨터 사용 능력 달성
- 모든 소프트웨어 인터페이스에 대한 범용 에이전트
- 완전한 자율 업무 수행

### 5.3 학문과 산업에 미치는 영향

Fara-7B는 컴퓨터 사용 에이전트 연구에서 **진정한 전환점**을 표시합니다. 합성 데이터 생성과 소형 모델의 가능성을 동시에 입증함으로써, 향후 AI 에이전트 연구는 **"비용 효율적이고 배포 가능한" 솔루션 개발에 초점**을 맞출 것으로 예상됩니다.

***

## 참고 문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/158e21ec-c938-4473-8d85-ea04502f8717/Fara-7B-An-Efficient-Agentic-Model-for-Computer-Use.pdf)
[2](https://openreview.net/forum?id=efFmBWioSc)
[3](https://openaccess.thecvf.com/content/CVPR2024/papers/Hong_CogAgent_A_Visual_Language_Model_for_GUI_Agents_CVPR_2024_paper.pdf)
[4](https://arxiv.org/html/2505.13909v1)
[5](https://arxiv.org/abs/2412.04494)
[6](https://aclanthology.org/2025.findings-acl.962)
[7](https://www.semanticscholar.org/paper/ca277c154deaa3bca0f93cb2ab076df4743a0ffc)
[8](https://arxiv.org/abs/2508.03117)
[9](https://arxiv.org/abs/2506.18504)
[10](https://seohyun00.tistory.com/53)
[11](https://arxiv.org/abs/2510.06607)
[12](https://arxiv.org/abs/2506.14866)
[13](https://dl.acm.org/doi/10.1145/3613904.3642377)
[14](https://arxiv.org/abs/2510.04607)
[15](https://ieeexplore.ieee.org/document/10831260/)
[16](https://arxiv.org/abs/2506.23774)
[17](https://dl.acm.org/doi/10.1145/3686215.3688378)
[18](https://arxiv.org/abs/2510.19438)
[19](https://www.semanticscholar.org/paper/0a0e029696266e95462a9ad9710c5cb2ccd5cedd)
[20](https://arxiv.org/abs/2506.07672)
[21](http://arxiv.org/pdf/2504.04485.pdf)
[22](https://arxiv.org/pdf/2408.09955.pdf)
[23](https://arxiv.org/pdf/2409.03215.pdf)
[24](http://arxiv.org/pdf/2503.03459.pdf)
[25](https://arxiv.org/pdf/2411.07464.pdf)
[26](http://arxiv.org/pdf/2503.21460.pdf)
[27](https://arxiv.org/pdf/2312.15224.pdf)
[28](http://arxiv.org/pdf/2409.02977.pdf)
[29](https://www.emergentmind.com/topics/agentic-multimodal-models)
[30](https://cs231n.stanford.edu/papers/text_file_840592471-CS_231N_Final_Report.pdf)
[31](https://pushsecurity.com/blog/considering-the-impact-of-computer-using-agents/)
[32](https://www.sparkouttech.com/multi-model-ai-agent/)
[33](https://arxiv.org/abs/2506.10172)
[34](https://icml.cc/virtual/2025/workshop/39960)
[35](https://arxiv.org/abs/2510.10991)
[36](https://www.classicinformatics.com/blog/how-llms-and-multi-agent-systems-work-together-2025)
[37](https://www.lgresearch.ai/blog/view?seq=565)
[38](https://arxiv.org/abs/2410.14853)
[39](https://dl.acm.org/doi/10.1145/3721201.3725444)
[40](https://arxiv.org/abs/2406.14773)
[41](https://link.springer.com/10.1007/978-3-031-70415-4_7)
[42](https://arxiv.org/abs/2502.03078)
[43](https://ojs.aaai.org/index.php/AAAI/article/view/34645)
[44](https://arxiv.org/abs/2403.04190)
[45](http://arxiv.org/pdf/2411.03250.pdf)
[46](http://arxiv.org/pdf/2201.12677.pdf)
[47](https://arxiv.org/pdf/2301.07573.pdf)
[48](https://arxiv.org/pdf/2503.14023.pdf)
[49](https://arxiv.org/pdf/2304.03722.pdf)
[50](https://www.mdpi.com/1424-8220/24/1/266/pdf?version=1704185657)
[51](https://arxiv.org/html/2410.11963v1)
[52](https://insights.exsquared.com/why-synthetic-data-is-the-hottest-ai-trend-in-2025/)
[53](https://www.sciencedirect.com/science/article/abs/pii/S1566253525006955)
[54](https://dataconomy.com/2025/11/25/microsofts-fara-7b-new-agentic-llm-from-screenshots/)
[55](https://www.futuremarketinsights.com/reports/synthetic-data-generation-market)
[56](https://milvus.io/ai-quick-reference/can-visionlanguage-models-generalize-to-new-domains-without-retraining)
[57](https://www.marktechpost.com/2025/11/24/microsoft-ai-releases-fara-7b-an-efficient-agentic-model-for-computer-use/)
[58](https://www.iwco.co/synthetic-data-generation-a-comprehensive-overview/)
[59](https://www.microsoft.com/en-us/research/publication/fara-7b-an-efficient-agentic-model-for-computer-use/)
[60](https://aimatters.co.kr/news-report/english-news/24033/)
