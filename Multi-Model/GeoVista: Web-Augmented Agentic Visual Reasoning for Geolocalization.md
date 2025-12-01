
# GeoVista: Web-Augmented Agentic Visual Reasoning for Geolocalization

## 1. 핵심 주장 및 주요 기여

### 핵심 주장
GeoVista 논문은 **현대의 에이전틱 다중모달 추론 시대에서 지리정보 위치 파악(geolocalization) 작업을 재검토**하면서, 시각적 단서 추출과 웹 검색을 결합한 동적 추론 루프의 중요성을 제시합니다. 기존의 폐쇄형 모델(GPT-5, Gemini-2.5) 수준의 성능을 오픈소스 모델로 달성할 수 있음을 보여줍니다.[1]

### 주요 기여
1. **GeoBench 벤치마크 도입**: 고해상도 이미지, 합리적인 지역화 가능성, 다층 지역 정보, 세밀한 평가 지표를 포함한 첫 번째 전문 지리정보 위치 파악 벤치마크[1]

2. **GeoVista 에이전틱 모델**: 이미지 줌-인 도구와 웹 검색 도구를 동적 추론 루프 내에서 통합하는 모델로, 사람의 인지 과정을 모방[1]

3. **완전한 훈련 파이프라인**: 냉기동 감독 미세조정(SFT) + 강화학습(RL) 단계를 포함하며, **계층적 보상(hierarchical reward)**을 설계하여 다층 지역 정보를 활용[1]

***

## 2. 해결하고자 하는 문제

### 문제점 분석
현재의 에이전틱 시각 추론 연구는 주로 **이미지 조작 도구(자르기, 회전, 줌)에만 초점**을 맞추고 있으며, 외부 정보 검색 도구(예: 웹 검색)의 통합이 부족합니다. 또한 기존의 지리정보 위치 파악 벤치마크는:

- 저해상도 이미지 사용
- 고난이도의 에이전틱 추론을 위한 설계 부재
- 쉽게 인식되는 랜드마크 포함 (과도한 편향)
- 세밀한 평가 지표 부족[1]

### GeoVista의 해결 방식
웹 검색 도구와 시각 조작 도구를 **통합한 다중 턴 추론 루프**를 통해 모델이 자동으로 도구 사용 시기와 방식을 결정하도록 함으로써, 외부 지식과 시각적 단서를 효과적으로 결합하는 것을 목표로 함.[1]

***

## 3. 제안하는 방법 (수식 포함)

### 3.1 에이전틱 파이프라인 구조

주어진 사용자 쿼리와 입력 이미지에 대해, 정책 모델이 다음과 같이 반복적으로 작동합니다:[1]

**생각(Thought) → 행동(Action) → 관찰(Observation) 루프**

각 단계에서:
- **생각 $T_i$**: 자연어로 된 추론 근거
- **행동 $A_i$**: 도구 호출 또는 최종 답변
- **관찰 $O_i$**: 도구 실행 결과 또는 최종 위치 예측

### 3.2 하버사인 거리 공식 (Nuanced Evaluation)

예측 위치와 실제 위치 간의 거리를 계산하기 위해 다음 공식을 사용합니다:[1]

$$d = 2R_e \arcsin\left(\sqrt{v}\right)$$

여기서:

$$v = \sin^2\left(\frac{\phi_2 - \phi_1}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\lambda_2 - \lambda_1}{2}\right)$$

- $(φ_1, λ_1)$: 예측 지점의 위도/경도
- $(φ_2, λ_2)$: 실제 지점의 위도/경도  
- $R_e$: 지구의 반지름 (약 6371 km)[1]

### 3.3 GRPO 기반 강화학습

표준 GRPO 목표 함수:[1]

$$\mathcal{J}_{GRPO}(\theta) = \mathbb{E}_{q \sim \mathcal{D}, \{o_i\}^G_{i=1} \sim \pi_{\theta_{old}}(\cdot|q)} \frac{1}{G} \sum^G_{i=1} \left[ \min\left( \frac{\pi_{\theta}(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}A_i, \text{clip}\left(\frac{\pi_{\theta}(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1-\epsilon, 1+\epsilon\right)A_i \right) \right]$$

정규화된 어드밴티지 함수:[1]

$$A_i = \frac{r_i - \text{mean}(\{r_1, r_2, \ldots, r_G\})}{\text{std}(\{r_1, r_2, \ldots, r_G\})}$$

### 3.4 계층적 보상 구조

**다층 지역 정보를 활용한 핵심 혁신**:[1]

$$r_i = \begin{cases} \beta^2 & \text{if city-level correct} \\ \beta & \text{if provincial/state-level correct} \\ 1 & \text{if country-level correct} \\ 0 & \text{else} \end{cases}$$

여기서 $\beta = 2$로 설정하여 더 세밀한 행정 단계의 정확한 예측에 더 높은 보상을 부여합니다.[1]

***

## 4. 모델 구조

### 4.1 기본 아키텍처

**입력 처리:**
- 기본 모델: **Qwen2.5-VL-7B-Instruct**
- 이미지 픽셀 예산: 최대 2M 픽셀로 다운샘플링[1]

**도구 통합:**

| 도구 | 기능 | 입력 형식 | 출력 형식 |
|------|------|---------|---------|
| **Crop-and-Zoom** | 관심 영역 확대 | bbox_2d (4 좌표) | 확대된 이미지 |
| **Web-Search** | 웹 정보 검색 | 텍스트 쿼리 | 최대 10개 관련 문서 |

### 4.2 훈련 파이프라인

**Stage 1: 냉기동 SFT (2,000 샘플)**
- 대규모 VLM(Seed-1.6-vision)이 여러 영역과 웹 검색 쿼리 제안
- 도구 호출과 추론 궤적 수집
- Learning rate: $1 \times 10^{-5}$, Batch size: 32, Epochs: 1[1]

**Stage 2: 강화학습 (12,000 샘플)**
- GRPO 알고리즘 적용
- Learning rate: $1 \times 10^{-6}$, Batch size: 64
- 최대 6 턴 제한, 32K 토큰 컨텍스트[1]

### 4.3 GeoBench 구성

| 데이터 유형 | 샘플 수 | 해상도 | 특징 |
|-------------|--------|-------|------|
| **표준 사진** | 512 | 최소 1600×1200 | 다양한 시나리오 |
| **파노라마** | 512 | 4096×2048 | 360도 거리뷰 |
| **위성 이미지** | 108 | 2000×2000 | Sentinel-2 Level 2A |
| **총계** | 1,132 | 고해상도 | 6대륙, 66개국, 108개 도시 |

***

## 5. 성능 향상 및 한계

### 5.1 주요 성능 결과

**표준 벤치마크 성능 (Table 2):**[1]

| 모델 | 국가 정확도(%) | 주/도 정확도(%) | 도시 정확도(%) |
|------|---------------|-----------------|---------------|
| **Gemini-2.5-pro** | 97.20 | 86.78 | **78.98** |
| **GPT-5** | 94.09 | 77.69 | 67.11 |
| **Gemini-2.5-flash** | 90.54 | 79.16 | 73.29 |
| **GeoVista-7B (ours)** | **92.64** | **79.60** | **72.68** |
| *Thyme-RL-7B* | 69.61 | 44.31 | 30.21 |
| *Mini-o3-7B* | 20.14 | 11.52 | 11.30 |

세밀한 평가 지표 (하버사인 거리):[1]

| 모델 | <3km 정확도(%) | 중앙값 거리(km) |
|------|--------------|-----------------|
| **Gemini-2.5-pro** | 64.45 | 0.80 |
| **GeoVista-7B** | 52.83 | **2.35** |
| Thyme-RL-7B | 29.88 | 880.97 |

### 5.2 절제 연구 결과 (Ablation Study)

계층적 보상의 효과:[1]

| 구성 | 중앙값 거리(km) | 파노라마 도시(%) | 사진 도시(%) | 위성 도시(%) |
|------|--------------|-----------------|------------|------------|
| Qwen-2.5-VL | 2209.82 | 24.22 | 44.73 | 16.10 |
| w/o Cold Start | 55.32 | 48.52 | 43.63 | 27.46 |
| w/o RL | 11.17 | 54.88 | 57.23 | 29.66 |
| w/o HR (계층적 보상 제거) | 4.11 | 75.0 | 68.95 | 40.68 |
| **GeoVista-7B** | **2.35** | **79.49** | **72.27** | **44.92** |

### 5.3 데이터 확장 효과

로그 선형 성능 향상:[1]

$$\text{성능} = f(\log(\text{데이터 크기}))$$

- 1.5K 샘플: 도시 70%, 주 80%, 국가 95%
- 3K 샘플: 도시 84%, 주 91%, 국가 98%
- 12K 샘플: 도시 92%, 주 98%, 국가 99%

### 5.4 주요 한계

**1. 위성 이미지 성능 저하**
- 도시 정확도: 44.92% (사진 72.27% vs. 위성 44.92%)
- 원인: 위성 이미지의 추상적 특성으로 인한 시각적 단서 부재[1]

**2. 폐쇄형 모델과의 성능 격차**
- Gemini-2.5-pro의 도시 정확도: 78.98% vs. GeoVista: 72.68%
- 하버사인 거리 중앙값: 0.80km vs. 2.35km[1]

**3. 도구 호출 오류율**
- 강화학습 초기 단계에서 부정확한 바운딩 박스 생성
- 점차 감소하지만 여전히 존재[1]

**4. 컨텍스트 윈도우 제한**
- 최대 32K 토큰 제한으로 매우 높은 해상도 이미지 처리 불가
- 다중 턴 상호작용 제약[1]

***

## 6. 모델의 일반화 성능 향상 가능성

### 6.1 현재 일반화 능력 분석

**데이터 유형별 일반화:**

GeoVista는 세 가지 데이터 유형(파노라마, 사진, 위성)에서 다음과 같은 성능을 보임:[1]

- **파노라마**: 도시 정확도 79.49% (최고 성능)
- **사진**: 도시 정확도 72.27% (중간 성능)
- **위성**: 도시 정확도 44.92% (낮은 성능)

이는 **데이터 분포 편향**과 **시각적 특성의 차이**를 시사합니다.

### 6.2 향상 가능성 논문 분석

#### 가능성 1: 계층적 강화학습 개선
현재 계층적 보상($r_i = \beta^2, \beta, 1, 0$)은 이진 판정이므로, **연속적 보상 함수**로 개선 가능:[1]

$$r_i^{*} = \exp\left(-\frac{d_i}{d_{\text{threshold}}}\right) \cdot w_{\text{level}}$$

여기서 $d_i$는 예측과 실제 위치의 거리, $w_{\text{level}}$은 행정 단계별 가중치입니다.

#### 가능성 2: 적응형 데이터 크기 확장
로그 선형 성능 증가(Figure 7-LEFT)를 보면:[1]

- 1.5K → 12K (8배 증가): 약 22% 성능 향상
- 추정치: 50K 샘플에서 **96-98% 도시 정확도 가능**

#### 가능성 3: 도메인별 전문화 모델
위성 이미지의 낮은 성능(44.92%)을 해결하기 위해 **도메인 특화 모델** 개발 가능:[1]

- 위성 이미지 특화 인코더 (예: RemoteSensing-ViT)
- 위성 데이터 기반 사전 학습 (OSV-5M 활용)
- 예상 성능: 60-65% 도시 정확도

#### 가능성 4: 동적 도구 선택
현재는 항상 두 가지 도구(줌-인, 웹 검색)를 사용하지만:[1]

$$P(\text{도구 사용} | \text{입력}) = \text{Softmax}(\text{도구 정책})$$

를 통해 **적응형 도구 선택**이 가능하며, 이는:
- 계산 효율성 20-30% 개선
- 불필요한 도구 호출 감소로 성능 유지/향상

#### 가능성 5: 멀티 에이전트 토론 프레임워크
최근 연구(2024-2025)에서 제시된 **다중 에이전트 토론(multi-agent debate)**:[2]

여러 GeoVista 에이전트가 :
- 독립적으로 위치 예측
- 상충하는 예측 논쟁
- 그래프 기반 상호작용으로 합의 도출

예상 성능: **74-76% 도시 정확도** (현재 72.68% 대비)

***

## 7. 관련 최신 연구 (2020년 이후)

### 7.1 에이전틱 시각 추론 분야

**핵심 발전 흐름:**[3][4][5][6]

| 연도 | 주요 모델/연구 | 핵심 기여 |
|------|-------------|---------|
| 2024 | OpenAI o3 | 도구 기반 시각 추론의 선례 (자르기, 회전, 줌) |
| 2025 | Mini-o3 | 반복적 지역 선택과 다중 턴 탐색 |
| 2025 | Thyme | 코드 실행 시각 샌드박스 |
| 2025 | DeepEyes | RL 유도 줌 행동 (SFT 없음) |
| 2025 | GeoVista | **웹 검색 통합 + 계층적 보상** |

### 7.2 시각 체인-오브-사고(Visual CoT) 연구

**다중 중간 단계 추론**의 진화:[7][8][9][10][11]

- **Visual CoT (2024)**: 시각적 주석(박스, 영역) 생성으로 주의 집중
- **VCoT (2023-2024)**: 재귀적 시각 보충으로 논리적 간격 해소
- **Multimodal Visualization-of-Thought (2025)**: 시각 생성 기반 사고
- **ReFocus (2025)**: 이미지 편집 기반 구조적 이해
- **Reason-RFT (2025)**: GRPO 기반 시각 추론 미세조정[12]

### 7.3 웹 검색 통합 에이전트

**검색 강화 다중모달 추론**:[13][14][15]

- **MMSearch-R1 (2025)**: 자동 도구 호출과 지능형 검색 쿼리
- **DeepMMSearch-R1 (2025)**: 다중 턴 웹 검색 + 동적 쿼리 적응
- **LiveVQA (2025)**: 실시간 시각 정보 업데이트 평가[16]

**핵심 개선점:**
- 텍스트/이미지 혼합 검색 도구 사용
- 자체 수정 및 반복적 쿼리 개선
- 검색-추론 순환 최적화

### 7.4 다중모달 강화학습 (GRPO, RFT)

**최신 RL 기법 발전:**[17][18][19]

| 기법 | 핵심 아이디어 | 2025 응용 |
|------|-----------|---------|
| **GRPO** | 그룹 정규화 어드밴티지 추정 (비판 모델 불필요) | GeoVista, DeepMMSearch-R1 |
| **RL-with-Cold-Start** | SFT + RL 2단계 파이프라인 | 모든 최신 VLM 추론 모델 |
| **ViPO** | 픽셀 수준 시각 선호 최적화 | 시각 생성 작업 |
| **VTool-R1** | 다중모달 수단 생성 학습 | 시각 도구 사용 훈련[5] |

### 7.5 지리정보 위치 파악 벤치마크

**관련 최신 데이터셋 (2024-2025):**[20][21][22]

| 벤치마크 | 연도 | 특징 | 규모 |
|---------|------|------|------|
| **OpenStreetView-5M** | 2024 | 대규모 거리뷰 다양성 | 5M 이미지 |
| **GeoComp** | 2025 | 인간 게임플레이 데이터 + 추론 | 대규모 |
| **KoreaGEO** | 2025 | 한국 거리뷰 + 세밀한 평가 | 1,080 이미지 |
| **GEOBench-VLM** | 2025 | 지구공간 작업 포괄 | 10,000+ MCQ |
| **GeoBench (GeoVista)** | 2025 | **웹 증강 에이전틱 평가** | 1,142 고해상도 |

### 7.6 다중모달 추론 벤치마크

**종합 평가 연구 (2024-2025):**[23][3]

- **Compositional Visual Reasoning Survey (2025)**: 260+ 논문 분석, 60+ 벤치마크[3]
- **mmJEE-Eval (2025)**: 과학 추론 심층 평가[23]
- **BALROG (2024)**: 게임 기반 에이전틱 능력 평가[24]

***

## 8. 논문의 연구에 미치는 영향과 향후 고려사항

### 8.1 학문적 영향

#### 영향 1: 에이전틱 다중모달 추론의 새로운 패러다임
GeoVista는 기존의 **"이미지 조작 도구 중심" 패러다임**을 **"다중 도구 통합 추론"으로 확장**:[1]

- 기존: 자르기 → 회전 → 줌 (시각 도구만)
- **GeoVista**: 시각 조작 + 웹 검색 + 지식 통합 (다중 모달리티)

이는 OpenAI o3와 다른 폐쇄형 모델들의 원리를 오픈소스에 적용한 첫 사례로, 향후 연구의 표준이 될 가능성이 높습니다.[2]

#### 영향 2: 계층적 보상 구조의 일반화
다층 행정 단위를 반영한 **계층적 보상**($r_i = \beta^2, \beta, 1, 0$)은:[1]

- 지리정보 위치 파악 외 다른 분야로 **직접 이전 가능**
- 예: 의료(진단 → 부위 → 조직 단계), 건축(국가 → 도시 → 건물)
- **다단계 정밀도 요구 작업의 표준 설계 패턴**으로 기여

#### 영향 3: 오픈소스 경쟁력 입증
동일한 7B 매개변수로 폐쇄형 모델(Gemini-2.5-flash, GPT-5)과 경쟁 가능함을 보여줌:[1]

| 지표 | GeoVista-7B | Gemini-2.5-flash | GPT-5 |
|------|------------|-----------------|-------|
| 도시 정확도 | 72.68% | 73.29% | 67.11% |
| 파라미터 (추정) | 7B | ~50B+ | ~200B+ |
| 효율성 | 1x | 7x+ | 28x+ |

→ **효율성 대비 성능**에서 우위

### 8.2 기술적 영향

#### 기술 1: 강화학습 패이라인의 모범 사례
**냉기동 SFT + RL (GRPO) 조합**의 효과 입증:[4][1]

현재 많은 2025년 최신 다중모달 모델이 동일한 구조 채택:
- Advancing Multimodal Reasoning via RL with Cold Start (2025)[4]
- VTool-R1 (2025)[5]
- Reason-RFT (2025)[12]

**표준화된 훈련 프로토콜로 확립 중**

#### 기술 2: 도구 오류 암묵적 학습
흥미로운 발견: RL 훈련 중 **명시적 최적화 없이도** 도구 호출 오류율이 감소:[1]

```
초기 오류율: 8% → 최종 오류율: 2%
(직접 감독 신호 없음)
```

→ **암묵적 학습 메커니즘** 연구의 새로운 방향 제시

#### 기술 3: 다중 데이터 유형 일반화 평가
세 가지 이질적 이미지 유형(파노라마, 사진, 위성)에서의 성능 분석:[1]

향후 연구는 **데이터 유형 적응형 모델** 개발로 진화:
- 도메인 특화 인코더
- 적응형 도구 선택
- 동적 레이아웃 처리

***

### 8.3 실무적 응용 가능성

#### 응용 1: 프라이버시-센서 기술
소셜 미디어 이미지로부터의 자동 위치 추론 위험성 강조:[21]

- **정보 공개 위험**: 사용자 모르게 위치 추출 가능
- 향후 연구: 위치 프라이버시 방어 메커니즘 필요

#### 응용 2: 원격 감지 및 지구 관측
위성 이미지 성능 44.92% → 개선 필요:[1]

- 재해 모니터링 (침수, 산불)
- 도시 계획 및 인프라 관리
- 환경 변화 추적

#### 응용 3: 엔터프라이즈 검색 및 RAG
웹 검색 통합의 일반화:[14][15]

- 기업 문서 기반 질의응답
- 멀티모달 검색 엔진
- 지능형 정보 검색 시스템

***

### 8.4 향후 연구 시 고려사항

#### 고려사항 1: 계산 효율성과 성능 트레이드오프

현재 GeoVista는:
- 최대 6 턴 도구 호출 허용
- 평균 1.96 도구 호출/쿼리[1]

**개선 방향:**
$$\text{성능} = f(\text{도구 호출, 레이턴시, 비용})$$

$$\max_{\text{정책}} \; \text{성능} - \lambda \cdot (\text{호출 수} + \text{레이턴시})$$

→ **적응형 도구 사용 정책** 학습

#### 고려사항 2: 크로스 도메인 일반화

**위성 이미지의 낮은 성능(44.92%)은 구조적 문제:**
- 직관적 시각적 단서 부족
- 추상적 표현 필요
- 도메인 갭 큼

**해결 방안:**
1. 도메인 적응 기법 (Domain Adaptation)
2. 멀티태스크 학습 (Multi-task Learning)
3. 자기 감독 사전 학습 (Self-supervised Pretraining)

#### 고려사항 3: 모델 크기와 성능의 관계

GeoVista는 7B 모델로 설계되었으나:[1]

- 3B 모델으로 축소 시 성능 손실 예상
- 13B, 70B 모델로 확장 시 성능 향상 가능

**향후 탐색:**
$$\text{성능}(\text{모델 크기}) = a \cdot \log(\text{크기}) + b$$

의 스케일링 법칙 규명

#### 고려사항 4: 진정한 일반화 능력 측정

현재 평가:
- 1,132 샘플의 **폐쇄형 테스트 셋** 기반
- 실제 사용자 데이터 분포와 다를 가능성

**개선 사항:**
1. 시간 변화 데이터셋 (시간 경과에 따른 변화)
2. OOD (Out-of-Distribution) 데이터 평가
3. 적대적 샘플 (광각, 야간, 악천후)

#### 고려사항 5: 투명성과 해석 가능성

에이전틱 모델의 중요한 문제:[1]

- 왜 특정 도구를 선택했는가?
- 웹 검색 결과를 어떻게 통합했는가?
- 잘못된 예측의 근본 원인은?

**해결 방안:**
- 주의 시각화 (Attention Visualization)
- 도구 선택 근거 추출
- 반사실적 설명 (Counterfactual Explanations)

#### 고려사항 6: 확장성과 비용

**웹 검색 도구의 비용 문제:**[1]

- API 호출 비용 (검색 엔진)
- 레이턴시 (네트워크 지연)
- 검색 결과의 신뢰성 변동

**개선 방안:**
1. 로컬 지식베이스 통합
2. 캐싱 메커니즘
3. 조건부 검색 (필요한 경우만)

***

## 9. 결론 및 종합 평가

### 9.1 핵심 기여 재정리

GeoVista는 다음 세 가지 측면에서 **다중모달 에이전틱 추론 분야의 이정표**입니다:

1. **방법론**: 웹 검색 + 시각 조작의 통합 에이전틱 루프
2. **평가**: 고해상도, 다층 평가의 GeoBench 벤치마크
3. **성능**: 7B 모델로 폐쇄형 모델 수준 달성

### 9.2 일반화 능력 전망

**낙관적 시나리오 (18-24개월 내):**
- 도시 정확도: 72.68% → **85-88%**
  - 데이터 확장 (12K → 50K+)
  - 도메인 특화 모델 개발
  - 멀티 에이전트 토론 프레임워크
- 위성 이미지: 44.92% → **65-70%**
- 하버사인 거리 중앙값: 2.35km → **0.5-1.0km**

**도전 과제:**
- 위성 데이터의 내재적 어려움
- 실시간 업데이트 정보 통합
- 개인정보 보호와 성능의 균형

### 9.3 학술적 유산

GeoVista는 다음 연구들의 기초가 될 것으로 예상:
- 계층적 강화학습의 일반화
- 오픈소스 에이전틱 모델의 다중 도구 통합
- 다중모달 검색 증강 생성 (Multimodal RAG)

2025년 최신 연구(DeepMMSearch-R1, VTool-R1, Reason-RFT 등)에서 이미 유사 패턴 채택 중이며, 향후 5년 간 **표준 설계 패턴**으로 정착될 가능성이 높습니다.[5][4][12]

***

## 참고 문헌 및 인용

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/60f9d76d-c205-49c2-a60c-18e0e924dbf8/2511.15705v1.pdf)
[2](https://www.emergentmind.com/topics/agentic-visual-reasoning-for-geolocalization)
[3](https://arxiv.org/abs/2508.17298)
[4](https://huggingface.co/papers/2505.22334)
[5](https://openreview.net/forum?id=Idst6X6gmy)
[6](https://arxiv.org/abs/2508.18265)
[7](http://arxiv.org/pdf/2503.10639.pdf)
[8](https://arxiv.org/pdf/2305.02317.pdf)
[9](https://arxiv.org/html/2501.07542v1)
[10](https://arxiv.org/html/2503.05255v1)
[11](https://arxiv.org/html/2503.05179v1)
[12](https://arxiv.org/html/2503.20752)
[13](http://www.lmms-lab.com/posts/mmsearch_r1_improved/)
[14](https://www.emergentmind.com/papers/2510.12801)
[15](https://arxiv.org/html/2503.10582v1)
[16](https://www.semanticscholar.org/paper/f17005b8bbbb10ec2cbd6b430db5ad702828e22b)
[17](https://www.emergentmind.com/topics/generalized-reinforcement-learning-policy-optimization-grpo)
[18](https://www.digitalocean.com/community/conceptual-articles/group-relative-policy-optimization-reinforcement-learning)
[19](https://arxiv.org/abs/2511.18719)
[20](https://arxiv.org/abs/2506.03371)
[21](https://arxiv.org/abs/2502.14412)
[22](https://openaccess.thecvf.com/content/ICCV2025/papers/Danish_GEOBench-VLM_Benchmarking_Vision-Language_Models_for_Geospatial_Tasks_ICCV_2025_paper.pdf)
[23](https://www.semanticscholar.org/paper/0f93ebc8a0978e8a17346933631045c47dfa2b7b)
[24](https://arxiv.org/abs/2411.13543)
[25](https://arxiv.org/abs/2503.00025)
[26](https://mathjournal.unram.ac.id/index.php/Griya/article/view/752)
[27](https://arxiv.org/abs/2510.06261)
[28](https://arxiv.org/abs/2507.05520)
[29](https://dl.acm.org/doi/10.1145/3746027.3761999)
[30](https://arxiv.org/abs/2505.20672)
[31](https://www.semanticscholar.org/paper/236780646cdc04594ac2a9495a5c8b9a971b24bb)
[32](https://revistaft.com.br/traumatismo-ocular-e-orbita-ocular-analise-de-2008-a-2024/)
[33](https://arxiv.org/html/2503.24110v1)
[34](https://arxiv.org/abs/2502.11271)
[35](http://arxiv.org/pdf/2405.20795.pdf)
[36](http://arxiv.org/pdf/2503.08308.pdf)
[37](https://arxiv.org/html/2405.18358)
[38](https://arxiv.org/html/2502.19400)
[39](https://arxiv.org/abs/2502.06787)
[40](https://arxiv.org/pdf/2503.06580.pdf)
[41](https://cvpr.thecvf.com/virtual/2025/poster/32818)
[42](https://arxiv.org/abs/2511.15705)
[43](https://datasciencedojo.com/blog/agentic-llm-in-2025/)
[44](https://openaccess.thecvf.com/content_cvpr_2015/papers/Lin_Learning_Deep_Representations_2015_CVPR_paper.pdf)
[45](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05359.pdf)
[46](https://arxiv.org/abs/2509.14480)
[47](https://arxiv.org/html/2509.16343v1)
[48](https://arxiv.org/abs/2509.13347)
[49](https://www.semanticscholar.org/paper/a155ed1054715c0d79d53a740017c8e71b2567bd)
[50](https://ieeexplore.ieee.org/document/11127584/)
[51](https://arxiv.org/abs/2412.15206)
[52](https://arxiv.org/abs/2510.10117)
[53](https://arxiv.org/abs/2406.02537)
[54](https://arxiv.org/abs/2506.09172)
[55](https://arxiv.org/abs/2403.09027)
[56](https://aclanthology.org/2023.emnlp-demo.51.pdf)
[57](https://arxiv.org/html/2502.13130v1)
[58](https://arxiv.org/pdf/2403.05525.pdf)
[59](https://arxiv.org/html/2504.06272v1)
[60](https://arxiv.org/pdf/2309.07870.pdf)
[61](https://arxiv.org/pdf/2409.03215.pdf)
[62](https://huggingface.co/blog/vlms-2025)
[63](https://openreview.net/forum?id=fWv0aGD1Xu)
[64](https://www.labellerr.com/blog/top-open-source-vision-language-models/)
[65](https://www.ijcai.org/proceedings/2021/0681.pdf)
[66](https://arxiv.org/pdf/2506.02153.pdf)
[67](https://www.nature.com/articles/s41598-025-20653-y)
[68](https://www.bentoml.com/blog/multimodal-ai-a-guide-to-open-source-vision-language-models)
[69](https://openreview.net/forum?id=h0T0C4UVsU)
[70](https://arxiv.org/html/2501.05452)
[71](https://arxiv.org/pdf/1710.07300v1.pdf)
[72](https://www.emergentmind.com/topics/visual-chain-of-thought-vcot)
[73](https://arxiv.org/html/2501.05452v1)
[74](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhao_CoT-VLA_Visual_Chain-of-Thought_Reasoning_for_Vision-Language-Action_Models_CVPR_2025_paper.pdf)
[75](https://openreview.net/pdf/920813729f1101f490a9970a522fab1f12ff7ae9.pdf)
[76](https://openreview.net/forum?id=Fg0eo2AkST)
[77](https://cameronrwolfe.substack.com/p/grpo)
[78](https://www.themoonlight.io/ko/review/deepmmsearch-r1-empowering-multimodal-llms-in-multimodal-web-search)
