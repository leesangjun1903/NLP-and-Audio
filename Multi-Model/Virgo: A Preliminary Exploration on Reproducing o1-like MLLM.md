
# Virgo: A Preliminary Exploration on Reproducing o1-like MLLM

## 요약 개요

**Virgo** 논문은 OpenAI의 o1과 유사한 느린 사고(slow-thinking) 능력을 멀티모달 대규모 언어 모델(MLLM)에 구현하는 방법을 탐색합니다. 핵심 발견은 **텍스트 기반의 장문 사고 데이터만으로도 MLLM을 fine-tuning하면 강력한 slow-thinking 추론 능력을 획득할 수 있다**는 것입니다. 이는 멀티모달 모델에서 느린 사고 능력이 본질적으로 언어 모델 컴포넌트와 연결되어 있으며, 모달리티 간에 전이 가능함을 시사합니다.[1]

***

## 1. 핵심 주장 및 주요 기여

### 1.1 연구의 중심 질문

논문은 두 가지 핵심 질문에 대한 답을 제시합니다:[1]

1. **텍스트 기반 장문 사고 데이터를 통한 능력 전이 가능성**: 텍스트 전용 데이터로 fine-tuning하면 MLLM의 slow-thinking 능력이 전이될 수 있는가?

2. **텍스트 vs 시각 데이터의 효과 비교**: 텍스트 기반 데이터로부터 파생된 능력이 시각적 slow-thinking 데이터 증류 방식과 비교하여 얼마나 효과적인가?

### 1.2 주요 발견

| 발견 항목 | 내용 |
|---------|------|
| **텍스트 데이터 효과성** | 5K 텍스트 장문 사고 데이터만으로도 경쟁력 있는 성능 달성[1] |
| **모달리티 전이** | Slow-thinking 능력이 언어 모델 컴포넌트와 근본적으로 연결되어 있으므로 모달리티 간 전이 가능[1] |
| **시각 데이터의 한계** | 멀티모달 데이터(DQVQ, DSD)는 예상보다 텍스트 데이터보다 효과적이지 않음[1] |
| **문제 난이도 상관성** | 더 어려운 벤치마크에서 더 큰 성능 향상 (OlympiadBench +18.1%, MathVision +12.4%)[1] |
| **최적 추론 길이** | 2000-4000 토큰 범위의 추론이 최적이며, 4000-8000 토큰은 성능 저하 유발[1] |

***

## 2. 해결하고자 하는 문제

### 2.1 연구 동기

기존의 slow-thinking 추론 시스템(OpenAI o1, DeepSeek R1, Qwen QwQ) 연구는 다음과 같은 제한사항이 있었습니다:[1]

- **주로 텍스트 기반**: 텍스트 작업에 집중되어 있고 멀티모달 시나리오를 충분히 고려하지 않음
- **멀티모달 시스템 부족**: 멀티모달 slow-thinking 시스템은 o1, QVQ와 같은 상용 시스템에 현저히 뒤떨어짐[1]
- **구조적 복잡성**: MLLM은 시각적 이해(perception)와 추론(reasoning)을 모두 수행해야 하므로 slow-thinking 구현이 더 복잡함[1]

### 2.2 핵심 문제 진술

**MLLM에 slow-thinking 능력을 효율적으로 부여할 수 있는가?** 특히, 텍스트 전용 추론 데이터로 fine-tuning하면 멀티모달 입력에 대해서도 느린 사고 능력이 발현될 수 있는가?[1]

***

## 3. 제안하는 방법론

### 3.1 전략 개요

Virgo는 두 가지 상호보완적인 접근법을 제시합니다:[1]

#### **(1) 텍스트 기반 장문 사고 데이터를 통한 능력 전이**

**데이터 수집:**[1]
- 출처: DeepSeek-R1-Lite-Preview와 QwQ-32B-preview에서 증류
- 규모: 약 5K 인스턴스
- 도메인 구성:
  - 수학(Math): 3.7K
  - 과학(Science): 0.9K
  - 코드(Code): 0.2K
  - 퍼즐(Puzzle): 0.1K

**데이터 포맷:**[1]
```
<|begin_of_thought|>
[장문의 추론 과정]
<|end_of_thought|>
<|begin_of_solution|>
[최종 답변]
<|end_of_solution|>
```

**Fine-tuning 절차:**[1]
- Base 모델: Qwen2-VL-72B-Instruct
- 학습 대상: LLM 파라미터 + 크로스모달 커넥터만 업데이트
- 시각 인코더는 고정
- 학습률: 7e-6
- 배치 크기: 128
- 에폭: 10 (5번째 에폭에서 검증 성능 기반 선택)
- 프레임워크: LLaMA-Factory

#### **(2) Slow-thinking MLLM으로부터 시각 데이터 증류**

**데이터셋 선정:**[1]
- 8개 공개 VQA 데이터셋 활용:
  - **기하학(Geometry)**: Geos (279), GeoQA+ (563), Geometry3K (551), UniGeo (555)
  - **테이블/차트(Tables & Figures)**: TabMWP (568), FigureQA (589), ChartQA (509)
  - **객체(Objects)**: CLEVR (548)

**증류 전략:**[1]
- QVQ로부터 직접 증류 (DQVQ): 6.6K 샘플
- Self-distillation (DSD): fine-tuned Virgo-72B-DT를 이용한 생성 (7K 샘플)
- Rollout 방식으로 복수 응답 생성 및 필터링

**다단계 학습:**[1]
1. 텍스트 데이터(DT)로 초기 fine-tuning → M0 획득
2. M0를 이용해 시각 데이터(DSD) 생성
3. DSD로 원본 MLLM 재학습 (또는 DT ∪ DSD 조합 학습)

### 3.2 모델 구조

```
Virgo 아키텍처:
┌─────────────────────────────────┐
│    Base MLLM                    │
│  Qwen2-VL-72B-Instruct         │
│  ┌──────────┐    ┌──────────┐  │
│  │ Vision   │    │   LLM    │  │
│  │ Encoder  │←→  │Component │  │
│  │(Frozen)  │    │(Fine-tune)   │
│  └──────────┘    └──────────┘  │
│       ↑               ↑         │
│     Images        Text Output   │
└─────────────────────────────────┘
```

**주요 설계 원칙:**[1]
- 시각 인코더는 고정: 기학습된 시각 이해 능력 유지
- LLM 컴포넌트 업데이트: slow-thinking 추론 능력 습득
- 크로스모달 커넥터 학습: 멀티모달 정보 통합 능력 강화

***

## 4. 성능 향상 분석

### 4.1 종합 성능 비교

| 모델 | MathVerse | MathVision | OlympiadBench | MMMU | 평균 |
|------|-----------|-----------|---------------|------|------|
| **GPT-4o** | 30.4 | 25.9 | 69.1 | - | - |
| **Gemini-Pro** | 35.3 | 19.2 | 4.2 | 65.8 | 31.13 |
| **Claude-3.5-Sonnet** | - | 38.0 | - | 70.4 | - |
| **OpenAI o1** | - | - | - | 77.3 | - |
| **QVQ-72B-preview** | 41.5 | 38.2 | 27.9 | 66.0 | 43.40 |
| **Qwen2-VL-72B-Instruct** (baseline) | 41.3 | 26.1 | 11.2 | 64.5 | 35.78 |
| **Virgo-72B (DT)** | **48.4** | **38.8** | **29.9** | 64.6 | **45.43** |
| **Virgo-72B (DT ∪ DSD)** | 48.1 | 38.6 | 28.5 | 65.0 | 45.05 |

[1]

**핵심 성과:**[1]
- 텍스트 데이터만으로 **QVQ를 능가하는 성능** 달성
- 4개 벤치마크 평균 **45.43%** (기존 QVQ 43.40%)
- MathVerse에서 +7.1%, MathVision에서 +12.4% 향상

### 4.2 벤치마크별 상세 분석

#### **4.2.1 난이도별 성능 (MMMU)**

| 난이도 | QVQ | Qwen2-VL | Virgo-72B |
|-------|-----|----------|-----------|
| **Easy** | 76.95% | 74.58% | 72.88% |
| **Medium** | 65.80% | 62.26% | 62.97% |
| **Hard** | 48.62% | 50.28% | **54.70%** |
| **전체** | 66.0% | 64.5% | 64.6% |

[1]

**해석:**
- Hard 문제에서는 Virgo가 우수 (+6.08%)
- Easy/Medium에서는 약간의 성능 저하
- 어려운 추론이 필요한 문제에서 slow-thinking의 가치 입증

#### **4.2.2 추론 길이와 성능 상관성**

```
성능 향상 vs 평균 생각 길이:

높음
   │                    ●
   │                OlympiadBench (+18.1%)
   │           ●
   │        MathVision (+12.4%)
향 │
상 │  ●
도 │ MMMU
   │ (+0.1%)
   │
   └─────────────────────────
      짧음              길음
      (생각 길이)
```



**발견:**
- 평균 생각 길이가 길수록 성능 향상이 큼
- OlympiadBench (가장 긴 평균 생각): 최대 향상
- MMMU (짧은 평균 생각): 미미한 향상

### 4.3 데이터 스케일링 효과

| 데이터량 | 72B 모델 |  |  |  | 7B 모델 |  |  |  |
|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| | MathVerse | MathVision | OlympiadBench | MMMU | MathVerse | MathVision | OlympiadBench | MMMU |
| **1K** | 42.5 | 39.5 | 26.2 | 61.8 | 22.5 | 23.7 | 8.6 | 42.8 |
| **3K** | 44.4 | 40.5 | 26.4 | 58.2 | 30.2 | 24.9 | 9.6 | 44.6 |
| **5K** | **48.4** | **38.8** | **29.9** | **64.7** | **31.9** | **24.6** | **9.2** | **47.1** |

[1]

**결론:**
- 1K → 5K: 평균 7.7% 성능 향상
- OlympiadBench에서는 1.8% 향상 (도메인 특이성)
- 더 많은 데이터가 일반적으로 도움이 되나, 벤치마크 특성에 따라 변동

***

## 5. 한계(Limitations)

### 5.1 명시된 한계

#### **5.1.1 MMMU 벤치마크 성능 부진**

**관찰:**[1]
- MMMU는 상대적으로 낮은 추론량을 요구하는 벤치마크
- Slow-thinking 능력이 오히려 성능 저하를 초래할 수 있음
- 기존 모델(Qwen2-VL-72B): 64.5% vs Virgo: 64.6% (거의 변화 없음)

**원인:**
- 장문 추론이 불필요한 문제에 강제된 느린 사고 적용
- 과도한 사고 프로세스로 인한 확률적 편차 증가

#### **5.1.2 시각 데이터 품질 한계**

**발견:**[1]
- 증류된 시각 데이터(DQVQ, DSD)가 텍스트 데이터보다 효과적이지 않음
- 인간 검토 결과, 많은 시각 질문이 지각(perception) 의존적이고 실제 추론이 부족

**테이블 6: 난이도별 시각 데이터 성능**[1]

| 난이도 | MathVerse | MathVision | OlympiadBench | MMMU |
|-------|-----------|-----------|---------------|------|
| **Medium** | 48.1 | 38.6 | 28.5 | 65.0 |
| **Hard** | 47.4 | 39.1 | 29.7 | 63.0 |
| **Random** | 47.9 | 38.5 | 29.3 | 64.8 |

**결론:** 난이도 간 성능 차이가 미미 → 더 정교한 시각 데이터 합성 전략 필요[1]

#### **5.1.3 인식(Perception) 오류에 대한 반성 능력 부족**

**사례 분석 (테이블 8):**[1]

논문은 Virgo가 **시각 인식 오류에 대한 반성이 부족함**을 지적합니다:

- 모델이 차트에서 실직업자 수를 잘못 읽음
- 추론 과정에서 오류를 감지했으나, **인식 결과 자체의 타당성은 의문하지 않음**
- 동일한 인식 오류에 기반한 결론을 반복 생성

**시사점:** Slow-thinking을 텍스트 데이터에서만 학습하면 시각 정보의 신뢰도 검증 능력이 부족[1]

### 5.2 모델 크기별 차이

**Virgo-7B vs Virgo-72B:**[1]

| 특성 | 72B | 7B |
|------|-----|-----|
| 텍스트 데이터 효율성 | 매우 높음 | 중간 |
| 시각 데이터 효율성 | 낮음 | **중간~높음** |
| MMMU 성능 | 거의 변화 없음 | **급락** (-6.3%) |
| 복잡 추론 처리 | 우수 | 어려움 |

**해석:** 더 작은 모델은 복잡한 장문 사고 처리가 어려우며, 시각 정보가 더 필요할 수 있음[1]

***

## 6. 모달리티 간 일반화 성능

### 6.1 핵심 발견: 언어-중심 전이(Language-Centric Transfer)

논문의 가장 중요한 발견은 **slow-thinking 능력이 기본적으로 언어 모델 컴포넌트와 연결**되어 있다는 것입니다:[1]

```
텍스트 데이터로 학습한 LLM의 느린 사고 능력
           ↓
    멀티모달 입력에도 자동 전이
           ↓
   시각 인식 + 학습된 추론 프로세스
           ↓
   경쟁력 있는 MLLM 성능
```

### 6.2 일반화 메커니즘 분석

#### **6.2.1 긍정적 증거**

**텍스트만으로도 충분한 이유:**[1]
1. MLLM의 핵심 추론은 LLM이 담당
2. 시각 인코더는 이미 이미지 이해에 최적화됨
3. Fine-tuning은 추론 프로세스만 개선하면 됨

**사례 (테이블 7):** Virgo가 기하학 문제에서 우수한 이유[1]
- 이미지에서 반원들의 방정식을 정확히 파악 (시각 능력 - 고정 인코더)
- 방정식에서 적분값 계산 (추론 능력 - fine-tuned LLM)
- 자기 검증을 통한 오류 수정 (slow-thinking - 텍스트로 학습)

#### **6.2.2 부정적 증거 및 한계**

**시각 특화 데이터의 부분적 효과:**[1]
- 텍스트 + 시각(DT ∪ DSD): 거의 개선 없음
- 이유: 시각 데이터의 추론 요구도가 낮음

**도메인 특화성:**[1]
- 수학 중심의 텍스트 데이터 → 수학 벤치마크에서 최적
- 다양한 도메인 벤치마크에서 완전히 동일한 효과 없음

### 6.3 일반화 조건

**효과적인 일반화를 위한 조건:**[1]

| 조건 | 설명 | 영향 |
|------|------|------|
| **베이스 모델 품질** | Qwen2-VL-72B의 강력한 멀티모달 능력이 전제 | 필수적 |
| **추론 복잡도** | 장문 추론이 필요한 문제 | +향상도↑ |
| **데이터 도메인 일치** | 텍스트 데이터의 도메인과 평가 벤치마크 일치도 | +효율성↑ |
| **모델 크기** | 복잡 추론 처리 능력 (72B > 7B) | +안정성↑ |

***

## 7. 2020년 이후 관련 최신 연구 비교

### 7.1 연구 진화 흐름

```
2020-2023: Chain-of-Thought (CoT)의 등장
  ↓ (텍스트 기반 추론 단계별 분해)
  
2024년 초: Slow-thinking 시스템의 부상
  ↓ (o1, DeepSeek R1, QwQ 공개)
  
2024년 중후반: 멀티모달 확장 시도
  ├─ Virgo (텍스트→MLLM 전이)
  ├─ Insight-V (시각 CoT 데이터 생성)
  └─ LlamaV-o1 (구조화된 시각 추론)
  
2025년: RL 기반 멀티모달 slow-thinking
  ├─ Vision-R1 (RL + 강화학습)
  ├─ DAST (난이도-적응형 slow-thinking)
  └─ Select2Reason (효율적 데이터 선택)
```

### 7.2 주요 선행 연구와의 비교

#### **A. STILL-2 (Min et al., 2024)**[2]

| 차원 | Virgo | STILL-2 |
|-----|-------|--------|
| **대상** | MLLM | LLM (텍스트) |
| **핵심 방법** | Fine-tuning | Imitate-Explore-Self-improve |
| **학습 알고리즘** | SFT (감독 학습) | RL + 자기개선 |
| **데이터량** | 5-12K | 1,100 (seed) → 자동 생성 |
| **성능** | QVQ 능가 (평균 45.43%) | 산업 수준 경쟁력 |
| **복잡도** | 낮음 (간단한 fine-tuning) | **높음** (3단계 RL 사이클) |

**상대적 장점:**
- Virgo: **단순하고 실용적**, MLLM에 즉시 적용 가능
- STILL-2: **더 강력한 성능**, 일반적 도메인 전이 능력[2]

#### **B. Insight-V (Dong et al., 2024)**[3]

| 차원 | Virgo | Insight-V |
|-----|-------|----------|
| **데이터 생성** | QVQ로부터 증류 (제약적) | 자동 파이프라인 + 품질 검증 |
| **데이터 특성** | 시각 데이터 효율성 낮음 | **장문 구조화 시각 데이터** |
| **훈련 파이프라인** | 단일 단계 | **다중 에이전트 시스템** |
| **DPO/RL 사용** | 미사용 | **반복적 DPO** 적용 |
| **평가 벤치마크** | MathVerse, MathVision 등 | LLaVA-Next 기반 (다양한 시각 작업) |

**차이점:** Virgo는 텍스트 전이, Insight-V는 시각 특화 데이터 생성 중심[3]

#### **C. LlamaV-o1 (2026)**[4]

| 차원 | Virgo | LlamaV-o1 |
|-----|-------|----------|
| **구조** | Fine-tuning 기반 | **구조화된 단계별 학습** |
| **핵심 기여** | 텍스트→멀티모달 전이 | 다단계 시각 추론 구조 |
| **속도** | 기준선 (5배 느림 보도) | **5배 빠른 추론** |
| **성능** | Llava-CoT 대비 3.8% 향상 | **평균 67.3%** (더 높음) |

**차이:** Virgo는 단순 fine-tuning, LlamaV-o1은 구조적 개선[4]

#### **D. Vision-R1 (Huang et al., 2025)**[5]

| 차원 | Virgo | Vision-R1 |
|-----|-------|-----------|
| **학습 방식** | SFT (감독 학습) | **강화학습 (RL)** |
| **데이터 생성** | 직접 증류 | 모달리티 브리징 + 자동 필터링 |
| **혁신** | 텍스트→멀티모달 전이 발견 | **RL에서 자발적 reasoning 창발** |
| **성능** | 평균 45.43% | **MathVista 73.5%** (o1에 매우 근접) |
| **복잡도** | **낮음** (간단) | 높음 (GRPO + PTST) |

**트렌드:** RL 기반 접근이 더 강력한 성능 제공[5]

### 7.3 연구 패러다임 비교표

| 연구 | 연도 | 주요 방법 | 효율성 | 성능 | 적용 용이성 |
|-----|------|---------|--------|------|-----------|
| **Chain-of-Thought 연구들** | 2020-2023 | 프롬팅 | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **STILL-2** | 2024.12 | RL + 자기개선 | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **Virgo** | 2025.01 | 텍스트 SFT | **⭐⭐⭐⭐** | ⭐⭐⭐ | **⭐⭐⭐⭐⭐** |
| **Insight-V** | 2024.11 | 시각 데이터 + DPO | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **LlamaV-o1** | 2026.01 | 구조화 학습 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Vision-R1** | 2025.03 | RL + 강화학습 | ⭐ | **⭐⭐⭐⭐⭐** | ⭐⭐ |

***

## 8. 기술적 수식 및 메커니즘

### 8.1 학습 목표 함수

논문에서 명시적인 수학적 공식을 제시하지는 않지만, 다음과 같이 해석할 수 있습니다:

#### **감독 학습 목표:**

$$\mathcal{L}_{SFT} = -\frac{1}{N} \sum_{i=1}^{N} \log P_\theta(y_i^* | x_i, v_i)$$

여기서:
- $$N$$: 훈련 샘플 수
- $$x_i$$: 텍스트 입력
- $$v_i$$: 시각 입력
- $$y_i^*$$: 장문 사고 형식의 정답 (`<|begin_of_thought|>...<|end_of_thought|><|begin_of_solution|>...</|end_of_solution|>`)
- $$P_\theta(\cdot)$$: MLLM의 다음 토큰 확률

#### **파라미터 업데이트 대상:**

$$\Theta_{update} = \{W_{LLM}, W_{connector}\}$$
$$\Theta_{frozen} = \{W_{vision\_encoder}\}$$

비전 인코더의 가중치 $$W_{vision\_encoder}$$는 고정되어 있으므로, LLM 컴포넌트 $$W_{LLM}$$과 크로스모달 커넥터 $$W_{connector}$$만 학습됩니다.

### 8.2 다단계 학습 프로세스

#### **Phase 1: 텍스트 기반 초기화 (Text Initialization)**

$$M_0 = \text{FineTune}(\text{Qwen2-VL}, D_T, \theta_{SFT})$$

여기서 $$D_T$$는 텍스트 장문 사고 데이터셋:

$$D_T = \{(q_i, \langle \text{thought}_i, \text{solution}_i \rangle)\}_{i=1}^{5K}$$

#### **Phase 2: 자기-증류 (Self-Distillation)**

시각 질문에 대해 M0로 다중 시도(rollout)를 수행:

$$D_{SD} = \{(q_v, I, \text{best}(\text{Rollout}(M_0, q_v, I, n_{rollout})))\}$$

여기서:
- $$q_v$$: 시각 질문
- $$I$$: 이미지
- $$n_{rollout}$$: 롤아웃 횟수 (문제 난이도에 따라 조정)

#### **Phase 3: 결합 학습 (Combined Training)**

$$M_{final} = \text{FineTune}(\text{Qwen2-VL}, D_T \cup D_{SD}, \theta_{SFT})$$

또는 단순히 $$D_T$$만 사용:

$$M_{final} = \text{FineTune}(\text{Qwen2-VL}, D_T, \theta_{SFT})$$

### 8.3 최적 추론 길이 분석

논문의 발견을 수식화하면:

$$\text{Optimal Thought Length} \in [2000, 4000) \text{ tokens}$$

성능 함수:

$$\text{Accuracy}(l) = f(l) \text{ where } l \in \text{response length}$$

특성:
- $$l \in [0, 2000)$$: 부족한 추론으로 정확도 낮음
- $$l \in [2000, 4000)$$: 최적 범위, 최고 정확도
- $$l \in $$: 과도한 추론으로 정확도 감소 (hallucination 증가)

데이터 난이도 분포:

$$p(l|d) = \text{Difficulty-weighted distribution}$$

여기서:
- 어려운 문제(d=hard): 대부분 4000 미만의 추론 필요
- 쉬운 문제(d=easy): 2000 미만의 추론으로 충분

***

## 9. 앞으로의 연구에 미치는 영향

### 9.1 패러다임 전환

#### **9.1.1 MLLM 설계 철학의 변화**

**이전 접근:**
- 시각 인코더 + LLM의 단순 조합
- 시각과 언어의 동등한 업그레이드 필요

**Virgo 이후:**
- **LLM이 slow-thinking의 핵심** 역할
- 시각 인코더는 고정, LLM만 fine-tune으로도 충분[1]
- 멀티모달 성능 향상 = 주로 LLM 개선[1]

**실무 함의:**
```python
# 이전 방식
fine_tune(vision_encoder) + fine_tune(llm)

# Virgo 방식 (더 효율적)
fine_tune(llm)  # 비전 인코더는 고정
```

#### **9.1.2 데이터 수집 전략의 재평가**

**기존 신념:**
- 멀티모달 slow-thinking은 "멀티모달 데이터"가 필수

**Virgo의 발견:**
- 텍스트 데이터만으로 충분 또는 더 효과적[1]
- 시각 데이터 합성은 실제 성능 향상으로 이어지지 않을 수 있음[1]

**연구 방향:**
- **텍스트 데이터 품질 향상**에 더 많은 자원 투입
- 시각 데이터는 보충적 역할로 재평가

### 9.2 기술 트렌드에 미치는 영향

#### **9.2.1 효율적 추론 모델 개발**

**문제 제기:**[1]
- OpenAI o1은 강력하지만 추론 토큰 비용 높음
- MLLM에 slow-thinking을 추가할 때 비용-성능 트레이드오프

**Virgo의 시사점:**
- 5K 샘플 fine-tuning으로 큰 성능 향상 가능
- **매우 저비용의 방법**으로 slow-thinking 능력 부여 가능[1]

**실무 응용:**
```
Edge 디바이스에서의 slow-thinking:
1. 경량 MLLM 사용
2. 텍스트 CoT 데이터로만 fine-tune
3. 강력한 추론 능력 획득 가능
```

#### **9.2.2 도메인 특화 모델 개발**

**발견:**[1]
- 도메인별 텍스트 데이터로 fine-tune하면 해당 도메인에서 최적화
- 예: 수학 텍스트 데이터 → 수학 비전 문제에서 우수

**응용 가능성:**
| 도메인 | 최적화 방법 |
|-------|-----------|
| **의료 진단** | 의료 텍스트 CoT 데이터 + 의료 이미지 |
| **산업 검사** | 기술 텍스트 데이터 + 제품 이미지 |
| **과학 연구** | 과학 논문의 추론 방식 + 실험 데이터 |

***

## 10. 앞으로 연구 시 고려할 점

### 10.1 기술적 과제 (Technical Challenges)

#### **10.1.1 인식-추론 분리 문제**

**현재 한계:**[1]
- 시각 인식 오류에 대한 반성 부족
- 잘못된 인식에 기반한 추론이 반복됨

**해결 방안:**
```
개선 아이디어:

(1) 다중 시도 검증
    - 동일 이미지에서 여러 추론 경로 생성
    - 불일치 발견 시 인식 재검토 강제

(2) 지각-추론 분리 학습
    - Perception verification 모듈 추가
    - "이 인식이 맞는가?" 단계 명시화

(3) 비전-언어 alignment 강화
    - 텍스트 + 시각 CoT 데이터 모두 사용
    - 인식 오류에 대한 명시적 수정 데이터 포함
```

#### **10.1.2 모델 크기별 차별화**

**문제:**[1]
- 7B 모델과 72B 모델의 데이터 효율성 차이 큼
- 작은 모델에 텍스트 데이터 적용 시 오히려 성능 저하

**권장사항:**
```
모델 크기별 최적 전략:

Small (7B 이하):
  ✓ 시각 데이터 비중 높임
  ✓ 짧은 추론 길이 (1000-2500 토큰)
  ✓ 추론 깊이 제한

Medium (13B-30B):
  ✓ 균형잡힌 텍스트-시각 혼합
  ✓ 중간 추론 길이 (2000-4000 토큰)

Large (70B+):
  ✓ 텍스트 데이터 우선
  ✓ 최대 추론 길이 (3000-5000 토큰)
  ✓ 도메인 특화 데이터 활용
```

#### **10.1.3 MMMU 벤치마크의 과제**

**현상:**[1]
- 지식 집약적 문제 (MMMU)에서 slow-thinking의 성능 향상 미미
- 추론이 아닌 지식 회상(knowledge recall) 중심

**대응 전략:**
```
1. 이원화 접근:
   - 추론 필요 문제: slow-thinking 적용
   - 지식 필요 문제: 일반 모드 유지
   
2. 적응형 추론 깊이:
   - 문제 분류: "이 문제는 추론이 필요한가?"
   - 필요시에만 slow-thinking 활성화
   - 계산 비용 절감
```

### 10.2 데이터 관련 고려사항 (Data Considerations)

#### **10.2.1 시각 데이터 품질 향상**

**논문의 발견:**[1]
- 현재 시각 데이터 대부분 "인식 중심" → "추론 중심"으로 개선 필요

**개선 방법:**
```
High-quality 시각 CoT 데이터 생성:

1. 자동 필터링:
   - Q: "이 문제가 pure visual reasoning인가?"
   - 필터: perception 개입 최소화
   
2. 수동 큐레이션:
   - 전문가가 진정한 "시각 추론" 문제 선별
   - 도형 회전, 공간 관계 등 (순수 시각 추론)
   
3. 도메인 특화:
   - 기하학: 도형의 공간 관계
   - 의료: 영상 해석의 추론 과정
   - 과학: 현상의 인과 관계 분석
```

#### **10.2.2 데이터 효율성 연구**

**열린 질문:**
- "최소 몇 개의 샘플이 필요한가?"
- 현재: 5K 샘플로 충분[1]
- 극한: 수백 개로는 가능한가?

**제안 연구:**
```
Data Efficiency Study:
- 100, 500, 1K, 2K, 5K, 10K 샘플로 실험
- 수렴 곡선(convergence curve) 분석
- 데이터 다양성 vs 양의 상충관계 조사
```

### 10.3 평가 방법론의 개선 (Evaluation Improvements)

#### **10.3.1 프로세스 기반 평가**

**현재 방식:** Outcome-only (최종 답만 평가)

**제안:** 추론 과정 평가도 포함
```
멀티모달 slow-thinking 평가 프레임워크:

Dimension 1: 최종 정답 정확도
Dimension 2: 추론 논리 타당성
Dimension 3: 시각 인식 정확도
Dimension 4: 추론-인식 통합 적절성

총점 = W₁×정답 + W₂×논리 + W₃×인식 + W₄×통합
```

#### **10.3.2 강건성(Robustness) 평가**

**고려사항:**
- 이미지 회전, 왜곡 시 모델 성능
- 노이즈 추가 시 견딜 수 있는 정도
- 드물지만 중요한 케이스 처리

```
Robustness Benchmark 예시:

Base: 이미지 원본
Variant 1: ±10도 회전
Variant 2: 명도 70-130% 조정
Variant 3: 가우시안 노이즈 추가
```

### 10.4 이론적 이해 심화 (Theoretical Understanding)

#### **10.4.1 "왜 텍스트 데이터가 더 효과적인가?"의 근본**

**가설 검증이 필요한 영역:**

| 가설 | 검증 방법 | 기대 결과 |
|------|---------|---------|
| **언어 모델의 우월한 표현력** | 다양한 LLM 백본 실험 | 가설 확인/기각 |
| **시각 정보의 중복성** | 이미지 처리 필요도 분석 | 수량화 |
| **도메인 일치성** | Cross-domain 전이 실험 | 일반화 원리 규명 |

#### **10.4.2 전이 학습의 메커니즘**

```
연구 방향:

Q: Slow-thinking 능력이 정확히 무엇이 전이되는가?

가능한 메커니즘:
1. 패턴 인식 (pattern recognition)
2. 오류 검증 능력 (error detection)
3. 백트래킹 기술 (backtracking)
4. 반성 능력 (reflection)
5. 장기 계획 (long-horizon planning)

분석: 어떤 메커니즘이 주요 역할?
```

### 10.5 실용적 배포 전략 (Deployment Considerations)

#### **10.5.1 계산 비용 최적화**

**동적 추론 깊이 조정:**
```python
def adaptive_reasoning_depth(question, image):
    complexity_score = estimate_complexity(question, image)
    
    if complexity_score < 0.3:  # 쉬운 문제
        max_thought_tokens = 1000  # 짧은 추론
    elif complexity_score < 0.7:  # 중간 난이도
        max_thought_tokens = 2500  # 중간 추론
    else:  # 어려운 문제
        max_thought_tokens = 4000  # 최대 추론
    
    return generate_response(question, image, max_thought_tokens)
```

#### **10.5.2 비용-성능 트레이드오프 최적화**

```
배포 시나리오별 권장 설정:

High-Accuracy Applications (의료, 금융):
  • 72B 모델 + 5K 텍스트 데이터
  • 최대 추론 깊이 (4000 토큰)
  • 검증 단계 포함
  • Cost: 높음 | Accuracy: 최고

Balanced Applications (고객지원):
  • 13B-30B 모델 + 2K-3K 데이터
  • 중간 추론 깊이 (2500 토큰)
  • Cost: 중간 | Accuracy: 우수

Cost-Sensitive Applications (모바일):
  • 7B 모델 + 시각 데이터 중심
  • 짧은 추론 (1500 토큰)
  • Cost: 낮음 | Accuracy: 합리적
```

***

## 11. 종합 평가 및 결론

### 11.1 Virgo의 기여도 평가

| 관점 | 평가 | 근거 |
|------|------|------|
| **혁신성** | ⭐⭐⭐⭐ | 텍스트→멀티모달 전이의 발견이 새로움[1] |
| **실용성** | ⭐⭐⭐⭐⭐ | 간단한 fine-tuning으로 강력한 성능[1] |
| **일반화성** | ⭐⭐⭐ | 주로 수학 관련 작업에 강함[1] |
| **완성도** | ⭐⭐⭐ | 시각 데이터 한계, 인식-추론 분리 미흡[1] |
| **재현성** | ⭐⭐⭐⭐⭐ | 공개 코드/데이터 제공[1] |

### 11.2 연구의 한계와 보완 필요성

**핵심 한계:**
1. **시각 데이터의 효과 부족**: 멀티모달 특성을 충분히 활용하지 못함[1]
2. **인식 오류 대응**: Perception 오류에 대한 자기 수정 능력 부족[1]
3. **모델 크기 의존성**: 7B 모델에서는 성능 저하 가능[1]
4. **도메인 협소**: 주로 수학 중심의 벤치마크에서만 검증[1]

**필요한 후속 연구:**
- 더 정교한 시각 slow-thinking 데이터 생성 방법
- 인식-추론 통합 검증 메커니즘
- 다양한 도메인(의료, 법률, 기술)에서의 적용 연구
- RL 기반 강화 학습과의 조합

### 11.3 학문적 의의

**Virgo가 제시한 핵심 인사이트:**

> "Slow-thinking 능력은 기본적으로 **언어 모델의 고유한 특성**이며, 멀티모달 입력에 대해서도 이 능력이 자동으로 전이될 수 있다."[1]

이 발견은 다음과 같은 새로운 연구 방향을 열었습니다:

1. **LLM-centric 설계**: 시각 모듈은 인식만, 추론은 LLM이 담당
2. **효율적 확장**: 강화학습 없이도 단순 fine-tuning으로 능력 부여
3. **도메인 전이**: 한 도메인의 텍스트 데이터가 다른 도메인의 멀티모달 작업에 전이 가능

### 11.4 최종 결론

**Virgo의 위치:**
- **현재**: MLLM slow-thinking 분야의 **초기 탐험자**
- **기여**: 실용적이고 효율적인 방법 제시
- **한계**: 완전한 멀티모달 활용은 미달
- **미래**: 더 정교한 멀티모달 slow-thinking 시스템으로 발전할 기초 제공

**연구의 의미:**
Virgo는 "멀티모달 slow-thinking은 비전 인코더 업그레이드가 아니라 **LLM의 추론 능력 강화**"라는 중요한 통찰을 제공함으로써, 이후 Vision-R1, LlamaV-o1 등의 더욱 정교한 연구로의 발판이 되었습니다.[2][4][5][1]

***

## 참고자료 및 인용

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9313a29d-8122-41b3-a5a2-1abc2488c7b5/2501.01904v2.pdf)
[2](https://arxiv.org/abs/2412.09413)
[3](https://arxiv.org/abs/2410.09918)
[4](https://www.mdpi.com/2073-431X/13/11/278)
[5](https://arxiv.org/abs/2409.17433)
[6](https://ieeexplore.ieee.org/document/10655637/)
[7](https://arxiv.org/abs/2410.23743)
[8](https://arxiv.org/abs/2410.01792)
[9](http://medrxiv.org/lookup/doi/10.1101/2024.09.25.24314342)
[10](https://www.semanticscholar.org/paper/b190ff60af7ee4ca177ae67d8eda91e00a24f121)
[11](https://www.cureus.com/articles/301598-openai-o1-preview-vs-chatgpt-in-healthcare-a-new-frontier-in-medical-ai-reasoning)
[12](https://arxiv.org/html/2503.22732v1)
[13](http://arxiv.org/pdf/2502.10867.pdf)
[14](https://arxiv.org/pdf/2412.12173.pdf)
[15](http://arxiv.org/pdf/2502.12853.pdf)
[16](https://arxiv.org/pdf/2412.21187.pdf)
[17](http://arxiv.org/pdf/2410.01792.pdf)
[18](https://arxiv.org/html/2503.15944)
[19](https://arxiv.org/pdf/2502.06807.pdf)
[20](https://www.sciencedirect.com/science/article/abs/pii/S0306457325003358)
[21](https://liner.com/review/visual-representation-alignment-for-multimodal-large-language-models)
[22](https://www.kore.ai/blog/chain-of-instructions-coi-fine-tuning)
[23](https://openai.com/index/learning-to-reason-with-llms/)
[24](https://openreview.net/forum?id=eCElREEUsr)
[25](https://arxiv.org/abs/2505.17266)
[26](https://huggingface.co/papers/2501.06186)
[27](https://www.emergentmind.com/topics/multimodal-large-language-model-mllm)
[28](https://proceedings.neurips.cc/paper_files/paper/2024/file/00d80722b756de0166523a87805dd00f-Paper-Conference.pdf)
[29](https://leehanchung.github.io/blogs/2024/10/08/reasoning-understanding-o1/)
[30](https://arxiv.org/abs/2411.01173)
[31](https://aclanthology.org/2025.naacl-long.584/)
[32](https://arxiv.org/html/2412.09413v2)
[33](https://www.themoonlight.io/ko/review/enhancing-visual-reasoning-with-autonomous-imagination-in-multimodal-large-language-models)
[34](https://liner.com/review/on-impact-finetuning-on-chainofthought-reasoning)
[35](https://arxiv.org/pdf/2412.09413.pdf)
[36](https://arxiv.org/abs/2503.06749)
[37](https://arxiv.org/html/2505.17266v3)
[38](https://arxiv.org/html/2503.04472v3)
[39](https://arxiv.org/abs/2411.14432)
[40](https://arxiv.org/pdf/2505.17266.pdf)
[41](https://arxiv.org/html/2502.10867v1)
[42](https://arxiv.org/abs/2504.15279)
[43](https://arxiv.org/abs/2407.00092)
[44](https://arxiv.org/pdf/2509.13351.pdf)
[45](https://arxiv.org/html/2505.02665v1)
[46](https://arxiv.org/abs/2409.15310)
[47](https://arxiv.org/html/2407.03181v1)
