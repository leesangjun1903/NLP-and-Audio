
# O1 Replication Journey – Part 2: Surpassing O1-preview through Simple Distillation Big Progress or Bitter Lesson?

## 1. 논문의 핵심 주장과 주요 기여 요약

본 논문은 OpenAI의 O1 모델을 재현하기 위한 노력에서 광범위하게 사용되고 있는 **지식 증류(Knowledge Distillation)** 기법의 효과성과 한계를 체계적으로 검증합니다. 핵심 주장은 다음과 같습니다:

### 1.1 주요 기여

**기술적 기여:**
O1의 API로부터 직접 지식을 추출하는 단순한 증류 방식과 감독 미세조정(Supervised Fine-Tuning, SFT)을 결합하면, 수만 개 수준의 증류 샘플만으로도 **O1-preview보다 우수한 성능을 달성할 수 있음**을 입증했습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

**메타과학적 기여:**
O1 재현 연구의 투명성과 재현성을 평가하기 위한 **기술 투명성 지수(Technical Transparency Index, TTI)** 라는 포괄적인 평가 프레임워크를 제시했습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

**윤리적/교육적 기여:**
단순 지식 증류에 대한 과도한 의존이 초래할 수 있는 심각한 부작용들(연구 문화 붕괴, 기초 이론 개발 침체, 미래 AI 연구자의 역량 저하)을 경고하는 **"쓸쓸한 교훈(Bitter Lesson)"**을 제시했습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

***

## 2. 해결하고자 하는 문제와 제안 방법

### 2.1 해결하는 핵심 문제

**1) 연구 투명성의 부재**
- 많은 기관들이 O1 재현에 성공했다고 주장하지만, 실제로는 지식 증류 기법을 사용하면서도 이를 명시하지 않음
- 이로 인해 필드의 진정한 기술 진보를 정확히 평가하기 어려움

**2) 성능 천장 효과(Ceiling Effect)**
- 증류 기반 모델은 자신의 교사 모델(O1)을 초과할 수 없음
- 새로운 영역으로의 확장 및 기존 벤치마크 초월의 어려움

**3) 장기적 혁신 정체**
- 기초 기술 연구보다 프롬프트 엔지니어링에 의존하는 경향 증대
- 차세대 AI 연구자들의 첫 원리 사고(First-Principles Thinking) 능력 약화

### 2.2 제안하는 방법론

#### **2.2.1 지식 증류 방식의 기술 파이프라인**

논문은 Part 1의 "Journey Learning" 방식을 기반으로, 더 단순한 증류 접근 방식을 제시합니다:

$$\text{Distillation Process} = \{O1 \text{ Prompting} \to \text{Long-Thought Generation} \to \text{SFT}\}$$ [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

**구체적인 단계:**

1. **O1 API를 통한 장시간 사고 생성(Long-Thought Synthesis)**
   - 수학 문제를 O1에 입력하여 장시간 사고 과정 생성
   - 반영(reflection)과 오류 수정 단계 포함

2. **데이터 큐레이션(Post-training Data Curation)**
   - 이미지 의존 문제 제거
   - 증명(proof) 기반 문제 필터링
   - 수치형 답을 갖는 문제만 유지

3. **재포매팅 기술(Reformatted Technology)**
   - GPT-4o-mini를 통해 원본 솔루션 재작성
   - 단계별, 상세한, 길이가 긴 형식으로 표준화
   - 최종 답을 `\boxed{}` 형식으로 명시

4. **감독 미세조정(Supervised Fine-Tuning)**
   - 초기 SFT: 장시간 형식에 모델 적응
   - 후속 SFT: 증류 데이터셋으로 추가 미세조정

#### **2.2.2 대안 방식과의 비교**

$$\text{Cost-Effectiveness} : \text{Distillation} < \text{Tree Search} < \text{Human Annotation}$$ [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

논문에서는 네 가지 장시간 사고 합성 방법을 비교합니다:

| 방법 | 비용 | 품질 | 스케일성 |
|------|------|------|---------|
| Human Annotation | 높음 | 우수 | 낮음 |
| Tree Search (MCTS) | 높음 | 우수 | 중간 |
| Multi-Agent Debate | 중간 | 중간 | 중간 |
| Distillation | 낮음 | 우수 | 높음 |

### 2.3 모델 구조

**기본 모델:** Qwen2.5-Math-72B [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)
- 72억 개 매개변수
- 수학 추론에 강력한 기초 성능
- 장시간 형식 생성 가능성

**학습 프레임워크:**

$$L = \mathcal{L}_{\text{SFT}}(y_{\text{model}}, y_{\text{distilled}})$$

여기서:
- $y_{\text{model}}$ = 모델의 생성 텍스트
- $y_{\text{distilled}}$ = O1에서 증류된 목표 텍스트

***

## 3. 성능 향상과 한계

### 3.1 수학 추론 성능

#### **AIME2024 벤치마크**

| 모델 | 정확도 | 평균 토큰 수 |
|------|--------|--------------|
| O1-preview | 12/30 (40%) | 9,083 |
| **Ours (72B)** | **13/30 (43.3%)** | 8,016 |
| O1-mini | 21/30 (70%) | 9,903 |

**분석:** 
- 모델이 O1-preview를 초과했지만, O1-mini의 70%에는 미치지 못함
- 더 짧은 토큰 길이로도 경쟁 수준의 성능 달성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

#### **MATH500 벤치마크**

| 모델 | 정확도 | 평균 토큰 수 |
|------|--------|--------------|
| O1-preview | 85.5% | 1,501 |
| **Ours (72B)** | **87.2%** | 2,235 |
| O1-mini | 90.0% | 944 |

**성능 분석 특징:**
- 더 긴 사고 과정으로 더 높은 정확도 달성
- 추론 시간 확장(Inference-Time Scaling)의 이점 활용 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

### 3.2 일반화 성능

#### **3.2.1 안전성(Safety)**

수학 문제 해결에만 훈련했음에도 불구하고, 안전성 벤치마크에서 향상을 보였습니다:

$$\text{안전 성능} : \text{Baseline} = 91.0\% \to \text{Ours} = 92.5\% \text{ (Flames)}$$ [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

**결과 해석:**
- Flames 데이터셋: 91.0% → 92.5% (개선)
- DiaSafety: 100% → 100% (유지)
- WildSafety: 92.0% → 86.5% (감소)

**사례 연구:** 전자 자전거 관련 질문에서
- 기본 모델: 보안 조치(잠금)에만 집중
- 미세조정 모델: 화재 위험 등 생명 위협을 우선 파악 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

이는 **체계적 사고 능력의 전이(Transfer of Systematic Thinking)**를 보여줍니다.

#### **3.2.2 할루시네이션(Hallucination) 감소**

$$\text{Sycophancy 개선} : 89.70\% \to 92.65\% \text{ (CFE-Sycophancy)}$$ [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

| 데이터셋 | 기본 모델 | 미세조정 모델 | 변화 |
|---------|---------|-------------|------|
| SimpleQA | 10.58% | 10.41% | -0.17% |
| C-SimpleQA | 47.08% | 45.76% | -1.32% |
| CFE-General | 69.08% | 62.65% | -6.43% |
| **CFE-Sycophancy** | **89.70%** | **92.65%** | **+2.95%** |

**중요한 발견:**
- 자기 반영 과정(Self-Reflection)이 잘못된 가정 감지에 도움
- 예: 진주강이 중국 2번째 긴 강이라는 잘못된 가정 교정 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

#### **3.2.3 일반 질의응답(General QA)**

$$\text{Auto-J 성능} : 81.6\% \to 88.0\% \text{ (+6.4 포인트)}$$

$$\text{LIMA 성능} : 77.2\% \to 87.2\% \text{ (+10 포인트)}$$ [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

**특징:**
- 수학 문제 해결만으로 훈련했음에도 일반 영역으로 우수한 전이
- 체계적 사고 패턴과 장시간 형식 생성 능력의 일반화

### 3.3 주요 한계

**1) 성능 천장**
- O1-mini 대비 약 30% 성능 격차
- 증류 기반의 본질적 한계 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

**2) 불완전한 생성물**
- 생성된 장시간 솔루션에 여전히 오류 존재
- 명확성과 정확성 개선 필요

**3) 할루시네이션 문제**
- 팩트기반 작업에서는 개선 미미 (SimpleQA: 10.58% → 10.41%)
- 검색 엔진 시뮬레이션으로 인한 추가 할루시네이션 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

**4) 안전성의 이중성**
- WildSafety에서 성능 저하 (92% → 86.5%)
- 장시간 사고만으로는 포괄적 안전 정렬 불충분

***

## 4. 모델 일반화 성능 심층 분석

### 4.1 일반화 메커니즘

#### **4.1.1 체계적 사고의 전이**

논문의 가장 중요한 발견 중 하나는 **수학 문제 해결에 특화된 훈련이 다른 영역으로 일반화된다**는 것입니다:

$$\text{Generalization Factor} = \frac{\text{Performance on OOD Tasks}}{\text{Performance on In-Distribution Tasks}} > 1.0$$ [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

이는 다음 세 가지 메커니즘으로 설명됩니다:

**1) 장시간 형식 적응**
- 모델이 \(<\text{question}, long-thought}, answer>\> 형식을 학습
- 이 형식이 다양한 작업에 유효

**2) 자기 반영 능력**
- 오류 감지 및 수정 패턴 습득
- 다양한 도메인의 오류에 적용 가능

**3) 체계적 분석 패턴**
- 문제를 하위 단계로 분해하는 능력
- 단계별 추론 능력의 전이 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

#### **4.1.2 사례 연구: Python asyncio 문제**

프롬프트: "Python에서 await asyncio.sleep()이 stuck인 이유는?"

**기본 모델(Qwen2.5):**
- 다섯 가지 주요 이유와 기본 설명
- 깊이 부족, 최적화 논의 부재

**미세조정 모델(Our Model):**
- 계층적 구조 및 논리적 흐름
- 고급 주제(체계적 디버깅, 이벤트 루프 관리) 포함
- 실전 가치 있는 조언과 문서 참조 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

### 4.2 일반화의 한계

#### **4.2.1 팩트 기반 작업에서의 한계**

수학 추론 학습이 팩트 회상 정확도를 높이지 못하는 이유:

$$P(\text{Factual Accuracy} | \text{Math CoT Training}) \approx P(\text{Baseline})$$ [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

**분석:**
- 모델이 검색 엔진 사용을 시뮬레이션하려 시도
- 실제 외부 도구 없이 할루시네이션 발생
- 이는 기초 지식의 부족이 아니라 **구조화된 사고의 오류 전파** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

#### **4.2.2 안전성의 양면성**

| 안전 벤치마크 | 성능 변화 | 해석 |
|------------|---------|------|
| Flames (깊은 사고 요구) | +1.5% | 체계적 사고 활용 |
| WildSafety (현실적 시나리오) | -5.5% | 일반화 부족 |

**결론:** 체계적 사고 능력만으로는 포괄적 안전 정렬에 불충분

***

## 5. 최신 연구와의 비교 분석 (2020-2025)

### 5.1 연쇄 사고(Chain-of-Thought) 진화

#### **5.1.1 기본 CoT (2022)**

원본 CoT 논문(Wei et al., 2022)
- **방식:** "한 단계씩 생각해보자" 프롬프트로 중간 추론 단계 유도
- **성능:** 540B GPT-3에서 GSM8K 88% 달성
- **한계:** 프롬프트 의존성, 모델 크기에 따른 효과 변동성

#### **5.1.2 CoT 디코딩 (2024)**

최신 CoT 디코딩 방식 [leehanchung.github](https://leehanchung.github.io/blogs/2024/10/08/reasoning-understanding-o1/)
- **혁신:** 프롬프트 없이 상위-k 토큰 경로 탐색으로 CoT 자동 추출
- **성능:** 기본 그리디 디코딩 대비 향상
- **특징:** 내재된(Intrinsic) 추론 능력 평가 가능 [leehanchung.github](https://leehanchung.github.io/blogs/2024/10/08/reasoning-understanding-o1/)

#### **5.1.3 O1 논문의 위치**

본 논문은 **CoT를 학습 가능한 내재 능력으로 전환**하는 중간 단계 제시:

$$\text{Prompting CoT} \to \text{Training for Long CoT} \to \text{O1-style Reasoning}$$ [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

### 5.2 지식 증류의 발전

#### **5.2.1 표준 지식 증류 (2015-2023)**

원본 지식 증류(Hinton et al., 2015) [arxiv](https://arxiv.org/pdf/2504.01857.pdf)
- **개념:** 대형 모델(교사)의 지식을 소형 모델(학생)로 전이
- **방식:** KL 발산 최소화
- **제한:** 분류 작업 중심 [arxiv](https://arxiv.org/pdf/2504.01857.pdf)

$$L_{KD} = \alpha L_{CE}(y, \hat{y}) + (1-\alpha) \text{KL}(p_{\text{teacher}}, p_{\text{student}})$$

#### **5.2.2 LLM 지식 증류 개선 (2023-2024)**

MiniLLM (Gu et al., 2023) [arxiv](https://arxiv.org/html/2406.11061v1)
- **특징:** RLHF와 유사한 선호도 기반 학습
- **성능:** 표준 KD 대비 우수한 일반화
- **제한:** 여전히 교사 모델의 상한선 제약 [arxiv](https://arxiv.org/html/2406.11061v1)

#### **5.2.3 이중 공간 지식 증류 (2024)**

Dual-Space Knowledge Distillation (Zhang et al., 2024) [alexlavaee](https://alexlavaee.me/blog/memorization-generalization-and-reasoning/)
- **혁신:** 교사/학생 모델의 출력 공간 통일
- **해결:** 다양한 어휘를 갖는 LLM 간 증류 문제 [alexlavaee](https://alexlavaee.me/blog/memorization-generalization-and-reasoning/)

**본 논문과의 관계:**
본 논문은 이러한 고급 KD 기법이 아닌 **단순 SFT를 사용하면서도 경쟁력 있는 성능 달성**함으로써, 복잡성과 효율성의 트레이드오프를 재조명 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

### 5.3 추론 시간 확장(Inference-Time Scaling)

#### **5.3.1 기본 개념 (2023-2024)**

추론 시간 확장의 핵심:
$$\text{Performance} = f(\text{Model Size}, \text{Training Compute}, \text{Inference Compute})$$

전통적으로 세 번째 변수는 무시되었으나, O1 등으로 중요성이 증대 [perplexity](https://www.perplexity.ai/sk/hub/blog/rl-training-for-math-reasoning)

#### **5.3.2 검색 알고리즘의 발전**

| 시기 | 방법 | 특징 | 참고 |
|------|------|------|------|
| 2023 | Tree-of-Thought (ToT) | 트리 탐색 기반 | 요청당 높은 비용 |
| 2024 | MCTS 기반 방법 | 효율적 탐색 | 여전히 비쌈 |
| 2024 | O1 방식 | RL로 최적화된 사고 | 내부 프로세스 비공개 |
| 2025 | AB-MCTS | 다중 모델 협업 | 신흥 방법 |

**혁신 지점:** 본 논문은 **검색 기반이 아닌 SFT로도 추론 성능 향상 가능**함을 보여줌 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

### 5.4 강화학습을 통한 수학 추론

#### **5.4.1 RLVR (2024-2025)**

강화학습 검증 가능(Reinforcement Learning with Verifiable Reward)
- **방식:** 정답/오답에 대한 이진 신호로 학습
- **특징:** 인간 주석 없이 자가 개선
- **성능:** 기본 모델 대비 2배 이상 향상

예시 (Qwen2.5-Math-1.5B):
$$\text{MATH500 정확도} : 36.0\% \to 73.6\% \text{ (1-shot RLVR)}$$

#### **5.4.2 교육 커리큘럼 중요성**

최신 연구(2025)의 중요 발견:
- 다양한 난이도 혼합이 RL 효율성 향상
- 점진적 응답 길이 증가 커리큘럼 효과
- 수학 전용 → 코드 전용 순차 학습의 이점

**본 논문과의 차이:**
- O1 논문: RL 중심의 추론 발달
- 본 논문: SFT와 증류 중심의 성능 달성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

### 5.5 일반화 및 전이 학습

#### **5.5.1 도메인 간 전이의 어려움**

최신 발견(2025):
$$\text{Result} : \text{수학 추론 성공} \not\Rightarrow \text{다른 영역 성공}$$

연구 결과: 많은 모델이 수학에서는 성공하지만 **다른 영역으로의 전이 실패**

**본 논문의 발견과의 대조:**
본 논문은 수학 데이터만으로도 안전성, 개방형 QA 등에서 **일반화 성공**을 보임으로써, 기존 통념에 대항 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

#### **5.5.2 약-강 일반화(Weak-to-Strong Generalization)**

약한 모델 피드백으로 강한 모델 개선 가능성
- **핵심 통찰:** 강한 모델은 약한 모델의 피드백 속에 숨겨진 잠재 지식 활용 가능
- **적용:** 본 논문의 SFT는 이 원리의 사례

#### **5.5.3 "쓸쓸한 교훈"과의 연결**

Rich Sutton의 "쓸쓸한 교훈(Bitter Lesson)"과의 연관 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf):
- 원본: 기초 이론보다 스케일과 컴퓨팅이 중요
- 본 논문의 확장: 증류로 쉬운 성능 향상이 가능하지만, 장기적으로는 **기초 기술 개발이 필수**

***

## 6. 기술 투명성 지수(TTI) 프레임워크

### 6.1 평가 체계

**총 100점 만점:**
- **데이터(14점):** 데이터셋 출처, 선택 기준, 합성 데이터 프로세스
- **방법론(33점):** 모델 상세, 검색/RL 알고리즘, 효과성 검증
- **평가(24점):** 벤치마크 선택, 평가 지표, 조건 설명
- **오픈소스(29점):** 데이터, 모델 가중치, 코드, 문서 공개 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

### 6.2 O1 재현 연구 비교

| 연구 | 데이터 | 방법론 | 평가 | 오픈소스 | **총점** |
|------|--------|--------|------|----------|---------|
| Open O1 | 0 | 8 | 20 | 5 | **33** |
| O1-Journey (Part 1) | 10 | 33 | 24 | 9 | **76** |
| LLaMA-O1 | 0 | 6 | 0 | 5 | **11** |
| K0Math | 0 | 0 | 16 | 0 | **16** |
| Skywork O1 | 0 | 0 | 0 | 0 | **0** |
| DeepSeek-R1-Lite | 0 | 0 | 20 | 0 | **20** |
| **O1-Journey (Part 2)** | **10** | **33** | **24** | **12** | **79** |

**해석:** 본 논문(Part 2)은 가장 높은 투명성 달성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

***

## 7. 앞으로의 연구에 미치는 영향

### 7.1 기술적 영향

#### **7.1.1 새로운 연구 패러다임의 필요성**

**문제 제기:**
- 증류의 성능 천장을 극복하기 위해서는 **원본 O1과 같은 새로운 기초 기술 개발** 필수
- 단순 스케일링이나 증류만으로는 근본적 한계 도달 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

**필요한 연구 방향:**
1. **검색 알고리즘 최적화:** MCTS, 다중 경로 탐색의 개선
2. **RL 프레임워크 고도화:** 순수 강화학습으로 추론 능력 습득
3. **새로운 아키텍처:** 추론을 명시적으로 지원하는 모델 구조 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

#### **7.1.2 영역별 특화 vs 범용성**

**발견:**
- 수학 데이터만으로도 다양한 영역에 일반화
- 하지만 팩트기반 작업에는 제한적 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

**연구 기회:**
$$\text{Cross-Domain Transfer} = f(\text{Task Type}, \text{Training Data}, \text{Architecture})$$

서로 다른 작업 특성에 맞는 최적 데이터 혼합 전략 개발 필요

### 7.2 방법론적 영향

#### **7.2.1 투명성 문화의 확산**

**중요성:**
- TTI 프레임워크로 표준화된 평가 체계 제공
- 재현성 있는 연구 문화 조성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

**실행 방안:**
- 벤치마킹 커뮤니티 표준 제시
- 주요 학술 회의에서 투명성 점수 공시
- 오픈소스 모델 증가 장려

#### **7.2.2 증류 기술의 정교화**

**기존 한계 극복:**
- 증류 데이터 품질 개선 기법
- 다양한 증류 전략의 체계적 비교
- 증류 효율성 최대화 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

**구체적 연구 주제:**
- 동적 데이터 선택(Dynamic Data Selection)
- 적응형 미세조정(Adaptive Fine-Tuning)
- 앙상블 증류 방법 [auai](https://www.auai.org/uai2015/proceedings/papers/123.pdf)

### 7.3 교육과 인력 양성

#### **7.3.1 "쓸쓸한 교훈"의 교육적 의미**

**핵심 경고:**
> "지능형 AI 구축도 중요하지만, 첫 원리 사고 능력을 갖춘 연구자 육성이 더욱 근본적인 인류의 사명이다." [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

**실제 위험:**
- 쉬운 성과에 집중하는 경향으로 인한 기초 기술 역량 저하
- 차세대 AI 연구자들이 프롬프트 엔지니어링에만 집중
- 진정한 알고리즘 이해와 창의적 문제 해결 능력 부족 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

#### **7.3.2 균형 잡힌 연구 포트폴리오**

**권장사항:**
1. **단기 성과 & 장기 혁신의 균형**
   - 증류로 빠른 성과 달성 (30% 노력)
   - 기초 연구에 70% 집중 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

2. **계산 인프라 투자의 지속성**
   - 증류 쉬움에도 고급 컴퓨팅 환경 유지
   - 검색, RL 등 기초 기술 개발 인프라 구축 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

3. **학생 교육 개선**
   - 신경망 아키텍처 설계 능력
   - 검색 알고리즘과 최적화 이론
   - 실험 설계와 과학적 사고 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

***

## 8. 연구 시 고려할 점

### 8.1 기술적 고려사항

#### **8.1.1 증류 의존성 최소화**

**문제:** 증류로 빠른 성과 달성 후 혁신 정체

**해결책:**
$$\text{Innovation Score} = \alpha \cdot \text{Performance} + (1-\alpha) \cdot \text{Novelty}$$

여기서 $\alpha < 0.7$로 설정하여 새로운 기술 개발 우대 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

#### **8.1.2 성능-기술력 추적 체계**

| 메트릭 | 의미 | 권장 기준 |
|--------|------|---------|
| TTI 점수 | 투명성과 재현성 | 70점 이상 |
| 원본 성과율 | O1 능력 비율 | 종료 기준 명시 |
| 새로운 기술 기여 | 순수 혁신 | 최소 50% |

#### **8.1.3 안전성 정렬의 이중성**

**발견:** 수학 추론 훈련이 특정 안전 벤치마크는 개선하지만 다른 것은 악화 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

**권장사항:**
- 증류 모델 사용 시 안전성 재평가 필수
- 명시적 안전 정렬 단계 추가 고려
- 다양한 안전 벤치마크 사용 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

### 8.2 연구 윤리 및 투명성

#### **8.2.1 방법 공시의 중요성**

**현재 문제:**
- 많은 기관이 O1 재현 주장만 하고 실제 방법 비공개
- "검은 상자(Black Box) 혁신" 경향 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

**개선 방안:**
1. 논문 작성 시 모든 기술 상세 공개
2. 코드와 데이터의 재현성 있는 공개
3. TTI 점수 자체 평가 및 공개 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

#### **8.2.2 과장된 주장 방지**

**체크리스트:**
- [ ] 성능 주장이 증류 대비 얼마의 추가 혁신인지 명시?
- [ ] 벤치마크가 학습 분포(In-Distribution)인지 확인?
- [ ] 오픈소스 데이터로 재현 가능한가?
- [ ] TTI 자체 평가 점수 제공? [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

### 8.3 실무 적용 시 주의점

#### **8.3.1 프로덕션 배포 전 검증**

**증류 모델의 문제점:**
- O1-preview 수준 성능은 달성했지만, O1-mini에는 미치지 못함
- 프로덕션에서 기대 성능 미달 가능성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

**권장 검증:**
1. 오픈소스 대안과의 비교 평가
2. 도메인별 성능 편차 분석
3. 비용-성능 트레이드오프 분석

#### **8.3.2 장기적 기술 투자 균형**

**단기 vs 장기 전략:**

| 단계 | 전략 | 기간 | 기대 효과 |
|------|------|------|---------|
| 즉시 | 증류로 빠른 성과 | 3개월 | 경쟁 벤치마크 달성 |
| 중기 | 기초 기술 개발 | 1-2년 | 원본 수준 기술 확보 |
| 장기 | 새로운 패러다임 | 3-5년 | 독자 혁신 실현 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf) |

***

## 9. 결론

O1 Replication Journey Part 2는 **단순한 지식 증류 기법의 강력함과 깊은 한계를 동시에 드러내는 이중적 기여**를 합니다.

### 9.1 핵심 발견의 재정리

1. **기술적 성과:** 수만 개의 증류 샘플로 O1-preview 초과 가능
2. **놀라운 일반화:** 수학만 훈련해도 안전성, QA 등으로 전이 성공
3. **심각한 한계:** 성능 천장, 팩트 기반 작업 실패, 포괄적 안전성 부족
4. **메타 기여:** 투명성 프레임워크로 필드 표준 제시

### 9.2 미래 연구의 방향

$$\text{미래 = Distillation의 효율성} + \text{RL의 혁신성} + \text{새로운 아키텍처}$$ [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

**3대 개선 영역:**
1. **성능 천장 돌파:** 순수 RL과 검색 기술의 고도화
2. **일반화 강화:** 다양한 도메인을 포괄하는 통합 학습
3. **안전성 보강:** 명시적 안전 정렬 + 탄력적 추론 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

### 9.3 최종 경고와 제언

논문이 전달하는 핵심 메시지:

> **"효율적인 지식 증류는 강력한 도구이지만, 이 편의성에 빠져 기초 기술 혁신을 포기해서는 안 된다. 지능형 AI 개발만큼 첫 원리 사고를 하는 연구자 양성이 AI 미래를 결정한다."** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f487704-d5c0-4c85-82ee-8b2c405f71da/2411.16489v1.pdf)

이는 단순히 기술 문제가 아니라, **AI 연구 커뮤니티의 장기적 건강과 혁신 역량**에 관한 경고입니다.

***

## 참고 문헌 (선별)

<span style="display:none">[^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86][^1_87]</span>

<div align="center">⁂</div>

[^1_1]: 2411.16489v1.pdf

[^1_2]: https://leehanchung.github.io/blogs/2024/10/08/reasoning-understanding-o1/

[^1_3]: https://www.ve3.global/inference-time-scaling-the-next-frontier-in-ai-performance/

[^1_4]: https://arxiv.org/pdf/2504.01857.pdf

[^1_5]: https://arxiv.org/html/2406.11061v1

[^1_6]: https://alexlavaee.me/blog/memorization-generalization-and-reasoning/

[^1_7]: https://www.perplexity.ai/sk/hub/blog/rl-training-for-math-reasoning

[^1_8]: https://openai.com/index/learning-to-reason-with-llms/

[^1_9]: https://www.reddit.com/r/ArtificialInteligence/comments/1jwvhng/research_shows_that_reasoning_models_generalize/

[^1_10]: https://www.auai.org/uai2015/proceedings/papers/123.pdf

[^1_11]: https://arxiv.org/abs/2407.06023

[^1_12]: https://arxiv.org/abs/2410.13639

[^1_13]: https://arxiv.org/abs/2411.06198

[^1_14]: https://arxiv.org/abs/2412.18319

[^1_15]: https://arxiv.org/abs/2410.02884

[^1_16]: https://www.cureus.com/articles/301598-openai-o1-preview-vs-chatgpt-in-healthcare-a-new-frontier-in-medical-ai-reasoning

[^1_17]: https://arxiv.org/abs/2410.02162

[^1_18]: https://www.mdpi.com/2073-431X/13/11/278

[^1_19]: https://arxiv.org/abs/2412.16720

[^1_20]: https://arxiv.org/abs/2412.21187

[^1_21]: https://arxiv.org/abs/2409.18486

[^1_22]: https://arxiv.org/html/2412.16720

[^1_23]: http://arxiv.org/pdf/2409.19924.pdf

[^1_24]: http://arxiv.org/pdf/2502.10867.pdf

[^1_25]: https://arxiv.org/pdf/2502.06807.pdf

[^1_26]: https://arxiv.org/pdf/2501.05366.pdf

[^1_27]: https://arxiv.org/html/2503.22732v1

[^1_28]: http://arxiv.org/pdf/2411.06198.pdf

[^1_29]: http://arxiv.org/pdf/2410.01792.pdf

[^1_30]: https://en.wikipedia.org/wiki/O1_(generative_pre-trained_transformer)

[^1_31]: https://zilliz.com/learn/knowledge-distillation-from-large-language-models-deep-dive

[^1_32]: https://sakana.ai/ab-mcts/

[^1_33]: https://codefinity.com/blog/Introducing-OpenAI-o1-preview:-The-Future-of-AI-Reasoning

[^1_34]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12634706/

[^1_35]: https://arxiv.org/abs/2510.10787

[^1_36]: https://aclanthology.org/2024.emnlp-main.1010/

[^1_37]: https://rlhfbook.com/c/14-reasoning

[^1_38]: https://simonwillison.net/2024/Sep/12/openai-o1/

[^1_39]: https://aclanthology.org/2024.lrec-main.593.pdf

[^1_40]: https://encord.com/blog/openai-o1/

[^1_41]: https://arxiv.org/abs/2306.08543

[^1_42]: https://www.youtube.com/watch?v=QEDGOEJxQk4

[^1_43]: https://arxiv.org/html/2502.10867v1

[^1_44]: https://arxiv.org/pdf/2306.08543.pdf

[^1_45]: https://arxiv.org/html/2504.00294v1

[^1_46]: https://arxiv.org/abs/2504.11426

[^1_47]: https://arxiv.org/html/2412.16720v1

[^1_48]: https://arxiv.org/abs/2504.02495

[^1_49]: https://arxiv.org/abs/2402.13116

[^1_50]: https://arxiv.org/abs/2503.07154

[^1_51]: https://arxiv.org/abs/2409.19924

[^1_52]: https://arxiv.org/abs/2509.14526

[^1_53]: https://arxiv.org/abs/2501.18427

[^1_54]: https://arxiv.org/html/2306.08543v1

[^1_55]: https://link.springer.com/10.1007/s11604-024-01712-2

[^1_56]: https://www.jsr.org/hs/index.php/path/article/view/6812

[^1_57]: https://arxiv.org/abs/2407.02514

[^1_58]: https://arxiv.org/abs/2410.00151

[^1_59]: https://arxiv.org/abs/2305.17306

[^1_60]: https://www.semanticscholar.org/paper/9851db4cab76eed072355ec6d9d91ec187b3b13a

[^1_61]: https://aclanthology.org/2024.semeval-1.196

[^1_62]: https://arxiv.org/abs/2305.14215

[^1_63]: https://arxiv.org/abs/2305.13903

[^1_64]: https://aclanthology.org/2023.emnlp-main.936.pdf

[^1_65]: http://arxiv.org/pdf/2308.10379.pdf

[^1_66]: https://aclanthology.org/2023.findings-emnlp.811.pdf

[^1_67]: https://arxiv.org/html/2503.05179v1

[^1_68]: https://aclanthology.org/2023.findings-emnlp.179.pdf

[^1_69]: http://arxiv.org/pdf/2406.09136.pdf

[^1_70]: http://arxiv.org/pdf/2502.12134.pdf

[^1_71]: https://arxiv.org/abs/2406.09136

[^1_72]: https://huggingface.co/papers/2504.20571

[^1_73]: https://blog.iese.edu/artificial-intelligence-management/2024/chain-of-thought-reasoning-the-new-llm-breakthrough/

[^1_74]: https://www.perplexity.ai/hub/blog/rl-training-for-math-reasoning

[^1_75]: https://arxiv.org/abs/2402.10200

[^1_76]: https://proceedings.neurips.cc/paper_files/paper/2024/file/00d80722b756de0166523a87805dd00f-Paper-Conference.pdf

[^1_77]: https://www.youtube.com/watch?v=jyNFNECwjEg

[^1_78]: https://openreview.net/forum?id=ddOxvs4NAq

[^1_79]: https://arxiv.org/abs/2505.18116

[^1_80]: https://web3.arxiv.org/pdf/2511.22176

[^1_81]: https://arxiv.org/abs/2501.12948

[^1_82]: https://arxiv.org/pdf/2405.16236.pdf

[^1_83]: https://arxiv.org/html/2508.01191v2

[^1_84]: https://arxiv.org/abs/2505.16400

[^1_85]: https://arxiv.org/abs/2510.11184

[^1_86]: https://arxiv.org/pdf/2412.06769.pdf

[^1_87]: https://arxiv.org/abs/2201.11903
