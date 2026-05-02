
# s1: Simple Test-Time Scaling

> **논문 정보**
> - **제목:** s1: Simple Test-Time Scaling
> - **저자:** Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, Tatsunori Hashimoto (Stanford University 외)
> - **발표:** arXiv:2501.19393 (2025.01.31 제출), EMNLP 2025 게재
> - **오픈소스:** https://github.com/simplescaling/s1

---

## 1. 핵심 주장 및 주요 기여 요약

테스트 타임 스케일링(Test-time scaling)은 추론(inference) 시에 추가적인 연산(compute)을 활용하여 모델 성능을 향상시키는 유망한 새로운 언어 모델링 패러다임입니다.

OpenAI의 o1 모델이 이 능력을 보여줬지만 방법론을 공개하지 않아 많은 복제 시도가 이어졌으며, 본 논문은 테스트 타임 스케일링과 강력한 추론 성능을 달성하기 위한 **가장 단순한 접근법**을 탐구합니다.

### 🔑 핵심 기여 3가지

| # | 기여 | 내용 |
|---|---|---|
| 1 | **s1K 데이터셋 구축** | 1,000개의 고품질 질문-추론 트레이스 쌍 |
| 2 | **Budget Forcing 기법** | 테스트 타임 연산량을 제어하는 새로운 디코딩 기법 |
| 3 | **s1-32B 모델** | o1-preview를 능가하는 오픈소스 추론 모델 |

단 1,000개의 샘플로 next-token prediction을 학습하고, Budget Forcing이라는 간단한 테스트 타임 기법으로 사고 시간을 제어함으로써, 더 많은 테스트 타임 연산에 따라 성능이 향상되는 강력한 추론 모델을 구축할 수 있음을 보여줍니다.

---

## 2. 논문의 상세 분석

### 2-1. 해결하고자 하는 문제

지난 몇 년간 언어 모델의 성능 향상은 대규모 자기지도 사전학습(self-supervised pretraining)을 통한 **훈련 시 연산량(train-time compute) 스케일링**에 크게 의존해왔습니다.

이에 대한 새로운 패러다임으로 **테스트 타임 스케일링**이 등장했는데, 이는 테스트 시 연산량을 증가시켜 더 나은 결과를 얻는 방법으로, 모델이 실제 사용 중에 "두뇌"를 더 효과적으로 활용하도록 최적화하는 방식입니다.

DeepSeek R1은 수백만 개의 샘플과 여러 훈련 단계를 통한 강화학습으로 o1 수준의 성능을 복제했지만, 많은 o1 복제 시도들 중 **명확한 테스트 타임 스케일링 동작을 공개적으로 재현한 것은 없었습니다.**

---

### 2-2. 제안하는 방법

#### 📊 (1) s1K 데이터셋 구축

세 가지 기준(ablation으로 검증)에 따라 추론 트레이스와 짝지어진 1,000개 질문으로 구성된 소규모 데이터셋 **s1K**를 구축합니다: **난이도(Difficulty), 다양성(Diversity), 품질(Quality)**.

구체적으로 s1K는 **Gemini Thinking Experimental**에서 증류(distillation)한 추론 트레이스와 답변이 짝지어진 1,000개의 엄선된 질문으로 구성됩니다.

추론 트레이스가 길수록 해당 샘플이 더 복잡하다는 것을 나타내므로, **더 긴 추론 트레이스를 가진 샘플을 우선 선택합니다.**

훈련은 16개의 H100 GPU에서 단 **26분**의 supervised fine-tuning(SFT)만 필요합니다.

#### 🧮 (2) Budget Forcing (핵심 방법론)

훈련 후 Budget Forcing으로 모델이 사용하는 테스트 타임 연산량을 제어합니다: **(I) 모델이 원하는 한도보다 더 많은 thinking token을 생성하면, end-of-thinking 토큰 구분자를 강제로 추가하여 사고 프로세스를 종료시킵니다.** 이 방식은 모델이 답변 생성으로 전환하도록 만듭니다.

Budget Forcing의 두 가지 모드를 수식으로 표현하면:

**최대 토큰 제한 (Truncation):**

$$T_{\text{think}} > T_{\max} \Rightarrow \text{강제로 } \langle \text{end-of-think} \rangle \text{ 토큰 삽입}$$

**최소 토큰 강제 (Extension):**

$$T_{\text{think}} < T_{\min} \Rightarrow \text{thinking trace에 "Wait" 토큰 추가}$$

테스트 타임 연산을 제어하기 위해 모델의 사고 프로세스를 강제로 종료하거나, 모델이 종료하려 할 때 "Wait"를 여러 번 추가하여 사고를 연장합니다. 이는 모델이 답을 재확인하게 유도하여 **잘못된 추론 단계를 수정**하도록 만듭니다.

논문의 Table 4는 "Wait" 추가가 AIME24에서 53.3% 정확도를 달성한 반면, 중립적인 "Hmm"(50.0%)이나 아무것도 추가하지 않은 경우(50.0%)보다 높음을 확인합니다. Figure 3은 모델이 처음에 잘못된 답을 제시했다가 "Wait" 이후 재평가하여 수정하는 직접적인 증거를 보여줍니다.

#### 📐 Sequential vs. Parallel 스케일링 비교

테스트 타임 스케일링 방법은 1) **순차적(Sequential)** 방법(이후 연산이 이전 결과에 의존, 예: 긴 추론 트레이스)과 2) **병렬(Parallel)** 방법(독립적 연산, 예: majority voting)으로 분류됩니다. 순차적 스케일링에 집중하는 이유는, 이후 연산이 중간 결과를 기반으로 더 깊은 추론과 반복적 개선이 가능하기 때문입니다.

s1-32B에 Budget Forcing을 적용한 결과, 기반 모델에 majority voting을 통한 병렬 스케일링을 아무리 늘려도 s1-32B의 성능을 따라잡지 못하며, **순차적 스케일링이 병렬보다 더 효과적**임을 검증합니다.

---

### 2-3. 모델 구조

s1-32B의 전체 파이프라인:

```
[Qwen2.5-32B-Instruct] (사전학습 모델)
        ↓ SFT (26분, 16 × H100)
[s1K 데이터셋 1,000개 샘플]
        ↓
[s1-32B] (파인튜닝된 추론 모델)
        ↓ 테스트 타임
[Budget Forcing] → 성능 스케일링
```

왜 1,000개 샘플만으로 성능 향상이 가능한가? 모델은 이미 사전학습 중 수조 개의 토큰에 걸쳐 방대한 추론 데이터에 노출되어 있어, **추론 능력이 이미 모델에 내재**되어 있습니다. 소규모 파인튜닝 단계는 이를 활성화하고, Budget Forcing으로 테스트 타임에 더욱 확장합니다.

이는 LIMA(Zhou et al., 2023)에서 제안한 **"Superficial Alignment Hypothesis"**와 유사한데, 1,000개의 예시만으로도 모델을 정렬하기에 충분할 수 있다는 가설입니다.

---

### 2-4. 성능 향상

Qwen2.5-32B-Instruct를 s1K로 supervised finetuning하고 Budget Forcing을 적용한 결과, **s1-32B는 경쟁 수학 문제(MATH 및 AIME24)에서 o1-preview를 최대 27%까지 초과**합니다.

나아가 Budget Forcing으로 s1-32B를 스케일링하면 테스트 타임 개입 없이의 성능을 초과하여 외삽(extrapolate)할 수 있습니다: **AIME24에서 50%에서 57%로** 향상됩니다.

s1 발표 7일 후 s1.1도 공개했는데, s1K의 1,000개 샘플에 대한 추론 트레이스를 Gemini 대신 **DeepSeek R1으로 재생성**하여 s1K-1.1을 만들고, 동일한 훈련 절차로 s1.1을 훈련했습니다.

s1.1은 s1보다 **유의미하게 더 나은 성능**을 보입니다.

---

### 2-5. 한계점

이 접근법은 스케일링이 **플래토(plateau)에 도달하면 한계**가 있으며, 언어 모델의 **컨텍스트 윈도우 크기에 제약**을 받습니다.

Budget Forcing 기법은 AIME24에서 6회 적용 시 성능이 결국 평탄해집니다. 또한 end-of-thinking 토큰 구분자를 너무 자주 억제하면 계속된 추론 대신 **반복 루프(repetitive loops)**에 빠질 수 있습니다.

성능이 일정 길이를 초과하면 오히려 감소할 수 있다는 점이 지적되었으며, s1의 접근법은 주로 수학 태스크에 한정되었고, **전문 지식이 필요한 도메인별 태스크**에 대한 검증은 충분히 이루어지지 않았습니다.

테스트 타임 스케일링 동작의 **기저 메커니즘이 불명확**하며, 단순 테스트 타임 스케일링은 세 가지 핵심 요소에 의존합니다: (1) o1-유사 모델에서 증류된 긴 CoT 데이터로 파인튜닝, (2) 최대 길이를 강제하여 스케일 다운, (3) 종료 시도 시 "Wait"를 반복 추가하여 스케일 업.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. Superficial Alignment Hypothesis와 일반화

s1의 소규모 데이터셋 사용 접근법은 Meta의 "Less is More for Alignment" 논문에서 소개된 **Superficial Alignment Hypothesis**를 반영합니다. 이 가설은 모델의 지식과 능력이 사전학습 중에 대부분 습득되므로, 추론 능력이 이미 사전학습 모델에 존재하며, s1K와 같은 소규모 고품질 데이터셋으로 이를 충분히 활성화하고 개선할 수 있다고 주장합니다.

### 3-2. Sequential + Parallel 결합을 통한 일반화 확장

Budget Forcing의 한계를 극복하기 위해, Budget Forcing을 **병렬 스케일링 기법과 결합**하여 더욱 확장할 수 있습니다.

### 3-3. 다른 도메인으로의 적용 가능성

후속 분석 연구에 따르면, 모델들은 증류(distillation)를 받았는지 여부와 관계없이 **테스트 타임 스케일링 동작을 나타낼 수 있음**이 확인되었습니다.

그러나 수학 외 도메인에 대한 일반화는 제한적입니다. 법률 분야에서의 연구를 보면:
DeepSeek-R1과 OpenAI의 o1으로 대표되는 최근의 테스트 타임 스케일링 발전은 추론 중 사고 사슬을 연장함으로써 일반적 추론 성능을 크게 향상시킬 수 있음을 보여주지만, **법률 추론과 같은 특수 도메인에 미치는 영향은 충분히 탐구되지 않았습니다.**

### 3-4. 데이터 품질의 일반화에 대한 영향

ablation 실험은 데이터 선택 과정의 중요성을 드러냅니다: 무작위 선택이나 단일 기준만 사용하면 성능이 크게 저하되는 반면, 더 많은 데이터셋을 사용해도 추가적인 이점은 미미합니다. 즉, **데이터의 양보다 질(난이도+다양성+품질)이 일반화에 핵심적**입니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려 사항

### 4-1. 연구 영향

#### ① 소규모 데이터 + 테스트 타임 스케일링 패러다임의 정립
이 연구는 효과적인 추론 모델이 방대한 데이터셋이나 복잡한 아키텍처 없이도 **최소한의 리소스로 개발될 수 있음**을 보여주어, 방대한 데이터와 훈련 시간의 필요성에 도전장을 내밉니다.

#### ② 오픈소스 o1 수준 추론의 재현
Budget Forcing을 결합한 이 접근법은 OpenAI의 테스트 타임 스케일링 곡선을 **최초로 재현**하였습니다. 왜 1,000개의 샘플만으로 이러한 성능 향상이 가능한지에 대한 중요한 질문을 제기합니다.

#### ③ 후속 연구 촉발
s1 발표 7일 만에 s1.1이 공개되었는데, 동일한 s1K 질문에 Gemini 대신 **DeepSeek R1의 추론 트레이스를 재활용**하여 훨씬 나은 성능을 달성함으로써, 교사 모델(teacher model) 선택의 중요성이라는 새로운 연구 방향을 열었습니다.

#### ④ 비판적 재검토 연구 촉발
후속 분석 연구("It's Not That Simple")에서는 **최대 길이 강제(scaling down)**가 테스트 타임 스케일링 동작의 주된, 혹은 유일한 기여 요인일 수 있음을 지적하였습니다.

R1과 QwQ에서 솔루션 길이를 연장한다고 반드시 성능이 향상되지 않으며, 이는 모델의 제한된 자기 수정(self-revision) 능력과 관련됩니다. 추가 연구에서는 이 현상이 "모델 과사고(underthinking)"와 관련된다고 분석하며, **모델이 처음에 올바른 중간 답에 도달했다가 이후 확장된 추론 과정에서 잘못된 결론으로 이탈한다**고 설명합니다.

---

### 4-2. 앞으로 연구 시 고려할 점

| 분야 | 고려 사항 |
|------|----------|
| **도메인 일반화** | 수학 외 법률, 의료, 과학 등 다양한 도메인에 대한 검증 필요 |
| **교사 모델 선택** | 증류 소스(Gemini vs. R1 vs. Claude 등)가 성능에 미치는 영향 심층 연구 |
| **Budget Forcing 메커니즘** | "Wait" 토큰의 실제 작동 메커니즘(시간 연장 vs. 의미론적 개입) 규명 |
| **스케일링 한계 극복** | 순차적+병렬 스케일링 결합으로 플래토 문제 해결 방안 |
| **효율성 최적화** | 긴 CoT 추론이 증가시키는 추론 비용과 지연 시간 개선 |
| **자기 수정 능력** | 모델이 "Wait" 후 진정한 재추론을 하는지, 단순 반복에 빠지는지 연구 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

지난 몇 년간 언어 모델 성능 향상은 **훈련 시 연산량 스케일링**에 의존해왔으며(Kaplan et al., 2020 — Scaling Laws for Neural Language Models), 이는 GPT-3, LLaMA 등 대규모 모델의 발전을 이끌었습니다.

| 연구 | 방법론 | 데이터 규모 | 특징 | s1 대비 |
|------|--------|-----------|------|---------|
| **Chain-of-Thought (Wei et al., 2022)** | 프롬프팅 기반 추론 유도 | 없음 (zero/few-shot) | 단계별 추론 명시 | s1은 SFT+테스트타임 개입으로 더 강력 |
| **STaR (Zelikman et al., 2022)** | 부트스트랩 추론 | 자가 생성 | 자기 생성 추론으로 파인튜닝 | s1은 외부 증류, 더 단순 |
| **LIMA (Zhou et al., 2023)** | SFT, 1,000개 샘플 | 1,000개 | 소규모 정렬 가설 검증 | s1의 이론적 근거, 추론으로 확장 |
| **OpenAI o1 (2024)** | RL + 긴 CoT | 수백만 (비공개) | 강력한 추론, 비공개 방법론 | s1이 공개 방법으로 복제 시도 |
| **DeepSeek-R1 (2025)** | 순수 RL + 다단계 학습 | 수백만 | o1 수준 성능 달성 | 더 많은 데이터, 복잡한 파이프라인 |
| **LIMO (Ye et al., 2025)** | SFT, 소규모 | 817개 | "Less is More for Reasoning" | s1과 유사하나 더 적은 데이터 |
| **s1 (Muennighoff et al., 2025)** | SFT + Budget Forcing | 1,000개 | 최초 공개 테스트 타임 스케일링 재현 | **본 논문** |

여러 연구들(Kimi Team, DeepSeek-AI)이 o1 모델의 성능을 복제하기 위해 CoT 추론 과정 길이를 늘려 테스트 타임 연산을 확장하는 방식으로 LLM을 훈련하려 했으며, DeepSeek-R1은 순수 강화학습 과정을 통해 LLM이 자연스럽게 테스트 타임 연산을 늘리고 일관된 성능 향상을 보일 수 있음을 보여주었습니다.

그러나 이러한 o1 유사 모델들이 진정한 테스트 타임 스케일링 능력을 보유하는지는 여전히 탐구 중입니다. 연구에 따르면 **긴 CoT가 반드시 더 높은 정확도를 보장하지 않으며**, 오히려 올바른 솔루션이 잘못된 솔루션보다 짧은 경우도 많습니다. 이 현상은 더 긴 CoT에서 더 많은 자기 수정이 포함되어 성능 저하를 초래할 수 있는 것과 밀접하게 관련됩니다.

---

## 📚 참고 자료 출처

1. **arXiv 논문 원문:** Muennighoff et al. (2025), "s1: Simple test-time scaling", arXiv:2501.19393. https://arxiv.org/abs/2501.19393
2. **EMNLP 2025 게재본:** ACL Anthology, EMNLP 2025 Main, pp. 20275–20321. https://aclanthology.org/2025.emnlp-main.1025/
3. **공식 프로젝트 페이지:** https://simplescaling.github.io/
4. **GitHub 저장소:** https://github.com/simplescaling/s1
5. **ar5iv (HTML 논문):** https://ar5iv.labs.arxiv.org/html/2501.19393
6. **AI Papers Academy 해설:** "s1: Simple Test-Time Scaling – Can 1k Samples Rival o1-Preview?" https://aipapersacademy.com/s1/
7. **Hugging Face Paper Page:** https://huggingface.co/papers/2501.19393
8. **Medium 리뷰:** Sulbha Jain, "s1: Simple test-time scaling — Paper Review" https://medium.com/@sulbha.jindal/s1-simple-test-time-scaling-paper-review-79a5e7bf9677
9. **비판적 후속 연구:** "It's Not That Simple. An Analysis of Simple Test-Time Scaling", arXiv:2507.14419. https://arxiv.org/html/2507.14419
10. **관련 비교 연구:** "Revisiting the Test-Time Scaling of o1-like Models", arXiv:2502.12215. https://arxiv.org/abs/2502.12215
11. **법률 도메인 평가 연구:** "Evaluating Test-Time Scaling LLMs for Legal Reasoning: OpenAI o1, DeepSeek-R1, and Beyond", arXiv:2503.16040 / EMNLP Findings 2025. https://aclanthology.org/2025.findings-emnlp.742/
12. **인퍼런스 타임 스케일링 최신 동향:** Introl Blog, "Inference-Time Scaling Research", December 2025. https://introl.com/blog/inference-time-scaling-research-reasoning-models-december-2025

# s1: Simple Test-Time Scaling
## 1. 핵심 주장 및 주요 기여
**s1** 논문의 핵심 주장은 **테스트 타임 스케일링(test-time scaling)을 구현하기 위해 복잡한 강화학습(RL)이나 대규모 데이터가 필수가 아니라는 것**입니다. OpenAI의 o1 모델이 테스트 타임 스케일링의 강력함을 증명했지만 방법론을 공개하지 않자, 여러 연구팀들이 거대한 데이터셋(DeepSeek R1: 800K 샘플)과 RL 기반 학습으로 재현하려고 노력했습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

본 논문의 **세 가지 핵심 기여**는:

1. **s1K 데이터셋**: 어려움(difficulty), 다양성(diversity), 품질(quality) 세 가지 기준으로 엄격히 선별된 1,000개 질문-추론 쌍 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
2. **버짓 포싱(Budget Forcing)**: 테스트 타임에 모델의 생각 과정을 제어하는 간단한 메커니즘 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
3. **s1-32B 모델**: 단지 1,000개 샘플로 SFT 학습하여 o1-preview를 능가하는 성능 달성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

논문이 보여준 **핵심 성과**:
- **AIME24에서 o1-preview 대비 27% 향상**: 50% → 57% 성능 달성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- **가장 샘플 효율적인 오픈소스 모델**: 1K 샘플로 17K-800K 샘플 필요한 다른 모델들과 경쟁 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- **완전 오픈소스**: 모델 가중치, 데이터, 코드 모두 공개 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

***

## 2. 해결하고자 하는 문제
### 2.1 근본적인 문제
OpenAI o1은 테스트 타임 스케일링의 강력함을 입증했지만, 그 방법론은 비공개였습니다. 이로 인해 복제 시도들이 다음과 같은 비효율을 야기했습니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

| 문제점 | 기존 접근 | s1의 해결책 |
|--------|-----------|-------------|
| 데이터 규모 | 800K (DeepSeek R1) | **1K (1/800 감소)** |
| 학습 방식 | 복잡한 다단계 RL | **간단한 SFT** |
| 계산 자원 | 매우 높음 | **26분 × 16 H100** |
| 재현 가능성 | 매우 낮음 | **완전히 공개됨** |

### 2.2 핵심 질문
논문의 중심 질문: **"테스트 타임 스케일링과 강한 추론 성능을 달성하기 위한 가장 간단한 접근법은 무엇인가?"**

***

## 3. 제안하는 방법 (수식 포함)
### 3.1 데이터 큐레이션 (s1K 구성)
#### 3.1.1 초기 풀 구성 (59K → 1K)
원본 59,029개 질문을 세 단계 필터링으로 축소합니다:

**Step 1: 품질 필터링**
$$\text{Quality Filtered} = \{\text{samples without API errors, formatting issues}\}$$

선별 기준:
- API 오류 제거: 59,029 → 54,116
- 포맷 문제 제거 (ASCII art, 이미지 참조, 비정상 번호): 54,116 → 51,581
- 고품질 데이터셋 선별: 384 샘플 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

**Step 2: 어려움 필터링**

모델 기반 필터링을 사용하여 너무 쉬운 샘플 제거:

$$D_{\text{filtered}} = \{q \in D : \neg(\text{Qwen2.5-7B} \land \text{Qwen2.5-32B predict } q \text{ correctly})\}$$

여기서:
- $q$ = 질문
- $\text{difficulty indicator} = \text{token length}(\text{reasoning trace})$
- 가정: 어려운 문제일수록 더 많은 생각 토큰 필요 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

결과: 24,496 샘플

**Step 3: 다양성 필터링**

$$\text{s1K} = \text{DiversitySampling}(D_{\text{difficulty}}, K=1000)$$

알고리즘:
1. 모든 도메인 식별 (수학주제분류 체계): 50개 도메인 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
2. 균등 분포에서 도메인 선택: $P(\text{domain}) = \frac{1}{|\text{domains}|}$
3. 토큰 길이에 따른 가중치 샘플링:
   $$w_i = 2^{-\text{rank}_{\text{token length}}(i)}$$
4. 반복: 1,000개 샘플 도달까지

결과: **1,000개 샘플, 50개 도메인 균형** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

#### 3.1.2 데이터 특성
- **정확성**: 53.6% 정답 (추론 과정 학습에 초점, 절대 정답이 아님) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- **출처**: Google Gemini Thinking Experimental에서 생성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- **오염제거**: 8-gram 중복 제거로 평가 벤치마크 오염 방지 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

### 3.2 버짓 포싱 (Budget Forcing) 메커니즘
**정의**: 생각 토큰 수를 제어하는 테스트 타임 디코딩 개입 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

#### 3.2.1 최대값 제약 (조기 종료)

$$\text{if } T_{\text{current}} \geq T_{\max} \text{ then}$$

$$\text{Append } \langle\text{end thinking}\rangle \text{ token}$$

$$\text{Transition to answer generation phase}$$

**작동 방식**:
- 모델이 최대 생각 토큰에 도달하면 강제 종료 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- 답 생성 단계로 전환 (선택적으로 "Final Answer" 추가) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- 미완성 추론이라도 현재 최선의 답변 제공 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

#### 3.2.2 최소값 제약 (계속 격려)

$$\text{if } \text{model attempts to end AND } T_{\text{current}} < T_{\min} \text{ then}$$

$$\text{Suppress } \langle\text{end thinking}\rangle \text{ token}$$

$$\text{Append "Wait" to current reasoning trace}$$

**작동 방식**:
- 종료 토큰 생성 억제 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- 현재 추론 뒤에 "Wait" 문자열 추가 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- 모델이 답변을 재검토하도록 격려 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- 종종 잘못된 추론 단계 수정 (예: "raspberry"의 'r' 개수) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

#### 3.2.3 제어 가능성 지표

```math
\text{Control} = \frac{\text{\# instances with } T_{\min} \leq T_{\text{actual}} \leq T_{\max}}{\text{\# total instances}} \times 100\%
```

다양한 방법의 제어 가능성:
- Budget Forcing: **100%** (완벽한 제어) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- Token-Conditional: 40% (모델이 토큰 계산 못함) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- Step-Conditional: 60% (모델이 스텝 경계 지킴) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- Rejection Sampling: 100% (사후 필터링) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

### 3.3 모델 학습 구성
#### 3.3.1 기본 모델 및 설정
- **기본 모델**: Qwen2.5-32B-Instruct (사전 학습 + 지시 튜닝 완료) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- **학습 방식**: 감독 미세조정(SFT) 전용 (RL 없음) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- **특수 토큰**: 
  $$\text{Input} = \text{[Question]} + \langle\text{imstart|think}\rangle + \text{[Reasoning]} + \langle\text{imstart|answer}\rangle + \text{[Answer]}$$

#### 3.3.2 학습 하이퍼파라미터

| 항목 | 값 |
|------|-----|
| 에포크 | 5 |
| 배치 크기 | 16 |
| 총 그래디언트 스텝 | 315 |
| 정밀도 | bfloat16 |
| 학습률 | $1 \times 10^{-5}$ |
| 워밍업 | 선형 (5/16 스텝) |
| 스케줄 | 코사인 감쇠 |
| 옵티마이저 | AdamW |
| $\beta_1$ (모멘텀) | 0.9 |
| $\beta_2$ | 0.95 |
| 가중치 감쇠 | $10^{-4}$ |

#### 3.3.3 학습 효율성
$$\text{Training Time} = 26 \text{ minutes on 16 NVIDIA H100 GPUs}$$
$$\text{Total GPU Hours} = 7 \text{ H100 hours (vs. 394 for 59K training)}$$

**손실 함수**: 질문이 아닌 추론 추적과 답변에만 손실 계산 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

***

## 4. 모델 구조 및 아키텍처
### 4.1 기본 구조
$$\text{s1-32B} = \text{SFT}(\text{Qwen2.5-32B-Instruct}, \text{s1K})$$

**구성 요소**:
- **인코더-디코더**: Qwen2.5 트랜스포머 기반 (32B 파라미터) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- **생각 단계**: 명시적 구분 토큰 ($\langle\text{imstart|think}\rangle$) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- **답변 단계**: 생각 후 답변 생성 ($\langle\text{imstart|answer}\rangle$) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- **테스트 타임**: 버짓 포싱으로 단계별 토큰 수 제어 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

### 4.2 순차 vs 병렬 스케일링
#### 4.2.1 순차 스케일링 (Sequential Scaling)
$$\text{Reasoning}_{\text{sequential}} = f(f(f(...f(\text{Question})...)))$$

특징:
- 이전 추론 결과를 다음 추론에 입력 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- 더 깊은 추론 및 반복적 개선 가능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- **s1의 접근**: 버짓 포싱으로 제어 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

**성능**: AIME24에서 명확한 양의 기울기 달성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

#### 4.2.2 병렬 스케일링 (Parallel Scaling)
$$\text{Answer}_{\text{parallel}} = \text{MajorityVote}(f(\text{Q}), f(\text{Q}), ..., f(\text{Q}))$$

특징:
- 독립적으로 여러 추론 경로 샘플 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- 다수결 투표로 최종 답 결정 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- Qwen2.5-32B-Instruct (베이스)에서 약한 스케일링 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

**결론**: 순차 > 병렬 스케일링 (이전 결과에 기반할 수 있으므로) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

### 4.3 시퀀스 길이 영향
$$\text{Performance} = f(\text{Sequence Length})$$

| Sequence Length | Training | Evaluation |
|-----------------|----------|------------|
| 4,096 | 30% AIME24 | Long thinking (~20K tokens) |
| 32,768 | 50% AIME24 | Shorter thinking (~7K tokens) |

**발견**: 더 긴 시퀀스 길이 → 더 짧은 테스트 타임 생각 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- 이유: 학습 중 답변 섹션이 완전히 포함되어 답변 우도 증가 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

***

## 5. 성능 향상 및 일반화 능력
### 5.1 벤치마크 성능 비교
### 5.1.1 절대 성능
| 모델 | AIME24 | MATH500 | GPQA Diamond |
|------|---------|---------|--------------|
| **s1-32B** | **50.0%** | **92.6%** | **56.6%** |
| **s1-32B (BF)** | **56.7%** | **93.0%** | **59.6%** |
| o1-preview | 44.6% | 85.5% | 73.3% |
| o1-mini | 70.0% | 90.0% | 60.0% |
| QwQ-32B | 50.0% | 90.6% | 54.5% |
| r1-distill | 72.6% | 94.3% | 62.1% |
| r1 | 79.8% | 97.3% | 71.5% |

**주요 발견**:
- **s1-32B (BF)는 o1-preview를 AIME24에서 27% 능가**: 56.7% vs 44.6% [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- MATH500에서 강한 성능 (93.0%) 달성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- GPQA는 상대적으로 약함 (59.6%) - 과학 도메인 특화 필요 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

### 5.2 테스트 타임 스케일링 곡선
**스케일링 동작**:
- AIME24: 가장 강한 선형 증가 (512 → 8192 토큰에서 50% → 56.7%) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- MATH500: 한계 수익 체감 (빠르게 포화, 93% 근처) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- GPQA Diamond: 완만한 증가 (56.0% → 59.6%, 천천히 포화) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

**수학 공식**:

$$\text{Scaling Curve} = \text{Piecewise Linear}(\text{thinking tokens})$$

$$\text{Slope}_{\text{AIME24}} = \frac{\Delta \text{Accuracy}}{\Delta \text{Thinking Tokens}} > 0$$

### 5.3 샘플 효율성 혁신
**혁신적 발견**:
$$\text{s1-32B: 1K samples} \rightarrow \text{50.0% AIME24}$$
$$\text{r1: 800K samples} \rightarrow \text{79.8% AIME24}$$

**샘플 효율성 비율**:
- Sky-T1: 17K 샘플 (17배), 43.3% 성능 (낮음) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- Bespoke-32B: 17K 샘플 (17배), 63.3% 성능 (더 높음) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- r1-distill: 800K 샘플 (800배), 72.6% 성능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

**결론**: 샘플 수가 아니라 **큐레이션 품질이 핵심** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

### 5.4 일반화 성능 향상
#### 5.4.1 도메인 간 일반화

**학습 도메인**: 50개 (수학, 물리, 화학, 생물, 컴퓨터과학 등)

**평가 도메인**: AIME24, MATH500, GPQA Diamond (학습 데이터와 분리)

$$\text{Generalization Index} = \frac{\text{Eval Performance}}{\text{Training Data Difficulty}}$$

**발견**: 학습 도메인과 무관하게 일반화 가능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

#### 5.4.2 외삽(Extrapolation) 가능성

모델은 학습 분포를 **넘어서** 확장 가능:

$$\text{AIME24: } 50\% \xrightarrow{6\times \text{ budget forcing}} 57\%$$

**의미**: 
- 학습 중 최대 생각 토큰: ~6K
- 테스트 시 가능: 2배 이상 확장 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- 추론 능력이 활성화되고 "고갈"되지 않음 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

#### 5.4.3 얕은 정렬 가설 (Superficial Alignment Hypothesis)

논문의 핵심 통찰:

> **"사전 학습된 모델은 이미 대규모 추론 능력을 가지고 있다. SFT는 이를 가르치는 것이 아니라 활성화하는 것일 뿐이다."** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

증거:
- Qwen2.5-32B-Instruct 베이스: 26.7% AIME24 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- 1K 샘플 추가 학습: 50% AIME24 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- 증가: 23.3 포인트 (1.87배 개선) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- 비교: LIMA 논문 (1K 지시 데이터로 정렬 충분) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

***

## 6. 한계 및 제한 사항
### 6.1 테스트 타임 스케일링 한계
#### 6.1.1 포화 효과 (Saturation)

$$\text{Performance} = a + b \log(\text{thinking tokens}) + \epsilon$$

관찰된 포화:
- AIME24: 6배 버짓 포싱 후 포화 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- MATH500: 빠른 포화 (93% 근처) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- 이유: 컨텍스트 윈도우 한계 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

#### 6.1.2 반복 루프 문제

"Wait" 반복 추가 시:
- 모델이 같은 추론 반복 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- 새로운 통찰 없음 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- 오류 수정 실패 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

**해결 제안**:
- 다양한 문자열 순환 (Wait, Hmm, Let's reconsider, etc.) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- 주파수 페널티 적용 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- 더 높은 온도 설정 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

### 6.2 거절 샘플링 역 스케일링 (Inverse Scaling)
**관찰**: 길이 제약 하에서 샘플링하면 오류 증가 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

$$\text{Performance}_{\text{rejection}} = f(\text{length}) \text{ where } f \text{ is decreasing}$$

**발견**: 
- 3,500 토큰: 정답률 높음 (정답 경로에 빠르게 도달) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- 8,000 토큰: 정답률 낮음 (역추적/의심 포함) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- **상관성**: 짧은 생성 → 정답, 긴 생성 → 역추적 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

**의미**: 더 많은 계산이 항상 더 나은 것은 아님 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

### 6.3 데이터 품질 한계
- **정확성**: 53.6% (일부 학습 데이터가 틀림) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- **편향**: Gemini에서 생성된 추론만 학습 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- **한계**: 신규 추론 패턴 발견 제한 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

***

## 7. 최신 연구(2020년 이후) 비교 분석
### 7.1 주요 경쟁 모델 비교
#### 7.1.1 OpenAI o1 (2024)
| 특성 | o1 |
|------|-----|
| 방식 | 대규모 RL (비공개) |
| 데이터 규모 | 수백만 개 (추정) |
| 공개 수준 | 매우 제한적 |
| AIME24 성능 | 44.6% (o1-preview) |
| 재현성 | 매우 낮음 |

**장점**: 우수한 성능  
**단점**: 폐쇄형, 비싼 API, 방법론 불명확

#### 7.1.2 DeepSeek R1 (January 2025)
| 특성 | r1 |
|------|-----|
| 방식 | 다단계: 거절 샘플링 → RL → SFT |
| 데이터 규모 | **800K 샘플** |
| 공개 수준 | 완전히 공개 |
| AIME24 성능 | 79.8% |
| 재현성 | 높음 |

**장점**: 
- 공개된 방법론
- 우수한 성능
- SFT-only 버전도 경쟁력 있음 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

**단점**:
- 매우 높은 계산 비용
- 800K 샘플 필요

#### 7.1.3 s1-32B (본 논문, January 2025)
| 특성 | s1-32B |
|------|--------|
| 방식 | SFT + 버짓 포싱 |
| 데이터 규모 | **1K 샘플** |
| 공개 수준 | 완전히 공개 |
| AIME24 성능 | 56.7% (BF 포함) |
| 재현성 | **매우 높음** |

**장점**:
- **극단적 샘플 효율성** (1/800 vs R1) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- 간단한 구현 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- 낮은 계산 비용 (26분) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- 완전 오픈소스 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

**단점**:
- 절대 성능은 r1보다 낮음 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- 테스트 타임 스케일링에 의존 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

### 7.2 테스트 타임 스케일링 방법 진화
#### 7.2.1 시간별 방법론 발전

| 시기 | 방법 | 논문 | 장점 | 단점 |
|------|------|------|------|------|
| 2023 | Chain-of-Thought | Wei et al. | 추론 명시화 | 성능 제한적 |
| 2023-2024 | Self-Consistency | Wang et al. | 다양성 활용 | 병렬 방식 비효율 |
| 2024 | 동적 할당 | Snell et al. | 문제 난이도 적응 | 복잡한 구현 |
| 2024 | o1 (RL 기반) | OpenAI | 강력한 성능 | 비공개, 비싼 학습 |
| 2024-2025 | DeepSeek R1 | DeepSeek | 공개된 RL | 높은 계산 비용 |
| 2025 | **Budget Forcing** | **본 논문** | **간단성 & 효율성** | **절대 성능** |

### 7.3 최신 관련 연구 추세 (2025년 현재)
#### 7.3.1 온도 차원 스케일링
**"On the Role of Temperature Sampling in Test-Time Scaling"** ()
- 다양한 온도에서 생성하여 성능 +7.3%
- 단일 온도 스케일링보다 효율적

#### 7.3.2 다중 라운드 생각
**"Think Twice: Enhancing LLM Reasoning by Scaling Multi-round Test-time Thinking"** ()
- 이전 답변을 프롬프트로 재입력
- QwQ-32B: 80.3% → 82.1% (AIME24)
- DeepSeek-R1: 79.7% → 82.0%

#### 7.3.3 테스트 타임 강화학습 (TTRL)
**"TTRL: Test-Time Reinforcement Learning"** ()
- 라벨 없는 데이터에서 RL
- 다수결을 보상으로 사용
- Qwen-2.5-Math-7B: 211% 개선 (AIME24)

#### 7.3.4 의료 도메인 확장
**"m1: Unleash the Potential of Test-Time Scaling for Medical Reasoning"** ()
- 의료 영역에 테스트 타임 스케일링 적용
- 최적 생각 토큰 예산: ~4K
- 과도한 생각은 오류 유발

#### 7.3.5 다국어 일반화 한계
**"Linguistic Generalizability of Test-Time Scaling"** ()
- 영어: 강한 스케일링 효과
- 다른 언어: 제한적 이득
- **중요**: 현재 테스트 타임 스케일링 방법들은 영어에 최적화됨

#### 7.3.6 종합 설문
**"A Survey on Test-Time Scaling in Large Language Models"** ()
- 50+개 방법 분류
- 4가지 차원: 무엇을, 어떻게, 어디서, 얼마나 잘
- 미래 방향: 더 많은 스케일링, 다양한 도메인, 효율성

### 7.4 s1이 최신 연구에 미친 영향
#### 7.4.1 샘플 효율성 기준 설정
본 논문은 **1K 샘플로 경쟁력 있는 성능** 달성이 가능함을 입증, 후속 연구들이 데이터 효율성에 집중하도록 영감 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

#### 7.4.2 버짓 포싱의 영향
- 간단한 메커니즘이 복잡한 RL과 경쟁 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- 다양한 도메인 (의료, 법률 등)에서 재사용
- 구현 용이 → 빠른 채택 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

#### 7.4.3 오픈 소스 투명성
- DeepSeek R1과 함께 추론 모델의 "민주화" 시작 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- 학계/산업계 모두 접근 가능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

***

## 8. 수식 및 핵심 알고리즘
### 8.1 다양성 선택 알고리즘 (Algorithm 1)
$$\text{Algorithm: } s1K \text{ 선택}$$

$$\text{Input: } D = \text{24,496 질문 (어려움 필터링 후)}$$
$$\text{Output: } S = \text{1,000개 선택된 질문}$$

**단계 1: AIME/GPQA 정답 선택**
$$\text{if } (\text{IsGeminiCorrect} \land (\text{IsAIME} \lor \text{IsGPQA})) \text{ then}$$
$$S \leftarrow S \cup \{\text{all correct solutions}\}$$

**단계 2: MATH 정답 (긴 추론) 선택**
$$\text{if } (\text{IsGeminiCorrect} \land \text{IsMATH} \land L(\text{trace}) > 5600) \text{ then}$$
$$S \leftarrow S \cup \{\text{correct MATH solutions with long traces}\}$$

**단계 3: 다양성 기반 샘플링**
$$\text{domains} \leftarrow \text{AllDomains}()$$
$$\text{while } |S| < 1000:$$
$$d \sim \text{Uniform}(\text{domains})$$
$$w_i = 2^{-\text{rank}_{L(t_i)}(i)} \text{ for } q_i \in \text{Domain}(d)$$
$$q \sim \text{Categorical}(w)$$
$$S \leftarrow S \cup \{q\}$$
$$\text{if } \text{Domain}(d) \text{ exhausted: domains} \leftarrow \text{domains} \setminus \{d\}$$

### 8.2 버짓 포싱의 수학적 표현
**상태**: 생성된 토큰 수 $T_t$, 최대 한계 $T_{\max}$, 최소 한계 $T_{\min}$

$$p(t_{i}) = \text{softmax}(\text{logits}_i)$$

**최대값 제약**:
$$\text{if } T_t \geq T_{\max}:$$

$$p(\langle\text{end thinking}\rangle) \leftarrow 1.0$$

$$\text{all other } p(t) \leftarrow 0$$

**최소값 제약**:

$$\text{if } T_t < T_{\min} \land t_i = \langle\text{end thinking}\rangle:$$

$$p(t_i) \leftarrow 0 \text{ (억제)}$$

$$\text{append "Wait" to sequence}$$

$$\text{continue generation}$$

**확장 가능성**: 다양한 문자열 반복

$$\text{prompt} \leftarrow \text{reasoning trace} + \text{["Wait", "Hmm", "Let's verify", ...]}$$

### 8.3 스케일링 메트릭
$$\text{Control} = \frac{|\{i: T_i \in [T_{\min}, T_{\max}]\}|}{N} \times 100\%$$

$$\text{Scaling} = \text{avg slope of piecewise linear fit}$$

$$\text{Performance} = \max(\text{accuracy across budgets})$$

***

## 9. 앞으로의 연구에 미치는 영향 및 고려 사항
### 9.1 긍정적 영향 및 기여
#### 9.1.1 테스트 타임 스케일링 민주화
**영향**: 전까지 불가능해 보였던 일이 가능함을 입증 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

$$\text{Before: } \text{Large-scale RL} \rightarrow \text{고비용, 재현 불가}$$
$$\text{After: } \text{SFT + Budget Forcing} \rightarrow \text{저비용, 완전 재현 가능}$$

구체적 숫자:
- 학습 시간: 26분 (h100 기준) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- 샘플: 1,000개 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- 코드: GitHub 공개 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

#### 9.1.2 샘플 효율성이 핵심이라는 메시지
후속 연구들이 다음을 강조하도록 유도: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- "더 큰 모델"보다 "더 나은 데이터" 중요 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- 큐레이션 전략 (품질, 다양성, 어려움) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- 활성화 vs 학습의 구분 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

#### 9.1.3 간단한 메커니즘의 힘 입증
"Wait" 한 단어가 우수한 제어 가능성 실현:
- 100% 제어성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- +15 스케일링 기울기 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- 56.7% AIME24 성능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

**교훈**: 복잡함이 항상 더 좋은 것은 아님 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

#### 9.1.4 재현성과 투명성의 가치
완전 오픈소스로 공개:
- 모델 가중치 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- 1K 샘플 데이터 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- 학습 코드 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- 평가 스크립트 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

**결과**: 수백 개의 후속 논문, 재사용, 변형 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

### 9.2 연구 중 고려할 사항
#### 9.2.1 테스트 타임 스케일링의 한계 인식

| 도메인 | 스케일링 효과 | 고려사항 |
|--------|---------------|---------|
| 수학 | **우수** (특히 AIME24) | 명확한 정답, 논리적 체계 |
| 과학 | **중간** (GPQA) | 지식 기반, 넓은 이해 필요 |
| 자연언어 | **약함** | 해석 여지, 문화적 맥락 |
| 의료 | **약함** | 과도한 생각은 오류 유발 |

**실제 최적 생각 토큰**: 도메인별로 4K~8K

#### 9.2.2 다국어 확대 필수
현재 연구 한계: **거의 모든 방법이 영어 중심**

**고려 사항**:
- 각 언어별로 데이터 큐레이션 필요
- 언어 특성 반영 필요
- 교차 언어 전이 가능성 낮음

#### 9.2.3 강화학습과의 조합 가능성

**질문**: SFT + Budget Forcing이 RL을 대체할 수 있는가?

**현재 증거**:
- 절대 성능: RL > SFT (79.8% > 56.7% on AIME24) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- 샘플 효율: SFT > RL (1K > 800K) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- **결론**: 상호 보완적 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

**미래 방향**:
$$\text{Performance} = \text{SFT}(\text{activation}) + \text{RL}(\text{exploration}) + \text{Budget Forcing}(\text{control})$$

#### 9.2.4 컨텍스트 윈도우 한계 극복

**현재 한계**: 6배 버짓 포싱 후 포화 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

**해결 방안**:
1. **재귀 기반 아키텍처**: 깊이를 명시적으로 늘림
2. **잠재 추론**: 토큰이 아닌 숨겨진 상태로 추론
3. **계층적 추론**: 상위 수준 계획, 하위 수준 실행

**참고**: 최신 연구 방향은 "인컨텍스트 최적화" → "잠재 공간 최적화"

#### 9.2.5 거절 샘플링의 역 스케일링 문제

**현상**: 더 많은 생성 = 더 많은 오류 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

**원인 분석**:
- 길이 = 자신감 감소 신호 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- 모델이 의심하고 역추적 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- 궁극적으로 잘못된 경로로 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

**해결책**:
- 품질 기반 선택 (길이 무시) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
- 프로세스 보상 모델 사용
- 신뢰도 점수 기반 필터링

### 9.3 미래 연구 로드맵
#### 9.3.1 단기 (2025)
```
1. 버짓 포싱 최적화
   - 다양한 문자열 발견
   - 문제별 적응형 예산

2. 도메인 확장
   - 의료 (m1 논문 참고)
   - 법률
   - 코드 생성

3. 효율성 개선
   - 추론 토큰 압축
   - 코드 기반 추론 (Z1)
```

#### 9.3.2 중기 (2025-2026)
```
1. 다국어 일반화
   - 언어별 데이터 큐레이션
   - 교차 언어 전이 연구

2. RL과 통합
   - SFT 초기화 + RL 미세조정
   - 예산 포싱과 RL 조합

3. 아키텍처 혁신
   - 재귀적 깊이 (Mesa, xLSTM)
   - 잠재 추론 (Scaling up with Latent)
```

#### 9.3.3 장기 (2026+)
```
1. 시스템-2 사고의 완성
   - 진정한 반사적 추론
   - 인간 수준의 자기 수정

2. 멀티모달 추론
   - 이미지-텍스트 통합 사고
   - 비주얼 추론 (ZoomEye)

3. 에이전트 추론
   - 도구 사용 통합 (Search-o1)
   - 행동-추론 루프
```

### 9.4 실무 적용 시 체크리스트
#### 9.4.1 데이터 큐레이션
$$\checkmark \text{ 어려움 필터링: 베이스 모델로 검증}$$
$$\checkmark \text{ 다양성: 50+ 도메인 커버}$$
$$\checkmark \text{ 품질: 포맷 검사, 오염제거}$$
$$\checkmark \text{ 크기: 1K 최소 (더 많을 필요 없음)}$$

#### 9.4.2 모델 선택
$$\checkmark \text{ 기반 모델: 이미 추론 능력 있는 것}$$
$$\checkmark \text{ 파라미터: 7B-70B (충분함)}$$
$$\checkmark \text{ 아키텍처: 특수 토큰 지원 필요}$$

#### 9.4.3 학습 설정
$$\checkmark \text{ 에포크: 3-5 (과적합 주의)}$$
$$\checkmark \text{ 배치 크기: 8-32 (자원 따라)}$$
$$\checkmark \text{ 학습률: 1e-5 ~ 5e-5}$$
$$\checkmark \text{ 손실 계산: 답변과 추론만}$$

#### 9.4.4 테스트 타임 설정
$$\checkmark \text{ 최대값: 4K-8K 생각 토큰}$$
$$\checkmark \text{ 문자열: "Wait"로 시작, 다양화}$$
$$\checkmark \text{ 반복: 2-6배 (과도한 반복 주의)}$$
$$\checkmark \text{ 검증: 성능 곡선으로 확인}$$

***

## 10. 결론
### 10.1 핵심 요약
s1 논문은 **테스트 타임 스케일링이 복잡한 강화학습 없이도 가능**함을 혁신적으로 입증했습니다. 세 가지 핵심 발견:

1. **데이터 품질이 양을 압도**: 1K 정교한 샘플 > 800K 일반 샘플 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
2. **버짓 포싱은 우아한 메커니즘**: 한 단어 ("Wait")로 완벽 제어 가능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)
3. **활성화가 학습을 능가**: 사전 학습 모델의 기존 능력 활용 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

### 10.2 실제 영향
| 영역 | 영향 |
|------|------|
| **연구 접근성** | 개인/소규모 팀도 추론 모델 개발 가능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf) |
| **계산 효율** | H100 26분으로 경쟁력 있는 성능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf) |
| **투명성** | 완전 오픈소스로 재현 가능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf) |
| **다음 세대** | 후속 100+ 논문의 기초 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf) |

### 10.3 남은 질문들
1. **왜 1K로 충분한가?** → 활성화 가설 (더 검증 필요)
2. **최적 예산은?** → 도메인마다 다름 (일반화된 전략 필요)
3. **RL은 필수인가?** → 보수적으로는 그렇지만 SFT도 경쟁력 있음
4. **다국어는?** → 현재 약점 (미해결)

### 10.4 최종 평가
**s1은 단순함의 힘을 보여주는 이정표 연구**입니다. 2024-2025년의 "테스트 타임 스케일링 시대"를 하나의 접근법(RL 중심)에서 다양한 경로(SFT + 버짓 포싱, 온도 스케일링, RL-프리 방법)로 민주화했습니다.

**앞으로의 시사점**:
- ✅ 샘플 효율성 우선 설계
- ✅ 도메인별 최적화 필수
- ✅ 다국어 대응 시급
- ✅ 컨텍스트 한계 극복 방법 탐색
- ✅ RL과의 시너지 연구

이 논문은 거대함이 아닌 **영리함**이 AI 진화의 미래임을 보여줍니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2d89c778-119f-40f4-8864-2f3b6107529e/2501.19393v3.pdf)

***

## 참고 문헌

<span style="display:none">[^1_10][^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_2][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_3][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_4][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_5][^1_6][^1_7][^1_8][^1_9]</span>

<div align="center">⁂</div>

[^1_1]: 2501.19393v3.pdf

[^1_2]: https://arxiv.org/abs/2504.13828

[^1_3]: https://arxiv.org/abs/2510.02611

[^1_4]: https://arxiv.org/abs/2503.19855

[^1_5]: https://arxiv.org/abs/2504.16084

[^1_6]: https://arxiv.org/abs/2505.15340

[^1_7]: https://arxiv.org/abs/2506.05233

[^1_8]: https://arxiv.org/abs/2411.16044

[^1_9]: https://aclanthology.org/2025.realm-1.27

[^1_10]: https://www.semanticscholar.org/paper/f26fcc2b9fc8944e054425d19c12b9d5cca64fcb

[^1_11]: https://arxiv.org/abs/2504.00869

[^1_12]: http://arxiv.org/pdf/2401.02954v1.pdf

[^1_13]: http://arxiv.org/pdf/2501.19393.pdf

[^1_14]: https://arxiv.org/pdf/2204.02311.pdf

[^1_15]: http://arxiv.org/pdf/2411.19477.pdf

[^1_16]: http://arxiv.org/pdf/2501.19306.pdf

[^1_17]: http://arxiv.org/pdf/2502.14382.pdf

[^1_18]: https://arxiv.org/pdf/2504.00810v1.pdf

[^1_19]: https://arxiv.org/abs/2502.05171

[^1_20]: https://iclr.cc/virtual/2025/32769

[^1_21]: https://www.emergentmind.com/topics/chain-of-thought-supervised-finetuning-sft

[^1_22]: https://huggingface.co/blog/Kseniase/testtimecompute

[^1_23]: https://aclanthology.org/2025.emnlp-main.1025.pdf

[^1_24]: https://www.nature.com/articles/s41586-025-09422-z

[^1_25]: https://introl.com/blog/inference-time-scaling-research-reasoning-models-december-2025

[^1_26]: https://arxiv.org/abs/2408.03314

[^1_27]: https://www.reddit.com/r/LocalLLaMA/comments/1ka0zov/finetuning_reasoning_models_without_messing_up/

[^1_28]: https://kimjy99.github.io/논문리뷰/deepseek-r1/

[^1_29]: https://onelineai.com/blog-research/linguistic-generalizability-of-test-time-scaling-in-mathematical-reasoning-en

[^1_30]: https://arxiv.org/pdf/2501.12948.pdf

[^1_31]: https://turingpost.co.kr/p/topic-26-test-time-compute

[^1_32]: https://discuss.pytorch.kr/t/deep-research-test-time-compute-test-time-scaling/6153

[^1_33]: https://magazine.sebastianraschka.com/p/understanding-reasoning-llms

[^1_34]: https://medtalk.tistory.com/entry/AI가-생각하는-시간을-갖게-되다-Test-time-Compute의-의미

[^1_35]: https://arxiv.org/html/2506.12928v1

[^1_36]: https://arxiv.org/html/2601.07180v1

[^1_37]: https://arxiv.org/html/2510.14232v1

[^1_38]: https://arxiv.org/html/2512.02008v1

[^1_39]: https://arxiv.org/html/2507.01921v1

[^1_40]: https://arxiv.org/html/2502.12215v1

[^1_41]: https://www.arxiv.org/pdf/2509.22230.pdf

[^1_42]: https://arxiv.org/abs/2503.16040

[^1_43]: https://arxiv.org/pdf/2501.19393.pdf

[^1_44]: https://arxiv.org/abs/2509.22230

[^1_45]: https://arxiv.org/abs/2504.01317

[^1_46]: https://arxiv.org/pdf/2505.11484.pdf

[^1_47]: https://arxiv.org/abs/2510.05132

[^1_48]: https://arxiv.org/abs/2510.03605
