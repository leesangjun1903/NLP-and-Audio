# Diffusion-LM Improves Controllable Text Generation

### 1. 핵심 주장과 주요 기여

**Diffusion-LM의 핵심 혁신**

Diffusion-LM 논문의 주요 주장은 **연속 확산 모델(Continuous Diffusion Models)을 이용한 비자귀적 언어 모델(Non-autoregressive Language Model)이 복잡한 미세한 텍스트 제어를 가능하게 한다**는 것입니다. 기존의 자귀적 언어모델(예: GPT 계열)은 왼쪽에서 오른쪽으로의 순차적 생성에 국한되어, 구문 구조나 의미적 내용과 같은 전역적 제약을 효과적으로 처리할 수 없었습니다. Diffusion-LM은 이러한 한계를 극복합니다.[1]

**주요 기여는 다음과 같습니다:**

1. **텍스트에 대한 연속 확산 모델의 첫 적용**: 텍스트의 이산적 특성을 극복하기 위해 연속 임베딩 공간에서 확산 과정을 수행합니다.[1]

2. **플러그 앤 플레이 제어의 획기적 성과**: 추가 학습 없이 기존 분류기를 활용하여 6가지 제어 작업에서 PPLM, FUDGE 등 기존 방법을 크게 능가합니다. 특히 구문 구조 제어에서 미세조정 모델까지 능가하는 성과를 달성했습니다.[1]

3. **계층적 잠재 표현의 활용**: 노이즈가 점진적으로 제거되는 과정에서 생성되는 계층적 연속 잠재 변수들이 **경사 기반 알고리즘(Gradient-based Algorithm)**을 통해 복잡한 제어를 가능하게 합니다.[1]

***

### 2. 해결하고자 하는 문제와 제안하는 방법

**문제 정의**

자귀적 언어모델의 주요 한계:
- **고정된 생성 순서**: 왼쪽에서 오른쪽으로만 생성 가능하여 오른쪽 문맥(Right Context)이 필요한 과제에 부적합
- **미세한 제어의 어려움**: 감정, 주제 같은 단순 속성 제어는 가능하지만, 구문 구조나 의미 내용과 같은 복잡한 세밀한 제어는 불가능
- **제어 작업의 구성 불가능**: 다중 제어 목표의 조합이 어려움

**제안 방법: Diffusion-LM**

**연속 확산 모정의 수학적 정식화**[1]

Diffusion-LM은 다음과 같은 마코프 연쇄로 정의됩니다:

$$p_\text{model}(x_0) = \int p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1}|x_t) dx_{1:T}$$

여기서 $$x_T \sim \mathcal{N}(0, I)$$는 가우시안 노이즈이고, 각 전이는:

$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t I)$$

**핵심 수정: 임베딩과 라운딩**[1]

1. **엔드-투-엔드 학습 목표**: 이산 단어 $$w$$를 연속 임베딩 공간으로 매핑

$$L_{\text{e2e}}^{\text{simple}}(w) = \mathbb{E}_{q(x_0^T|w)} \left[ \sum_{t=1}^{T} \|\mu_\theta(x_t, t) - \mu_\theta^*(x_t, x_0)\|^2 + \|\text{EMB}(w) - x_1\|^2 - \log p_w(x_0) \right]$$

2. **x₀-매개변수화(X0-Parametrization)**: 표준 확산 목표의 재매개변수화로 더 나은 라운딩 성과[1]

$$L_{\text{e2e}}^{\text{x0-simple}}(w) = \mathbb{E}_{q(x_0^T|w)} \left[ \sum_{t=1}^{T} \|f_\theta(x_t, t) - x_0\|^2 + \log p_w(x_0) \right]$$

여기서 $$f_\theta$$는 $$x_0$$를 직접 예측합니다.

3. **클램핑 트릭(Clamping Trick)**: 디코딩 중 예측 벡터를 가장 가까운 단어 임베딩으로 매핑하여 라운딩 오류 감소[1]

$$x_{t-1} = \text{Clamp}(f_\theta(x_t, t)) + \sqrt{1-\bar{\alpha}_{t-1}} \cdot \mathcal{N}(0, I)$$

**제어 가능 생성**[1]

제어 변수 $$c$$에 대해 사후 확률 $$p(x_0^T | c)$$에서 샘플링:

$$x_{t-1} \leftarrow x_{t-1} - \lambda \nabla_{x_{t-1}} \log p_\theta(x_{t-1}|x_t) - \nabla_{x_{t-1}} \log p_c(x_{t-1})$$

유창성 정규화 항 $$\lambda$$가 유창성(모델의 선호)과 제어(분류기의 선호)를 균형맞춥니다.[1]

***

### 3. 모델 구조

**전체 아키텍처**[1]

Diffusion-LM은 **80M 매개변수의 Transformer 기반 구조**로, 다음 요소들로 구성됩니다:

| 구성 요소 | 설명 |
|---------|------|
| **포워드 프로세스** | 이산 단어 → 가우시안 노이즈(T=2000 스텝)[1] |
| **역 프로세스** | Transformer 네트워크가 노이즈 제거 수행[1] |
| **임베딩 층** | 학습 가능한 단어 임베딩(E2E 데이터셋: d=16, ROCStories: d=128)[1] |
| **라운딩 층** | Softmax를 통한 확률 기반 단어 선택[1] |

**학습 특성**[1]

- **시퀀스 길이**: 64토큰
- **배치 크기**: 64
- **학습 반복**: E2E(200K), ROCStories(800K)
- **옵티마이저**: AdamW (초기 학습률 1e-4)
- **노이즈 스케줄**: 새로운 **sqrt 스케줄** 도입 $$\sigma_t = \sqrt{\frac{1-\bar{\alpha}_t}{\bar{\alpha}_t}}$$

***

### 4. 성능 향상 및 실험 결과

**6가지 제어 작업에서의 성과**[1]

| 제어 작업 | Diffusion-LM | FUDGE | PPLM | 미세조정(FT) |
|---------|:---:|:---:|:---:|:---:|
| 의미 내용 | 81.2% | 69.9% | 9.9% | 89.9% |
| 품사(POS) | 90.0% | 27.0% | - | 93.0% |
| 구문 트리 | 86.0% | 17.9% | - | 76.4% |
| 구문 스팬 | 93.8% | 54.2% | - | 54.4% |
| 길이 제어 | 99.9% | 46.9% | - | 100.0% |
| **평균 유창성(LM-Score)** | **2.55** | 3.39 | 5.32 | 3.31 |

주목할 점은 **Diffusion-LM이 구문 구조 제어에서 미세조정 모델까지 능가**한다는 것입니다.[1]

**다중 제어 조합**[1]

의미 내용 + 구문 제어를 동시에 수행:
- Diffusion-LM: 의미 69.8% + 구문 74.8%
- FUDGE: 의미 64.5% + 구문 24.1%

**삽입(Infilling) 작업**[1]

| 지표 | Diffusion-LM | COLD | DELOREAN | 자귀적 기준선 |
|------|:---:|:---:|:---:|:---:|
| BLEU-4 | 7.1 | 1.8 | 1.6 | 6.7 |
| CIDEr | 30.7 | 10.7 | 7.9 | 26.9 |
| BERTScore | 89.0 | 42.7 | 41.7 | 89.0 |

**절제 연구**[1]

| 설계 선택 | 영향 |
|---------|-----|
| 학습된 임베딩 vs 무작위 | ROCStories에서 1.2 lm-score 개선 |
| x₀-매개변수화 vs ε-매개변수화 | 높은 차원에서 안정적 성능 |
| Sqrt 노이즈 스케줄 | 모든 매개변수 설정에서 우수 |

***

### 5. 한계

**성능 상의 한계**[1]

1. **높은 복잡도**: 자귀적 Transformer(1.77 nats)에 비해 더 높은 NLL(2.28 nats on E2E, 3.88 on ROCStories)
   
2. **느린 디코딩**: 2000 확산 스텝으로 인해 자귀적 모델보다 7배 느림 (200 스텝 다운샘플링 후에도 여전히 느림)[1]

3. **느린 수렴**: 자귀적 모델에 비해 학습 속도가 느림

4. **생성 품질 대 계산 트레이드오프**: 우수한 제어 성능을 위해서는 높은 계산 비용이 필요

***

### 6. 모델의 일반화 성능 향상 가능성

**현재 일반화 능력**[1]

1. **임베딩 공간의 의미론적 구조**: t-SNE 시각화에서 같은 품사의 단어들이 자동으로 군집화되며, 이는 의미론적으로 의미 있는 표현을 학습했음을 시사합니다.[1]

2. **제어의 모듈성**: 학습된 분류기 없이도 새로운 제어 작업에 적응 가능 (예: 길이 제어는 분류기 불필요)[1]

3. **다양한 제어 조합**: 여러 독립적인 분류기를 조합하여 새로운 제어 조합 생성 가능 (일반화의 증거)[1]

**향후 일반화 성능 개선 방향**[1]

| 개선 방향 | 예상 효과 |
|---------|---------|
| **대규모 사전학습** | 더 많은 데이터로 임베딩 공간의 표현성 향상 |
| **모델 스케일링** | 더 큰 모델(300M 이상)로 복잡한 제어 능력 향상 |
| **하이브리드 접근** | 확산과 자귀적 방식의 장점 결합 |
| **적응적 노이즈 스케줄** | 작업별 최적의 노이즈 스케줄 학습 |

**최신 발전(2024-2025)**[2]

Google의 Gemini Diffusion은 Diffusion-LM의 아이디어를 확장하여 **상용 수준의 성능을 달성**했습니다:[2]
- 초당 1,479토큰 생성 (자귀적 모델의 5배 빠름)
- 코딩 작업에서 Gemini 2.0 Flash-Lite 초과 (30.9% vs 28.5%)
- TESS 2 모델은 대규모 데이터(8B 매개변수)에서 자귀적 모델과 경쟁력 있는 성능 달성[3]

***

### 7. 논문이 앞으로의 연구에 미치는 영향

**학문적 기여**[4][1]

1. **패러다임 전환의 촉매**: 텍스트 생성의 **다중 선택지**를 제시하며, 자귀적 방식에 의존하지 않는 새로운 가능성을 열었습니다.

2. **제어 가능한 생성의 새로운 표준**: 비자귀적, 플러그 앤 플레이 방식으로 **다중 제어의 모듈적 조합**이 가능함을 증명했습니다.

3. **수학적 엄밀성**: 확산 모델의 이론을 텍스트 도메인에 체계적으로 확장하는 수학적 틀을 제시합니다.[1]

**실제 응용 영역**

| 응용 분야 | 기대 효과 |
|---------|---------|
| **콘텐츠 필터링** | 독성 제거, 편향 완화, 안전성 향상 |
| **스타일 전이** | 미세한 문체 제어 및 개인화 |
| **다중 목표 최적화** | 품질, 길이, 의미 등 동시 제어 |
| **교정 및 재생성** | 오류 지역 수정 및 다시 생성 가능 |
| **데이터 증강** | 통제된 방식의 합성 데이터 생성 |

**후속 연구 방향**[5][6][7][4]

1. **이산 확산의 개선 (2023-2024)**:
   - **DiffusionBERT**: 토큰 정보 기반 적응적 노이즈 스케줄 도입
   - **SSD-LM**: 심플렉스 공간에서의 반자귀적 확산 모델
   - **InfoDiffusion**: 정보 엔트로피 기반 생성 순서 최적화[8]

2. **대규모 확산 언어모델 (2024-2025)**:
   - **TESS 2**: 대규모 지시사항 튜닝 확산 모델로 자귀적 모델과 경쟁[3]
   - **Large Language Diffusion Models (LLaDA)**: 2T 토큰으로 8B 모델 학습[9]
   - **Google Gemini Diffusion**: 상용급 성능 달성[2]

3. **이론적 분석**:
   - **일반화 특성 분석**: 확산 모델의 일반화 메커니즘 규명[10]
   - **하이브리드 접근법**: 연속-이산 확산의 통합 이론[11]

***

### 8. 앞으로 연구 시 고려할 점

**기술적 고려사항**

1. **계산 효율성 개선**
   - 현재 2000 스텝의 디코딩 비용이 주요 장애물
   - **가능한 해결책**: 지식 증류, 스텝 스킵 최적화, 병렬 디코딩 기법 활용

2. **길이 제어의 일반화**
   - Diffusion-LM은 고정 길이(64토큰)로 학습
   - **가능한 해결책**: 가변 길이 입력에 대한 데이터 증강, 반자귀적 방식 하이브리드

3. **대규모 모델 확장성**
   - 현재 80M 매개변수에서 테스트
   - **검증 필요**: 수십억 개 매개변수로 확장 시 성능 유지 가능성[3]

**방법론적 고려사항**

1. **제어 분류기 설계**
   - 효과적인 제어를 위해 분류기 품질이 중요
   - 약한 신호의 분류기도 활용 가능하지만, 명확한 신호는 더 나은 성과

2. **평가 지표의 다양화**
   - LM-Score(복잡도) 외에 BLEU, BERTScore 등 다중 지표 병행
   - 인간 평가의 중요성 강조

3. **제어 목표의 계층화**
   - 단순 속성(감정) → 구조(품사) → 복잡 제약(구문) 순서 학습
   - 커리큘럼 학습의 효과성

**연구 방향성**

1. **비자귀적 생성의 이점 극대화**
   - 양방향 문맥 활용이 필요한 작업에서의 우위 강화
   - 병렬 디코딩이 효율적인 시나리오 개발

2. **제어 다양성 확대**
   - 단순 분류기를 넘어 확률적 제약(예: "65% 확률로 긍정")
   - 열린-형식 제어(자유로운 텍스트 기반 제어)

3. **다중 모달 확장**
   - 텍스트-이미지, 텍스트-음성 모달리티 통합
   - 언어와 비언어 정보의 통합 제어

4. **안전성과 신뢰성**
   - 적대적 제어에 대한 견고성 연구
   - 생성 과정의 해석 가능성 향상

**기존 연구와의 통합**

1. **플러그 앤 플레이 방식의 표준화**
   - PPLM, FUDGE와의 공정한 비교를 위한 평가 프로토콜 확립
   - 다양한 제어 작업의 벤치마크 구축

2. **자귀적 모델과의 하이브리드**
   - 조기 단계: 자귀적 생성 (추론 속도)
   - 후기 단계: 확산 기반 재정제 (정교함)

3. **지식 통합**
   - 외부 지식 베이스(위키피디아, 본온코스)와의 통합
   - 제어된 생성에서의 사실성 향상

***

### 9. 2020년 이후 관련 최신 연구 탐색

**확산 모델 텍스트 생성의 진화 시계열**

**2020-2021: 기초 연구**
- Ho et al., 2020: DDPM(이미지) 성공 → 텍스트 적용의 동기 부여
- 초기 이산 확산 시도 (D3PM, Autoregressive Diffusion)

**2022: Diffusion-LM 출판**[1]
- 첫 연속 확산 언어모델, 구문 제어 달성
- 플러그 앤 플레이 제어의 혁신적 성과

**2023: 다양한 개선 시도**

- **DiffusionBERT**: BERT 구조 + 토큰 정보 기반 적응적 노이즈 스케줄[12]
- **SSD-LM**: 심플렉스 공간에서의 반자귀적 확산, 모듈성 강화[6]
- **Masked-Diffuse LM**: 언어학적 특징 활용, 구조화된 마스킹
- **DiffusER**: 편집 기반 재구성, 텍스트 재생성 능력[13]
- **InfoDiffusion**: 정보 엔트로피 기반 "키워드-우선" 생성 전략[8]

**2023-2024: 대규모 모델과 이론**

- **Fine-grained Text Style Transfer**: StylePTB에서 미세한 스타일 전이 달성[14]
- **GENIE**: 대규모 사전학습 확산 모델, 단락 단위 노이즈 제거[15]
- **Reparameterized Discrete Diffusion**: 이산 확산의 새로운 이론적 틀[7]
- **Segment-Level Diffusion (SLD)**: 토큰 의존성 무시 문제 해결, 세그먼트 레벨 생성[5]

**2024: 상용화와 성능 달성**

- **LaDiC**: 이미지-텍스트 생성에서 확산이 자귀적 모델과 경쟁 가능함을 입증[16][17]
  - BLEU: 38.2, CIDEr: 126.2 (MS COCO)
  - 양방향 문맥의 이점 강조

- **Energy-Based Diffusion Language Models**: 효율적 평행 중요도 샘플링으로 1.3배 속도 향상[18]

- **Self-Play Fine-Tuning (SPIN-Diffusion)**: 확산 모델의 자기 개선을 위한 강화학습 기법[19]

- **Diffusion Models in Text Generation Survey**: 포괄적 검토, 자귀적 모델과의 비교[20][21][4]

**2024-2025: 대규모 확산 언어모델의 성공**

| 모델 | 규모 | 주요 성과 | 참고 |
|-----|:---:|---------|------|
| **Google Gemini Diffusion** | 상용급 | 초당 1,479 토큰, 5배 빠른 생성 | [2] |
| **TESS 2** | 8B | 자귀적 모델과 경쟁하는 일반화 능력 | [3] |
| **Large Language Diffusion Models (LLaDA)** | 8B | 2T 토큰으로 학습, 우수한 추론 능력 | [9] |
| **Generalized Interpolating Discrete Diffusion (GIDD)** | - | 마스킹과 균일 노이즈 하이브리드, 자기 교정 능력 | [22] |

**특별 기여: 성능 비교 (2024-2025)**[2][3]

| 과제 | AR 모델 (예: Gemini 2.0) | 확산 모델 (Gemini Diffusion) |
|-----|:---:|:---:|
| 코딩 (LiveCodeBench) | 28.5% | 30.9% ✓ 초과 |
| 수학 | 우수 | 우수 |
| 일반 추론 (GPQA Diamond) | 56.5% | 40.4% (현재 약점) |
| 일반 지식 (Global MMLU) | 79.0% | 69.1% (개선 필요) |
| **추론 속도** | ~300 tokens/sec | 1,479 tokens/sec ✓ **5배 빠름** |

**이론적 진전**[23][24][10]

1. **일반화 특성 분석**: 확산 모델의 로컬 귀납적 편향(Local Inductive Bias)이 일반화를 설명
   
2. **도메인 일반화**: 확산 모델의 잠재 공간이 명시적 도메인 레이블 없이도 도메인별 변량 분리 (최대 4% 정확도 개선)[23]

3. **이론적 우선순위**: 연속-이산 확산 통합 이론 개발, ELBO 기반 성능 최적화[11]

**앞으로의 기대 방향**[25][2]

1. **하이브리드 아키텍처의 주류화**: 자귀적 + 확산의 장점 결합
2. **상용 배포 확대**: Google Gemini Diffusion의 성공에 따른 산업계 채택 증가
3. **멀티모달 통합**: 텍스트-이미지-음성의 통합 생성 모델
4. **적응적 제어**: 작업별 최적 제어 전략 자동 선택

***

### 결론

Diffusion-LM은 **텍스트 생성에서 진정한 제어 가능성과 유연성**을 구현한 획기적 연구입니다. 비자귀적 생성의 장점을 최대한 활용하면서도 플러그 앤 플레이 방식의 제어를 실현했습니다. 

2022년 발표 이후 2024-2025 현재, Google의 Gemini Diffusion이 상용급 성능을 달성함으로써 Diffusion-LM의 아이디어가 실제로 현실화되었습니다. 향후 연구는 **계산 효율성 개선, 대규모 모델 확장, 복잡한 추론 능력 강화**에 초점을 맞춰야 하며, 특히 자귀적 모델과의 하이브리드 접근법이 차세대 언어모델의 표준이 될 가능성이 높습니다.

제어 가능한 생성, 병렬 디코딩, 양방향 문맥 모델링이라는 고유의 장점을 갖춘 확산 기반 언어모델은 **AI 안전성, 멀티모달 생성, 대규모 시스템의 효율성** 측면에서 중요한 역할을 할 것으로 기대됩니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/96bef0b4-c94d-4e15-9ffe-1a8fdeeefe5b/2205.14217v1.pdf)
[2](https://huggingface.co/blog/ProCreations/diffusion-language-model)
[3](https://aclanthology.org/2025.acl-long.1029.pdf)
[4](https://peerj.com/articles/cs-1905)
[5](http://arxiv.org/pdf/2412.11333.pdf)
[6](https://arxiv.org/pdf/2210.17432.pdf)
[7](https://arxiv.org/pdf/2302.05737.pdf)
[8](https://aclanthology.org/2023.findings-emnlp.919.pdf)
[9](https://openreview.net/forum?id=KnqiC0znVF)
[10](https://arxiv.org/pdf/2311.01797.pdf)
[11](https://aclanthology.org/2025.acl-long.565.pdf)
[12](https://aclanthology.org/2023.acl-long.248.pdf)
[13](https://arxiv.org/pdf/2210.16886.pdf)
[14](https://aclanthology.org/2023.repl4nlp-1.6.pdf)
[15](https://arxiv.org/pdf/2212.11685.pdf)
[16](https://arxiv.org/abs/2404.10763)
[17](https://aclanthology.org/2024.naacl-long.373.pdf)
[18](https://openreview.net/forum?id=sL2F9YCMXf)
[19](https://arxiv.org/abs/2402.10210)
[20](https://pmc.ncbi.nlm.nih.gov/articles/PMC10909201/)
[21](https://peerj.com/articles/cs-1905/)
[22](https://arxiv.org/pdf/2503.04482.pdf)
[23](http://arxiv.org/pdf/2503.06698.pdf)
[24](https://arxiv.org/html/2411.19339v2)
[25](https://www.youtube.com/watch?v=1AqxiCeI-ZY)
[26](https://ieeexplore.ieee.org/document/10678076/)
[27](https://ieeexplore.ieee.org/document/10678525/)
[28](https://ieeexplore.ieee.org/document/10678183/)
[29](https://ieeexplore.ieee.org/document/10495711/)
[30](https://dl.acm.org/doi/10.1145/3707292.3707367)
[31](https://ieeexplore.ieee.org/document/10716806/)
[32](https://arxiv.org/abs/2401.05252)
[33](https://openreview.net/pdf?id=3s9IrEsjLyk)
[34](https://machinelearning.apple.com/research/non-autoagressive-neural-machine)
[35](https://aclanthology.org/2025.wnut-1.9.pdf)
[36](https://aclanthology.org/2024.findings-acl.452/)
[37](https://arxiv.org/abs/2205.14217)
[38](https://www.amazon.science/publications/non-autoregressive-sequence-to-sequence-vision-language-models)
[39](https://aclanthology.org/2024.naacl-long.261/)
[40](http://pubs.rsna.org/doi/10.1148/radiol.240609)
[41](https://journals.lww.com/10.1097/JS9.0000000000001850)
[42](https://www.isca-archive.org/interspeech_2024/li24y_interspeech.html)
[43](https://arxiv.org/abs/2405.07490)
[44](http://medrxiv.org/lookup/doi/10.1101/2024.10.15.24315526)
[45](https://arxiv.org/abs/2406.01432)
[46](http://pubs.rsna.org/doi/10.1148/radiol.240885)
[47](https://www.ahajournals.org/doi/10.1161/circ.150.suppl_1.4128519)
[48](http://www.proceedings.com/079017-3120.html)
[49](https://arxiv.org/abs/2402.00861)
[50](https://arxiv.org/pdf/2210.15629v3.pdf)
[51](https://arxiv.org/pdf/2406.11473.pdf)
[52](https://openreview.net/pdf?id=OpzV3lp3IMC)
[53](https://www.reddit.com/r/MachineLearning/comments/1c53pc5/diffusion_versus_autoregressive_models_for_image/)
[54](https://arxiv.org/html/2502.13917v1)
[55](https://www.sciencedirect.com/science/article/abs/pii/S0950705125014765)
[56](https://www.linkedin.com/posts/miltonmattox_artificialintelligence-machinelearning-activity-7306043505984843776-iW7Q)
[57](https://arxiv.org/abs/2505.15045)
[58](https://arxiv.org/html/2502.09622v1)
