
# ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장 (Core Claim)

기존의 대형 비전-언어 멀티모달 모델들은 이미지 전체 수준의 이해에 집중해 왔으며, 특정 영역(region)을 이해하는 데는 뚜렷한 공백이 존재한다. 텍스트 좌표나 공간 인코딩을 사용하는 기존 접근 방식은 시각적 프롬프팅을 위한 사용자 친화적 인터페이스를 제공하는 데 실패하는 경우가 많다.

이 논문의 핵심 주장은, 이러한 한계를 극복하기 위해 **임의의(arbitrary) 시각적 프롬프트를 이해할 수 있는 새로운 멀티모달 모델 ViP-LLaVA**를 제안한다는 것이다.

이 모델의 단순한 설계는 시각적 마커를 RGB 이미지 위에 직접 오버레이함으로써 복잡한 영역 인코딩이 필요 없으며, Visual7W, PointQA, Visual Commonsense Reasoning 벤치마크와 같은 영역 이해 태스크에서 최신 성능(state-of-the-art)을 달성한다.

---

### 1.2 주요 기여 (Contributions)

① 자연어와 임의의 시각적 프롬프트를 활용해 이미지와 직관적으로 상호작용하는 새로운 멀티모달 모델을 도입, 사용자 접근성과 모델 유연성을 강화하였다. ② 시각적 프롬프트를 이미지 위에 직접 오버레이하는 시각적 참조(visual referral) 접근 방식을 개발하여 성능 저하 없이 모델 아키텍처를 단순화하였다. ③ ViP-LLaVA는 기성 영역 인코딩 모델들을 능가하는 영역 이해 태스크의 최신 성능을 달성하였다. ④ 시각적 프롬프트 해석 역량을 평가하기 위한 ViP-Bench 벤치마크를 도입하여 향후 연구의 기반 플랫폼을 마련하였다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존의 영역 이해 연구들은 시각적 프롬프트를 처리하기 위해 새로운 모듈을 별도로 구성하는 방식을 채택해 왔다. 반면 ViP-LLaVA는 이미지와 겹쳐진 시각적 마커 모두를 인코딩하기 위해 CLIP의 기존 능력을 활용한다.

특히 CLIP이 시각적 데이터와 텍스트 데이터를 정렬하는 데 탁월하며, 최근 연구에 따르면 CLIP은 원, 사각형 등의 표시된 영역에 내재적으로 주의를 기울이는 능력을 가지고 있다.

구체적으로 해결하고자 하는 세 가지 문제는 다음과 같다:

1. **영역-특화(region-specific) 이해 부재**: 전체 이미지 수준의 이해에만 집중된 기존 LMM의 한계
2. **비직관적 인터페이스**: 텍스트 좌표 기반의 사용자 비친화적 상호작용 방식
3. **시각적 프롬프트 평가 기준 부재**: 기존 벤치마크는 영역-수준의 자유 형식 시각적 프롬프트를 종합적으로 평가하지 못함

---

### 2.2 제안하는 방법 (수식 포함)

#### (A) Alpha Blending (시각적 프롬프트 합성)

시각적 프롬프트를 원본 이미지 위에 알파 블렌딩(alpha blending)한 후, 결과 이미지를 시각적 인코더에 입력하여 다중 레벨(multi-level) 시각적 피처를 얻는다. 이 피처들을 연결(concatenate)하여 LayerNorm과 MLP 레이어를 통해 시각적 토큰(visual tokens)을 형성한다. 이후 시각적 토큰과 텍스트 명령어 토큰을 대형 언어 모델에 입력하여 자동회귀(auto-regressive) 방식으로 언어 응답을 생성한다.

알파 블렌딩 수식은 다음과 같이 정의된다:

$$\hat{\mathbf{X}}_{\mathrm{v}} = \alpha \cdot \mathbf{P}_{\mathrm{v}} + (1 - \alpha) \cdot \mathbf{X}_{\mathrm{v}}$$

여기서:
- $\hat{\mathbf{X}}_{\mathrm{v}}$: 시각적 프롬프트가 합성된 최종 이미지
- $\mathbf{P}_{\mathrm{v}}$: 시각적 프롬프트 마커 (화살표, 박스, 원, 낙서 등)
- $\mathbf{X}_{\mathrm{v}}$: 원본 RGB 이미지
- $\alpha$: 블렌딩 계수 (visual prompt 강도 조절)

#### (B) 훈련 목적 함수 (Autoregressive Language Modeling)

ViP-LLaVA는 자동회귀 언어 모델링을 통해 학습된다. 목적 함수는 정답 토큰 시퀀스의 조건부 로그 우도를 최대화하는 것으로 정의된다:

$$\mathcal{L} = \sum_{i=1}^{L} \log P\!\left(x_{a,i} \mid \hat{\mathbf{X}}_{\mathrm{v}},\, \mathbf{X}_{\mathrm{inst}},\, \mathbf{X}_{a,<i}\right)$$

여기서:
- $x_{a,i}$: $i$번째 답변 토큰
- $\hat{\mathbf{X}}_{\mathrm{v}}$: 알파 블렌딩된 시각적 입력
- $\mathbf{X}_{\mathrm{inst}}$: 텍스트 명령어 토큰
- $\mathbf{X}_{a,<i}$: 현재 예측 토큰 이전의 모든 답변 토큰
- $L$: 답변 시퀀스 전체 길이

#### (C) 다중 레이어 CLIP 피처 추출

시각적 프롬프트를 효과적으로 인식하기 위해 ViP-LLaVA는 저수준(low-level)과 고수준(high-level) 시각적 피처를 균형 있게 활용한다. CLIP의 깊은 레이어 피처가 저수준 세부사항을 간과하는 경향을 보완하기 위해 여러 CLIP 레이어에서 선택적으로 피처를 추출한다. 구체적으로, 하나의 초기 레이어(6번째)는 세밀한 기하학적 형태를 인코딩하고, 네 개의 깊은 레이어(15, 18, 21, 24번째)는 더 넓은 의미론적(semantic) 정보를 포착하는 데 사용된다.

이를 수식으로 표현하면 아래와 같다:

$$\mathbf{F}_{\mathrm{multi}} = \text{Concat}\!\left[\mathbf{f}^{(6)},\, \mathbf{f}^{(15)},\, \mathbf{f}^{(18)},\, \mathbf{f}^{(21)},\, \mathbf{f}^{(24)}\right]$$

$$\mathbf{V}_{\mathrm{tokens}} = \text{MLP}\!\left(\text{LayerNorm}\!\left(\mathbf{F}_{\mathrm{multi}}\right)\right)$$

여기서 $\mathbf{f}^{(l)}$은 CLIP의 $l$번째 레이어에서 추출된 피처 맵이다.

---

### 2.3 모델 구조 (Model Architecture)

다양한 시각적 프롬프트(화살표, 박스, 원, 낙서 등)를 원본 이미지 위에 직접 오버레이한 후, 해당 시각적 피처를 텍스트 임베딩과 함께 대형 멀티모달 모델에 입력하여 대화형 지원을 제공한다.

**전체 구조 요약:**

| 구성 요소 | 세부 내용 |
|:---|:---|
| **시각적 인코더** | CLIP ViT-L/14 336px (다중 레이어 피처 융합) |
| **프롬프트 합성** | Alpha Blending (원본 이미지 + 시각적 마커) |
| **비전-언어 커넥터** | 2-layer MLP with GELU (`mlp2x_gelu`) |
| **언어 모델 백본** | Vicuna v1.5 (7B / 13B) |
| **응답 생성** | Auto-regressive decoding |

ViP-LLaVA의 학습은 세 단계로 구성된다: (1) **피처 정렬 단계**: LAION-CC-SBU 데이터셋의 558K 서브셋을 사용하여 동결된 사전학습 비전 인코더와 동결된 LLM을 연결한다. (2) **시각적 명령어 튜닝 단계**: LLaVA-1.5의 665K 이미지-수준 명령어 데이터와 시각적 프롬프트를 활용한 520K 영역-수준 명령어 데이터를 사용한다. (3) **GPT-4V 데이터로 파인튜닝**하는 단계.

ViP-LLaVA의 영역 수준 멀티모달 대화 역량을 강화하기 위해 GPT-4V를 활용한 영역-특화 명령어 데이터를 설계하였다. 기존의 Shikra와 같은 접근 방식은 GPT4와 같은 텍스트 전용 모델을 사용하여 영역-수준 명령어 데이터를 생성하려 했으나, 이 방법은 시각적 맥락이 없어 단일 장면 내에서 동일 클래스의 여러 객체를 정확하게 참조하지 못하는 근본적인 한계를 가진다. 이를 극복하기 위해 GPT-4V를 활용한 명령어 데이터 큐레이션 방법을 개발하였다.

---

### 2.4 성능 향상

ViP-LLaVA-7B는 Visual7W에서 **86.09**, PointQA-LookTwice에서 **71.31**, RegionBench@Box에서 **48.4**, RegionBench@Human에서 **48.3**을 달성하였으며, ViP-LLaVA-13B는 Visual7W **88.28**, PointQA **71.77**를 달성하였다.

결과적으로 ViP-LLaVA-7B는 파라미터 수가 더 적음에도 불구하고 GPT4RoI 및 Shikra를 포함한 최신 방법들을 능가하며, ViP-LLaVA-13B는 더 큰 성능 향상을 달성하였다.

시각적 프롬프트는 PointQA-LookTwice와 ViP-Bench@Box 데이터셋에서 좌표 형식보다 월등히 높은 성능을 나타낸다.

**ViP-Bench 결과:**

GPT-4V는 여전히 제로샷 시각적 프롬프팅 이해에서 가장 강력한 멀티모달 모델이다. ViP-LLaVA는 ViP-Bench에서 인상적인 성능을 보이는 반면, Kosmos-2와 같은 대부분의 영역-특화 멀티모달 모델은 이미지-수준 멀티모달 모델보다도 낮은 성능을 보여준다.

---

### 2.5 한계 (Limitations)

현재 영역-수준 LMM들(Shikra, GPT4ROI, Kosmos-2 포함)은 수학, 관계 추론, 언어 생성 등을 포함하는 태스크에서 어려움을 겪는다. 이러한 경향은 주로 짧은 설명을 특징으로 하는 기존의 공개 영역-수준 데이터셋에 대한 잠재적 과적합(overfitting) 문제를 시사한다.

Llama-3-8B 및 Phi-3-mini-3.8B는 주로 시각적 이해 능력보다는 언어 추론 능력이 필요한 태스크에서 성능 향상을 가져오지 못한다. 예를 들어 Vicuna-1.5-13B는 여전히 MME, TextVQA, GQA, Visual7W, PointQA에서 더 나은 성능을 보인다. 이 결과는 핵심 시각적 이해 능력이 주로 필요한 태스크에서 더 나은 시각적 표현이 중요함을 나타낸다.

추가적인 한계로는:
- **GPT-4V와의 성능 격차**: 대부분의 경우 GPT-4V는 시각적 프롬프트 이해에서 강력하고 견고한 성능을 보인다.
- **연구 목적 전용 라이선스**: 데이터셋은 CC BY NC 4.0 라이선스(비상업적 사용만 허용)이며, 데이터셋을 사용해 학습한 모델은 연구 목적 이외로 사용할 수 없다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 훈련되지 않은 프롬프트 유형으로의 일반화

ViP-LLaVA는 마스크 윤곽, 타원, 바운딩 박스, 삼각형, 낙서, 점, 화살표, 마스크 등 8가지 유형의 시각적 프롬프트로 훈련되었음에도 주목할 만한 일반화 능력을 보인다. 메인 논문에서 ViP-LLaVA는 인간이 직접 그린 시각적 프롬프트를 이해할 수 있음을 보여준다. 또한 명시적으로 훈련되지 않은 다양한 두께와 마커로도 시각적 프롬프트를 적절히 처리한다.

더 나아가, Set-of-Mark에서 영감을 받아 텍스트 마커를 시각적 프롬프트로 효과적으로 해석하는 기능도 갖추고 있다.

### 3.2 LLM 백본 교체를 통한 일반화

이 연구는 이미지-수준 및 영역-수준 비전-언어 벤치마크 모두에 대한 대형 언어 모델(LLM) 백본의 영향을 조사하였다. Vicuna-1.5-7B, Vicuna-1.5-13B, Llama-3-8B, Phi-3-mini-3.8B 등 다양한 모델을 LLaVA-1.5와 ViP-LLaVA 모두에 대한 언어 모델 백본으로 활용하여 모든 기타 구성과 하이퍼파라미터를 일관되게 유지하였다.

최신 LLM인 Llama-3와 Phi-3는 언어 및 상식 추론이 필요한 태스크에서 탁월하며, Llama-3-8B 및 Phi-3-mini-3.8B는 MMBench와 ScienceQA와 같은 벤치마크에서 Vicuna-1.5-13B를 크게 능가한다.

### 3.3 의료 도메인으로의 전이(Transfer)

ViP-LLaVA 7B를 MedTrinity-20M 데이터셋의 서브셋으로 사전 학습시켜 의료 지식을 주입한 사례가 있다. 이 데이터셋은 이미지, ROI(관심 영역), 설명 형식의 삼중항으로 구성되어 있으며, 각 ROI는 바운딩 박스로 주석이 달린 이상 소견에 해당한다. 이 의료 이미지에 대한 사전 학습은 ViP-LLaVA에 도메인별 지식을 제공하여, 의료 환경에서 영역-특화 정보에 집중할 수 있게 한다.

### 3.4 단순 설계의 일반화 이점

직접 오버레이라는 단순한 설계는 여러 장점을 제공한다. 추가 처리 모듈을 우회함으로써 모델 복잡성을 줄이고, 사용자들이 다양하고 즉흥적인 시각적 마커를 자주 사용한다는 점에서 자연스러운 인간의 상호작용에 가깝게 정렬된다. 이러한 유연성은 ViP-LLaVA가 광범위한 사용자 생성 시각적 단서를 해석할 수 있게 하여, 실제 시나리오에서 적용 가능성을 높인다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

| 모델 | 발표 연도 | 영역 참조 방식 | 장점 | 단점 |
|:---|:---|:---|:---|:---|
| **Shikra** | 2023 | 텍스트 좌표(수치) 기반 | 별도 모듈 불필요 | 직관적이지 않음, 시각적 맥락 부재 |
| **Kosmos-2** | 2023 | 이산 위치 토큰 | 강력한 Grounding | 자유 형식 프롬프트 미지원 |
| **GPT4RoI** | 2023 | RoI 풀링 기반 특징 추출 | 영역 이해 특화 | 아키텍처 복잡성 증가 |
| **RegionGPT** | 2024 | 영역-수준 임베딩 삽입 | 다양한 형태 RoI 지원 | 추가 훈련 데이터 필요 |
| **ViP-LLaVA** | 2024 | 이미지 직접 오버레이 | 직관적, 구조 단순, SOTA | GPT-4V 대비 성능 한계 |
| **ControlMLLM** | 2024 | 학습 가능한 잠재 변수 최적화 | 학습 불필요(Training-free) | 도메인 외 일반화 제한 |
| **Groma** | 2024 | 지역화된 시각적 토크나이제이션 | 세밀한 시각적 인식 | 복잡한 아키텍처 |

최근 영역-인식 MLLM들인 KOSMOS-2, Shikra, MiniGPT-2, LLaVA 등은 텍스트 형태로 영역 정보를 입력하는 방식을 채택하며, 위치 해석을 언어 디코더에 크게 의존한다.

기존 접근 방식들은 좌표 표현 또는 RoI 피처를 학습하는 방향으로 연구되어 왔으며, 이는 유연하지 못한 시각적 참조 형식을 사용하거나 영역-수준 훈련 데이터 수집을 필요로 한다.

MLLM에 효과적으로 지시를 내리기 위해, 기존 언어 표현에 더하여 이미지에 브러시로 그림을 그려 객체를 참조하는 방식이 사용자의 의도를 특정 이미지 영역과 정렬하는 데 효과적인 도구로 부상하고 있다. 점, 박스, 마스크와 같은 가장 일반적인 시각적 프롬프트를 수용하기 위해, 기존 접근법들은 강조된 영역의 의미론을 포착하기 위한 특화된 피처 인코딩 모듈을 활용한다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5.1 앞으로의 연구에 미치는 영향

**① 벤치마크 표준화 기여**

ViP-Bench는 멀티모달 모델이 시각적 프롬프트를 이해하는 능력을 종합적으로 평가하는 최초의 제로샷 영역-수준 벤치마크이다. ViP-Bench는 인식, OCR, 지식, 수학, 객체 관계 추론, 언어 생성 등 6가지 능력을 평가하는 303개의 샘플로 구성된다. ViP-Bench는 바운딩 박스 형식과 사람이 주석을 달한 임의의 시각적 프롬프트 두 가지 형식을 갖추고 있다.

**② 인간-기계 상호작용 패러다임 전환**

임의의 시각적 프롬프트를 통합함으로써, ViP-LLaVA는 사용자 친화적 인터페이스와 영역 이해에 요구되는 정밀성 사이의 간극을 해소한다. ViP-LLaVA의 직관적 설계는 자연어 상호작용과 시각적 마커를 결합하여 이미지 주석 과정을 단순화하면서 시각적 참조의 명확성을 향상시킨다.

**③ 의료, 자율주행 등 응용 분야 확장**

최근 연구들은 시각적 프롬프트의 가치와 비전-언어 태스크에서의 응용을 점점 더 인정하고 있다. 이 방법론은 의료 영상 분석, 자율 주행, 산업 검사 등 특정 영역의 세밀한 이해가 요구되는 분야에 응용될 수 있다.

**④ 후속 연구의 기반 제공**

ViP-LLaVA는 지능형 시각 시스템 분야의 추가 탐구를 위한 기반을 마련하고 있다. ViP-LLaVA는 시각적 양상과 언어적 양상이 통합되는 방식을 동기 부여하여, 보다 정교하고 세밀한 인간-기계 상호작용을 가능하게 할 수 있다.

---

### 5.2 앞으로 연구 시 고려할 점

**① 시각적 표현 품질 향상**

더 나은 시각적 표현이 핵심 시각적 이해 능력이 주로 필요한 태스크에서 중요하다. 따라서 더 강력한 비전 인코더(예: DINOv2, InternViT 등)의 통합이 필요하다.

**② 오버레이 방식의 세밀도 개선**

알파 블렌딩 방식은 단순하지만, 프롬프트가 복잡하거나 객체가 겹칠 경우 $\alpha$ 값 선택에 따라 원본 이미지 정보가 손상될 수 있다. 적응적(adaptive) 알파 조절 메커니즘 연구가 필요하다:

$$\alpha^* = \underset{\alpha}{\arg\max}\; \mathcal{L}_{\text{region}}\!\left(\hat{\mathbf{X}}_{\mathrm{v}}(\alpha)\right) + \lambda \cdot \mathcal{L}_{\text{fidelity}}\!\left(\hat{\mathbf{X}}_{\mathrm{v}}(\alpha), \mathbf{X}_{\mathrm{v}}\right)$$

**③ 다국어·다모달 확장**

현재 ViP-LLaVA는 주로 영어 기반 명령어 데이터로 학습되어 있다. 한국어, 중국어 등 다국어 시각적 프롬프팅 데이터셋 구축 및 훈련이 필요하다.

**④ 실시간·경량화 연구**

ViP-LLaVA-13B 사전학습은 8x A100 (80G)에서 약 5.5시간이 소요되며, 7B는 약 3.5시간이 소요된다. 실제 배포를 위해서는 모델 경량화(knowledge distillation, quantization 등) 연구가 병행되어야 한다.

**⑤ 비디오 및 3D 확장**

ViP-Bench가 주로 도메인-특화 인스턴스-수준 참조 이해에 집중하는 반면, 더 넓은 범위의 크로스-인스턴스 및 크로스-타임스탬프 상호작용 등을 포함하는 벤치마크가 필요하다. 비디오 및 3D 공간에서의 시각적 프롬프팅 확장 연구가 중요하다.

**⑥ 적대적 공격 내성**

시각적 프롬프트 오버레이 방식은 적대적 공격(adversarial attack)에 취약할 수 있어, 모델의 견고성(robustness) 연구가 필요하다.

---

## 📚 참고 자료 및 출처

| 번호 | 제목 / 출처 |
|:---|:---|
| 1 | **[논문 원문]** Cai, M., et al. "ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts." CVPR 2024. arXiv:2312.00784 |
| 2 | **[arxiv]** https://arxiv.org/abs/2312.00784 |
| 3 | **[프로젝트 페이지]** https://vip-llava.github.io/ |
| 4 | **[GitHub 코드]** https://github.com/WisconsinAIVision/ViP-LLaVA |
| 5 | **[CVPR 2024 Open Access]** https://openaccess.thecvf.com/content/CVPR2024/papers/Cai_ViP-LLaVA_Making_Large_Multimodal_Models_Understand_Arbitrary_Visual_Prompts_CVPR_2024_paper.pdf |
| 6 | **[IEEE Xplore]** https://ieeexplore.ieee.org/document/10657559 |
| 7 | **[HuggingFace Model]** https://huggingface.co/llava-hf/vip-llava-7b-hf |
| 8 | **[관련 연구]** "RegionGPT: Towards Region Understanding Vision Language Model." CVPR 2024. |
| 9 | **[관련 연구]** "Groma: Localized Visual Tokenization for Grounding Multimodal Large Language Models." ECCV 2024. |
| 10 | **[관련 연구]** "Visual Prompting in Multimodal Large Language Models: A Survey." arXiv:2409.15310 |
| 11 | **[관련 연구]** "ControlMLLM: Training-Free Visual Prompt Learning for Multimodal Large Language Models." NeurIPS 2024. |
| 12 | **[관련 연구]** "Guiding Medical Vision-Language Models with Explicit Visual Prompts." arXiv:2501.02385 |
| 13 | **[LLM 백본 비교 연구]** https://github.com/WisconsinAIVision/ViP-LLaVA/blob/main/docs/study_llm_backbone.md |
