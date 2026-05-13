
# SANA 1.5: Efficient Scaling of Training-Time and Inference-Time Compute in Linear Diffusion Transformer 

---

## 1. 핵심 주장 및 주요 기여 요약

이 논문은 텍스트-이미지 생성에서 효율적인 스케일링을 위한 **선형 Diffusion Transformer**, SANA-1.5를 제안합니다. 텍스트-이미지 Diffusion 모델은 지난 1년간 놀라운 발전을 이루었으나, 더 큰 모델 크기를 향한 뚜렷한 트렌드와 함께 막대한 연산 비용이 수반되어 왔습니다. 예를 들어, 최근 산업 모델들은 PixArt의 0.6B 파라미터에서 Playground v3의 24B까지 급격히 성장하여 대부분의 연구자들에게 훈련·추론 비용이 감당하기 어려운 수준이 되었습니다.

이와 대조적으로 SANA-1.0은 계산 요구사항을 크게 줄이면서도 경쟁력 있는 성능을 달성한 효율적인 선형 Diffusion Transformer를 도입했습니다. 이 기반 위에서 본 연구는 ① 선형 Diffusion Transformer의 확장성은 어느 수준인가, ② 대형 선형 DiT를 어떻게 효율적으로 확장하면서 훈련 비용을 절감할 수 있는가라는 두 가지 근본적인 질문을 탐구합니다.

### 세 가지 핵심 혁신

**(1) 효율적인 훈련 스케일링**: 1.6B에서 4.8B 파라미터로의 스케일링을 크게 줄어든 연산 자원으로 가능하게 하는 **Depth-Growth Paradigm** — 메모리 효율적인 8-bit 옵티마이저와 결합.

**(2) 모델 깊이 가지치기(Depth Pruning)**: 최소한의 품질 손실로 임의의 크기로 효율적으로 모델을 압축하는 **블록 중요도 분석(Block Importance Analysis)** 기법.

**(3) 추론 시간 스케일링(Inference-Time Scaling)**: 파라미터 용량 대신 연산량을 활용하여 소형 모델이 대형 모델 수준의 품질을 달성할 수 있는 **반복 샘플링 전략**.

SANA-1.5 (Inference-time scaling)는 ICML-2025에 accept되었습니다.

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능, 한계

### 2-1. 해결하고자 하는 문제

텍스트-이미지 Diffusion 모델은 모델 크기 증가로 인해 막대한 연산 비용이 발생하며, 선형 Diffusion Transformer를 위한 전통적인 스케일링 방법은 비효율적입니다. 이 연구는 높은 생성 품질을 유지하면서 선형 Diffusion Transformer의 연산 요구를 줄이는 효율적인 스케일링 기법 개발을 목표로 합니다.

---

### 2-2. 제안하는 방법 (수식 포함)

#### ① Efficient Training Scaling — Depth-Growth Paradigm

SANA를 1.6B(20블록)에서 4.8B(60블록)으로 스케일링하는 효율적인 모델 성장 전략을 제안합니다. 전통적인 스크래치 학습과 달리, 이 방법은 소형 모델의 사전 학습된 지식을 유지하면서 추가 블록을 전략적으로 초기화합니다.

구체적으로, 4.8B SANA-1.5 모델의 처음 18개 레이어를 1.6B SANA-1.0 사전 학습 모델로 초기화하는 **Partial Preservation Initialization 전략**을 활용합니다. 이 방식은 4.8B 모델이 스크래치 학습 대비 60% 훈련 시간을 절감하면서 우수한 GenEval 성능을 달성하도록 합니다.

**Partial Preservation Initialization** 전략을 수식으로 표현하면:

$$
\theta_{4.8B}^{(i)} =
\begin{cases}
\theta_{1.6B}^{(i)} & \text{if } i \leq M_{\text{pretrained}} \\
\mathbf{W}_{\text{identity}} & \text{if } i > M_{\text{pretrained}}
\end{cases}
$$

여기서 $\theta_{4.8B}^{(i)}$는 4.8B 모델의 $i$번째 블록 파라미터, $M_{\text{pretrained}}$는 사전학습된 블록 수, $\mathbf{W}_{\text{identity}}$는 identity mapping으로 초기화된 가중치를 의미합니다.

새로운 레이어의 초기화 전략 선택은 훈련 안정성과 효과적인 지식 전달에 매우 중요합니다. Partial Preservation Initialization — 기존 레이어를 보존하고 새로운 레이어를 identity mapping으로 랜덤 초기화하는 방식 — 이 가장 안정적이고 효과적임을 확인했습니다. 반면, Cyclic 및 Block Replication 전략은 훈련 불안정성(NaN 손실)을 야기하였습니다.

또한, 최초의 8-bit CAME 옵티마이저를 도입하여 GPU 메모리 사용량을 크게 줄이고, 대형 Diffusion 모델의 효율적인 스케일링을 가능하게 합니다.

**CAME-8bit 옵티마이저**의 핵심 아이디어:

CAME-8bit 옵티마이저는 1차 모멘트에는 블록 단위 8-bit 양자화를 적용하고, 2차 통계량은 32-bit 정밀도로 보존합니다.

수식으로 표현하면:

$$
m_t = \text{Quantize}_{8\text{-bit}}(\beta_1 m_{t-1} + (1-\beta_1) g_t)
$$
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad (\text{32-bit 유지})
$$
$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t} + \epsilon} \cdot \text{Dequantize}(m_t)
$$

여기서 $g_t$는 그래디언트, $\beta_1, \beta_2$는 모멘텀 계수, $\eta$는 학습률입니다.

---

#### ② Model Depth Pruning — Block Importance Analysis

블록 중요도를 Diffusion Transformer의 입·출력 유사도 패턴으로 분석하여 덜 중요한 블록을 제거하고, 파인튜닝을 통해 빠르게 모델 품질을 회복합니다(예: 단일 GPU에서 5분). 이 성장-후-가지치기 접근 방식은 60블록 모델을 다양한 구성(40/30/20블록)으로 효과적으로 압축하여 유연한 배포를 가능하게 합니다.

**블록 중요도 점수** 수식:

$$
S_l = 1 - \text{CosSim}\bigl(\mathbf{h}_l^{\text{in}},\; \mathbf{h}_l^{\text{out}}\bigr) = 1 - \frac{\mathbf{h}_l^{\text{in}} \cdot \mathbf{h}_l^{\text{out}}}{\|\mathbf{h}_l^{\text{in}}\| \|\mathbf{h}_l^{\text{out}}\|}
$$

여기서 $S_l$은 $l$번째 블록의 중요도 점수이며, $\mathbf{h}_l^{\text{in}}$, $\mathbf{h}_l^{\text{out}}$은 해당 블록의 입력·출력 특징 벡터입니다. **유사도가 낮을수록(변화가 클수록) 중요한 블록**으로 간주합니다.

블록 중요도는 head 블록과 tail 블록에서 높게 나타나며, head 블록은 latent 분포를 diffusion 분포로 변환하고 tail 블록은 역변환을 수행하는 것으로 추측됩니다. 중간 블록들은 입력·출력 특징 간 높은 유사도를 보이며 생성 결과의 점진적 정제를 담당합니다.

가지치기된 모델을 완전한 정보로 복원하는 것은 놀랍도록 쉽습니다. 단 100번의 파인튜닝 스텝으로 가지치기된 1.6B 모델이 전체 4.8B 모델과 비교할 수 있는 품질을 달성하며 SANA-1.0 1.6B 모델을 능가합니다.

---

#### ③ Inference-Time Scaling — Best-of-N Sampling with VLM Judge

SANA를 위한 추론 시간 스케일링 전략을 제안하며, 소형 모델이 파라미터 스케일링이 아닌 연산을 통해 대형 모델 품질을 달성할 수 있게 합니다. 다중 샘플 생성과 VLM 기반 선택 메커니즘을 통해 GenEval 점수를 0.81에서 0.96으로 향상시켰습니다. 이 향상은 LLM에서 관찰된 로그-선형 스케일링 패턴과 유사하며, 연산 자원을 모델 용량과 효과적으로 교환할 수 있음을 보여주어 "더 큰 모델이 항상 필요하다"는 통념에 도전합니다.

**Best-of-N Sampling** 수식:

$$
x^* = \arg\max_{x^{(i)},\; i=1,\ldots,N} \text{VLM-Score}\bigl(x^{(i)}, c\bigr)
$$

여기서 $x^{(i)} \sim p_\theta(\cdot | c)$는 텍스트 조건 $c$에 대해 독립적으로 샘플링된 이미지, $\text{VLM-Score}(\cdot)$는 VLM Judge(NVILA-2B 등)가 이미지와 텍스트의 정합성을 평가하는 점수 함수입니다.

SANA와 많은 Diffusion 모델에서 추론 시간 연산을 늘리는 자연스러운 방법은 Denoising 스텝 수를 증가시키는 것입니다. 그러나 더 많은 Denoising 스텝은 두 가지 이유로 스케일링에 이상적이지 않습니다. 첫째, 추가 Denoising 스텝이 오류를 자체 수정할 수 없습니다. 초기 단계에서 잘못 배치된 객체는 이후 스텝에서도 변하지 않습니다. 둘째, 생성 품질이 빠르게 정체됩니다.

반면, 샘플링 후보의 수를 늘리는 것은 더 유망한 방향입니다. 소형 모델 SANA(1.6B)도 어려운 테스트 프롬프트에 대해 여러 번의 시도가 주어지면 올바른 결과를 생성할 수 있습니다 — 마치 부주의한 학생이 요청대로 그릴 수는 있지만 실행 중 실수를 저지르는 것처럼.

---

### 2-3. 모델 구조

SANA-1.5의 전체 구조는 다음 네 가지 핵심 설계 위에 구축됩니다:

**① Deep Compression Autoencoder (DC-AE)**

새로운 Deep Compression Autoencoder(DC-AE)는 스케일링 인수를 32로 적극적으로 높입니다. AE-F8 대비 AE-F32는 latent 토큰 수를 16배 줄여, 효율적인 훈련과 4K와 같은 초고해상도 이미지 생성에 필수적입니다.

**② Efficient Linear DiT**

새로운 선형 DiT를 도입하여 기존 Vanilla Quadratic Attention을 대체하고, 복잡도를 $O(N^2)$에서 $O(N)$으로 줄입니다. Mix-FFN은 MLP 내 $3\times3$ depth-wise convolution으로 토큰의 로컬 정보를 강화합니다.

선형 어텐션의 핵심 수식:

$$
\text{Attention}_{\text{vanilla}}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}\right)\mathbf{V} \quad [O(N^2)]
$$

$$
\text{Attention}_{\text{linear}}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \phi(\mathbf{Q})\bigl(\phi(\mathbf{K})^\top \mathbf{V}\bigr) \quad [O(N)]
$$

여기서 $\phi(\cdot)$는 커널 함수(예: ELU+1 또는 ReLU)로 softmax를 근사합니다.

**③ Decoder-only Text Encoder**

T5를 현대적인 decoder-only 소형 LLM으로 교체하고, in-context learning을 위한 복잡한 인간 명령어를 설계하여 이미지-텍스트 정합성을 향상시켰습니다.

**④ Flow-DPM-Solver**

Flow-DPM-Solver는 Flow-Euler-Solver 대비 추론 스텝을 28-50에서 14-20으로 줄이면서 더 나은 성능을 달성합니다.

**전체 파이프라인 구조:**

$$
\underbrace{x_{\text{text}} \xrightarrow{\text{Decoder-LLM}} c}_{\text{텍스트 인코딩}} \rightarrow \underbrace{z_T \xrightarrow{\text{Flow ODE (Linear DiT)}} z_0}_{\text{Denoising}} \rightarrow \underbrace{z_0 \xrightarrow{\text{DC-AE Decoder}} x_{\text{img}}}_{\text{이미지 디코딩}}
$$

---

### 2-4. 성능 향상

SANA-4.8B 모델은 소형 모델 대비 실질적인 개선을 보여줍니다. 구체적으로 SANA-1.6B에서 SANA-4.8B로의 스케일링은 GenEval에서 0.06 절대 이득(0.66→0.72), FID 0.34 감소(5.76→5.42), DPG 점수 0.2 향상(84.8→85.0)을 달성합니다.

이 전략들을 통해 SANA-1.5는 GenEval에서 0.81의 텍스트-이미지 정합 점수를 달성하며, VILA-Judge를 통한 추론 스케일링으로 0.96까지 향상되어 GenEval 벤치마크에서 새로운 SoTA를 수립합니다.

모델 성장 전략과 고품질 데이터에 대한 파인튜닝의 조합은 SANA-4.8B에서 GenEval 점수 0.81을 달성합니다. 이는 성장 접근법이 효율성을 제공할 뿐만 아니라, 훨씬 적은 파라미터와 낮은 지연 시간으로 Playground v3(24B)과 같은 모델을 능가하는 SoTA 결과의 강력한 기반을 마련함을 보여줍니다.

| 모델 | GenEval | FID | DPG Score |
|---|---|---|---|
| SANA-1.0 1.6B | 0.66 | 5.76 | 84.8 |
| SANA-1.5 4.8B (pre-trained) | 0.72 | 5.42 | 85.0 |
| SANA-1.5 4.8B (fine-tuned) | 0.81 | — | — |
| SANA-1.5 + Inference Scaling | **0.96** | — | — |

---

### 2-5. 한계

이 연구는 극도로 고해상도 이미지 생성에 있어 몇 가지 한계가 있습니다. 효율적이지만, 선형 어텐션 방식은 Quadratic Attention이 포착할 수 있는 세밀한 디테일을 놓칠 수 있습니다. 논문은 고도로 상세한 텍스처나 복잡한 기하학적 패턴과 같은 특수한 엣지 케이스에서의 성능 저하를 충분히 다루지 않습니다. 다양한 이미지 유형에 대한 모델 동작에 대한 추가 연구가 필요합니다.

추가적으로:
- Denoising 스텝 수 증가는 아티팩트를 자체 수정하지 못하는 경미한 개선만 보이므로 스케일링의 좋은 선택지가 아닙니다. 반면, 샘플링 노이즈를 스케일링하는 것이 더 효과적이며, VLM 전문가가 프롬프트와 일치하는 이미지를 검증·선택하는 데 도움을 줍니다.
- Best-of-N Sampling은 N이 매우 클 경우 추론 비용이 선형적으로 증가하여 실시간 응용에는 제약이 있을 수 있습니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. Grow-then-Prune 패러다임에 의한 일반화

세 가지 기술적 기여 — 모델 성장, 모델 깊이 가지치기, 추론 스케일링 — 은 효율적인 모델 스케일링을 위한 일관된 프레임워크를 형성합니다. 모델 성장 전략은 더 큰 최적화 공간을 탐색하여 더 나은 특징 표현을 발견합니다. 그런 다음 모델 깊이 가지치기가 이러한 필수 특징들을 식별하고 보존하여 효율적인 배포를 가능하게 합니다.

이는 **과학적 관점에서의 일반화** 향상으로 이어집니다:
- 대형 모델의 더 넓은 특징 공간을 학습한 후 가지치기하므로, 단순히 소형 모델을 처음부터 학습하는 것보다 일반화 능력이 향상됩니다.

### 3-2. VLM 기반 추론 스케일링과 일반화

소형 모델도 대형 모델을 능가할 수 있으며, 모든 모델 크기에서 일관된 성능 향상이 관찰됩니다.

이러한 혁신들은 다양한 연산 예산에서 효율적인 모델 스케일링을 가능하게 하는 동시에 높은 품질을 유지하여, 고품질 이미지 생성을 더 폭넓게 접근 가능하게 합니다.

VLM Judge가 여러 생성물 중 최적을 선택하는 구조는 다음의 일반화 효과를 가져옵니다:

$$
P(\text{정확한 생성}) = 1 - \prod_{i=1}^{N}(1 - p_i)
$$

여기서 $p_i$는 $i$번째 샘플이 프롬프트와 정합하는 확률이며, $N$이 증가할수록 성공 확률이 지수적으로 높아집니다.

다양한 모델 크기와 구성에서도 경쟁력 있는 성능을 보장하는 다재다능한 스케일링·가지치기 전략은 SANA-1.5가 더 큰 모델이 더 나은 성능을 보장한다는 통념에 효과적으로 도전하며, 최적화된 훈련 프로세스와 적응적 추론 전략의 중요성을 강조함을 나타냅니다.

### 3-3. 다중 도메인 일반화 가능성

선형 Diffusion 컴포넌트는 훈련 중 파라미터 업데이트를 더 효율적으로 수행하면서 복잡한 이미지 특징을 포착하는 모델의 능력을 보존합니다.

이는 다음을 의미합니다:
- **Grow-then-Prune** 전략은 비전 생성 외 비디오, 3D, 오디오 등 다양한 도메인에도 적용 가능한 범용 스케일링 패러다임으로 확장될 잠재력이 있습니다.
- 실제로 SANA-Video 720p 모델이 이미 출시되어 이 프레임워크의 비디오 도메인 확장이 진행 중입니다.

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 4-1. 향후 연구에 미치는 영향

#### (A) LLM 스케일링 법칙의 이미지 생성 영역 전이

LLM에서의 추론 시간 스케일링의 최근 성공에 영감을 받아, 추론 시간 스케일링이 생성 상한을 밀어붙이는 데 효과적임을 입증했습니다. SANA와 많은 Diffusion 모델에서 추론 시간 연산을 늘리는 자연스러운 방법 중 하나는 Denoising 스텝 수를 증가시키는 것입니다.

LLM에서 시작된 **Test-Time Compute Scaling** 패러다임이 이미지 생성으로 성공적으로 전이되었음을 보여주며, 향후 다양한 생성 모델(비디오, 3D, 오디오)에 대한 유사한 연구를 촉진할 것입니다.

#### (B) 효율적 AI 접근성 민주화

이 연구는 고품질 이미지 합성이 항상 대규모 연산 자원을 필요로 하지 않음을 입증합니다. 이는 AI 이미지 생성 기술의 민주화에 광범위한 영향을 미칠 수 있으며, 다양한 AI 응용에서의 모델 스케일링 접근 방식에 영향을 미칠 수 있는 유망한 방향을 제시합니다.

#### (C) 모델 압축 연구 패러다임 전환

SANA-1.5는 더 큰 모델이 더 나은 성능을 보장한다는 통념에 도전하며, 최적화된 훈련 프로세스와 적응적 추론 전략의 중요성을 강조합니다. 이 연구는 모델 스케일링에 대한 이론적 통찰을 제공할 뿐만 아니라 고품질 이미지 생성 기술에 대한 폭넓은 접근을 위한 실용적인 경로를 만들어 냅니다.

---

### 4-2. 향후 연구 시 고려할 점

#### ① 추론 비용과 지연 시간의 균형

Best-of-N Sampling은 강력하지만 N이 클수록 추론 비용이 증가합니다. 향후 연구는 다음을 고려해야 합니다:

$$
\text{Optimal } N^* = \arg\min_{N} \mathcal{C}(N) \quad \text{s.t.} \quad \text{Quality}(N) \geq \tau
$$

여기서 $\mathcal{C}(N)$은 N개 샘플 생성 비용, $\tau$는 목표 품질 임계값입니다. 적응적 조기 종료(Adaptive Early Stopping) 전략이 효율성을 높일 수 있습니다.

#### ② VLM Judge의 편향성과 신뢰도

다양한 이미지 유형에 대한 모델 동작에 대한 추가 연구가 필요합니다. 향후 연구는 선형 및 전통적 어텐션 메커니즘을 결합한 하이브리드 접근법을 탐구하여 세부 보존이 중요한 특정 태스크에 대응할 수 있습니다.

VLM Judge 자체의 평가 편향, 특히 문화적·미적 편향이 선택된 이미지의 다양성을 제한할 수 있음을 고려해야 합니다.

#### ③ Block Importance 메트릭의 시간적 변화

현재의 Block Importance는 정적 코사인 유사도로 측정되나, Denoising 타임스텝 $t$에 따라 블록 중요도가 달라질 수 있습니다:

$$
S_l(t) = 1 - \text{CosSim}\bigl(\mathbf{h}_l^{\text{in}}(t),\; \mathbf{h}_l^{\text{out}}(t)\bigr)
$$

타임스텝에 따른 동적 가지치기(Dynamic Pruning per timestep)가 추가적인 성능 향상을 가져올 수 있습니다.

#### ④ 다른 도메인으로의 확장 가능성

고품질 이미지 생성에는 일반적으로 막대한 연산력이 필요하여 첨단 AI 이미지 생성을 많은 연구자와 개발자들의 손이 닿지 않는 곳에 두었습니다. SANA-1.5는 보다 효율적이면서 최상위 이미지 생성을 달성하는 더 스마트한 AI 시스템을 개발하였습니다. 이 패러다임은 비디오, 3D 장면 생성, 오디오 합성 등으로의 확장을 연구할 가치가 있습니다.

#### ⑤ Reinforcement Learning과의 결합

SANA는 이미 Cosmos-RL과 파트너십을 맺어 완전한 RL 인프라를 제공하고 있습니다. Diffusion-NFT, Flow-GRPO와 같은 최신 알고리즘으로 SANA-Image 및 SANA-Video의 사후 학습(SFT/RL)이 가능합니다. VLM Judge를 보상 모델로 활용하는 RLHF/RLVF 프레임워크와의 결합은 일반화 성능을 추가로 향상시킬 수 있습니다.

---

## 2020년 이후 관련 최신 연구 비교 분석

| 모델/방법 | 연도 | 파라미터 | 어텐션 복잡도 | 핵심 기여 | GenEval |
|---|---|---|---|---|---|
| **DDPM** (Ho et al.) | 2020 | ~100M | $O(N^2)$ | Score-matching 기반 Denoising | — |
| **DiT** (Peebles & Xie) | 2022 | 0.6B | $O(N^2)$ | Transformer 기반 Diffusion | — |
| **PixArt-α** | 2023 | 0.6B | $O(N^2)$ | 효율적 고품질 T2I | 0.48 |
| **FLUX.1** | 2024 | 12B | $O(N^2)$ | Flow Matching + MMDiT | ~0.66 |
| **SANA-1.0** | 2024 | 1.6B | $O(N)$ | Linear DiT + DC-AE | 0.66 |
| **Playground v3** | 2024 | 24B | $O(N^2)$ | 대규모 모델 | ~0.70 |
| **SANA-1.5** | 2025 | 1.6B~4.8B | $O(N)$ | Grow+Prune+InfScale | **0.96** |

SANA는 4096×4096 해상도까지 효율적으로 이미지를 생성할 수 있는 텍스트-이미지 프레임워크이며, 현대의 대형 Diffusion 모델(예: Flux-12B)과 매우 경쟁력이 있으면서도 20배 작고 처리량은 100배 이상 빠릅니다.

---

## 참고 문헌 및 출처

1. **arXiv (주논문)**: Xie, E. et al., "SANA 1.5: Efficient Scaling of Training-Time and Inference-Time Compute in Linear Diffusion Transformer," arXiv:2501.18427, 2025. https://arxiv.org/abs/2501.18427

2. **NVIDIA Research Page**: https://research.nvidia.com/labs/eai/publication/sana-1.5/

3. **공식 프로젝트 페이지 (NVLabs)**: https://nvlabs.github.io/Sana/Sana-1.5/

4. **GitHub (NVLabs/Sana)**: https://github.com/NVlabs/Sana

5. **MIT HAN Lab (SANA-1.0 배경)**: https://hanlab.mit.edu/projects/sana

6. **OpenReview (ICML-2025)**: https://openreview.net/forum?id=27hOkXzy9e

7. **Hugging Face Paper Page**: https://huggingface.co/papers/2501.18427

8. **Semantic Scholar**: https://www.semanticscholar.org/paper/604a6b7ebfc21db8b4860ab0054d17548d2f48c8

9. **Moonlight Literature Review**: https://www.themoonlight.io/en/review/sana-15-efficient-scaling-of-training-time-and-inference-time-compute-in-linear-diffusion-transformer

10. **Liner Quick Review**: https://liner.com/review/sana-15-efficient-scaling-trainingtime-and-inferencetime-compute-in-linear

> ⚠️ **정확도 주의사항**: 논문 내 세부 수식(Block Importance Score, 옵티마이저의 정확한 수식 등)은 공개된 HTML 버전(arxiv.org/html/2501.18427)과 공식 프로젝트 페이지에서 확인된 내용을 기반으로 작성되었으며, 일부 수식은 논문의 설명을 형식화한 것입니다. 완전한 수식 확인을 위해 [논문 PDF](https://arxiv.org/pdf/2501.18427)를 직접 참조하실 것을 권장합니다.

# SANA 1.5: Efficient Scaling of Training-Time and Inference-Time Compute in Linear Diffusion Transformer

### 핵심 요약

SANA 1.5는 텍스트-이미지 생성 분야에서 근본적인 스케일링 패러다임 전환을 제시합니다. SANA-1.0을 기반으로 하는 이 연구는 단순히 모델 크기를 증가시키는 전통적 접근에서 벗어나, **더 나은 최적화 궤적**을 통해 효율적인 스케일링을 달성합니다. 3가지 핵심 혁신—훈련 시간 스케일링, 깊이 기반 가지치기, 추론 시간 스케일링—을 통합함으로써, 연구팀은 4.8B 파라미터 모델로 24B 파라미터 모델과 비교 가능한 또는 더 우수한 성능을 달성하면서 훈련 비용을 60% 감축했습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0fc575fd-e6f1-4416-9aff-8f7495cea32b/2501.18427v4.pdf)

### 해결하는 문제

텍스트-이미지 생성 모델의 급속한 규모 확대(PixArts 0.6B → Playground v3 24B)는 심각한 계산 병목을 야기합니다. 기존 접근법들은 다음의 한계를 가집니다:

1. **훈련 효율성**: 대규모 모델을 처음부터 학습하려면 엄청난 계산 자원 필요
2. **메모리 제약**: 최적화기(예: AdamW)의 메모리 오버헤드로 인해 고급 GPU 필요
3. **배포 유연성**: 고정된 모델 크기만 지원, 다양한 하드웨어 환경에 부적합
4. **품질-계산 트레이드오프**: 모델 크기 증가가 유일한 성능 향상 방법

이러한 문제들은 고품질 이미지 생성을 연구 기관과 상업적 주체에게만 접근 가능하게 만듭니다.

### 제안하는 방법 및 수식

#### 1. Efficient Model Growth (효율적 모델 성장)

**Partial Preservation Initialization (부분 보존 초기화):**

SANA-1.5는 N개 블록의 사전학습 모델을 N+M 블록으로 확장합니다 (N=20, M=40). 핵심 초기화 전략:

$$
R_\ell = \begin{cases} 
R_{\text{pre}}(\ell) & \text{if } \ell < N-2 \\
\mathcal{N}(0, \sigma^2) & \text{if } \ell \geq N
\end{cases}
$$

여기서 새로운 블록의 출력 프로젝션은 항등함수로 작동하도록:

$$
W_{\text{out}}^{\text{(new)}} \leftarrow 0
$$

이는 새로운 블록이 초기에 항등 변환을 수행하도록 하여 **안정적인 최적화 경로**를 보장합니다. 마지막 2개의 사전학습된 블록을 제거하는 이유는 블록 중요도 분석(Figure 5)에서 태스크 관련도가 높은 블록들이 잘 학습된 특성을 방해하기 때문입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0fc575fd-e6f1-4416-9aff-8f7495cea32b/2501.18427v4.pdf)

**메모리 효율적 CAME-8bit 최적화기:**

$$
\text{saved memory} = \sum_{\ell \in \Omega} 24 \cdot n_\ell \text{ bytes}
$$

여기서 $\Omega$는 정량화된 계층, $n_\ell$은 파라미터 수입니다. 블록 단위 정량화 함수:

$$
\tilde{v} = \text{round}\left(\frac{v - v_{\min}}{v_{\max} - v_{\min}} \times 255\right)
$$

**결과**: AdamW-32bit 대비 메모리 사용 8배 감소 (43GB → 36GB 효과적 감소), 25% 추가 절감 달성. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0fc575fd-e6f1-4416-9aff-8f7495cea32b/2501.18427v4.pdf)

#### 2. Model Depth Pruning (깊이 기반 모델 압축)

**블록 중요도 분석 (Block Importance Analysis):**

$$
BI_\ell = 1 - E_{X_{t,\ell}, X_{t,\ell+1}} \left[ \text{similarity}(X_{t,\ell}, X_{t,\ell+1}) \right]
$$

이 메트릭은 각 블록의 입출력 특징 유사성을 측정하여 정보 변환량을 정량화합니다. 확산 시간단계와 보정 데이터셋(100개 다양한 프롬프트)에 걸쳐 평균화됩니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0fc575fd-e6f1-4416-9aff-8f7495cea32b/2501.18427v4.pdf)

**주요 발견:**
- **헤드 블록** (1-5): 높은 중요도 (0.7-0.8) - 잠상(latent)을 확산 분포로 변환
- **테일 블록** (55-60): 높은 중요도 (0.6-0.7) - 확산 분포를 원래 이미지로 역변환
- **중간 블록** (20-50): 낮은 중요도 (0.3-0.5) - 점진적 세부사항 정제

**가지치기 후 미세조정:**

100 훈련 스텝(단일 GPU)으로 60블록 → 40/30/20블록 압축 가능, GenEval 성능 유지: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0fc575fd-e6f1-4416-9aff-8f7495cea32b/2501.18427v4.pdf)
- 60블록 → 40블록: 0.693 → 0.684
- 60블록 → 30블록: 0.693 → 0.675  
- 60블록 → 20블록: 0.693 → 0.672 (SANA-1.0 1.6B의 0.665 초과)

#### 3. Inference-Time Scaling (추론 시간 스케일링)

**Denoising Steps vs. Repeated Sampling:**

저자들은 **반복 샘플링이 Denoising 스텝 증가보다 우월**함을 입증합니다. 이유: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0fc575fd-e6f1-4416-9aff-8f7495cea32b/2501.18427v4.pdf)

1. Denoising 초기 오류는 자동 수정 불가능 (Figure 3a 참조)
2. 품질 개선이 빠르게 포화 (20 스텝에서 최적화)

따라서 여러 샘플 생성 후 VLM 기반 검증자로 최적 이미지 선택:

$$
\text{score} = P_{\text{VILA}}(\text{"match"} | \text{image, prompt})
$$

**VILA-Judge 설계:**

2M 프롬프트-이미지 데이터셋으로 미세조정된 NVILA-2B 모델. 토너먼트 형식 비교:

- 두 이미지 쌍을 반복 비교
- VILA 응답이 "yes/no"로 갈라지면 "yes" 이미지 선택
- 동일 응답이면 로그프롭(log probability) 기반 선택

**성능 개선:**

$$
\text{GenEval Score} = 0.81 \text{ (단일)} \to 0.96 \text{ (2048샘플)}
$$

세부 성능 향상: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0fc575fd-e6f1-4416-9aff-8f7495cea32b/2501.18427v4.pdf)
- 위치(Position): 0.59 → 0.96
- 색상 속성(Color Attribution): 0.65 → 0.87
- 계산(Counting): 0.86 → 0.97

### 모델 구조

**Linear Diffusion Transformer 아키텍처:**

선형 자기-어텐션과 바닐라 크로스-어텐션 결합. 훈련 안정성을 위해 **RMSNorm** 적용:

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \gamma
$$

쿼리와 키에 RMSNorm 적용하여 선형 어텐션의 로짓 폭발 방지 (FP16에서 $\geq 65504$ 오버플로우 회피). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0fc575fd-e6f1-4416-9aff-8f7495cea32b/2501.18427v4.pdf)

**하이브리드 정밀도 CAME-8bit:**

- **1차 모멘트**: 블록 단위 8비트 정량화
- **2차 모멘트**: 32비트 유지 (행렬 인수분해로 이미 메모리 효율적)

이는 최적화 안정성 보존과 메모리 절감 간 균형을 달성합니다.

### 성능 향상 및 일반화

#### 정량적 성능 개선

| 메트릭 | SANA-1.0 1.6B | SANA-1.5 4.8B (Pre) | SANA-1.5 4.8B (Ours) | 추론 스케일 |
|--------|---------------|-------------------|----------------------|-----------|
| GenEval | 0.66 | 0.72 | 0.81 | **0.96** |
| FID ↓ | 5.76 | 5.42 | 5.99 | - |
| CLIP | 28.67 | 29.16 | 29.23 | - |
| 처리량 (img/s) | 1.0 | 0.26 | 0.26 | - |
| 레이턴시 (s) | 1.2 | 4.2 | 4.2 | - |

**Playground v3 (24B, GenEval 0.76) 대비:** SANA-1.5는 5.5배 낮은 레이턴시, 6.5배 높은 처리량, 0.05 높은 GenEval 스코어. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0fc575fd-e6f1-4416-9aff-8f7495cea32b/2501.18427v4.pdf)

#### 일반화 성능 향상의 메커니즘

**1. 더 나은 최적화 궤적 발견:**

모델 성장 전략이 더 큰 최적화 공간을 탐색하여 더 좋은 특징 표현을 발견합니다. 작은 모델의 사전학습된 특징이 새로운 블록의 학습을 위한 견고한 기반(foundation)이 됩니다.

**2. 지식 전이의 효과:**

무작위 초기화 대비 부분 보존 초기화 사용 시:
- **수렴 시간 60% 단축** (같은 성능 도달)
- **훈련 안정성 향상** (Cyclic/Block Replication 시 NaN 손실 발생 방지)

**3. 블록 중요도 기반 지능형 설계:**

블록 중요도 분석이 두 방향으로 활용:
- **성장**: 새 블록을 마지막에 추가할 때 테일 블록(높은 중요도) 2개 제거
- **가지치기**: 중간 블록(낮은 중요도)를 안전하게 제거 후 미세조정

**4. 다양한 평가 메트릭에서 일관된 개선:**

고품질 데이터(GenEval 스타일 144K 프롬프트) 미세조정 후:

$$
\text{GenEval}_{v1} = 0.72 \to \text{GenEval}_{v2} = 0.81 
$$

(3% 개선)

**5. 다중언어 지원으로 일반화 확대:**

100K 영어 프롬프트를 GPT-4로 4가지 형식으로 확장:
- 순수 중국어
- 영어-중국어 혼합
- 이모지 포함

10K 미세조정 후 **다국어 및 이모지 프롬프트에서 안정적 출력** 달성. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0fc575fd-e6f1-4416-9aff-8f7495cea32b/2501.18427v4.pdf)

### 최신 연구와의 비교 분석 (2020년 이후)

#### 1. 스케일링 법칙 연구

**Scaling Laws for Diffusion Transformers (Liang et al., 2024):** [arxiv](https://arxiv.org/abs/2502.12048)
- 훈련 손실이 계산 예산과 멱법칙 관계 준수 발견
- 최적 모델/데이터 크기 예측 프레임워크 제시
- **SANA 1.5의 위치**: 실무적 검증과 추가 최적화 (QK normalization, CAME-8bit)

**Towards Precise Scaling Laws for Video Diffusion Transformers (Yin et al., 2024):** [semanticscholar](https://www.semanticscholar.org/paper/1a65219f0d3852b55d1fadf58e1ca75c1090805e)
- 비디오 생성에서 하이퍼파라미터 민감도 강조
- 학습률, 배치 크기 정확한 모델링 필요
- **SANA 1.5의 차별성**: 이미지 생성의 블록 구조적 스케일링에 집중

#### 2. 모델 압축 및 효율화

**Minitron: LLM Pruning and Distillation in Practice (Sreenivas et al., 2024):** [jst.tnu.edu](https://jst.tnu.edu.vn/jst/article/view/13790)
- LLM의 구조화된 가지치기 (블록 제거)
- 추론 후(post-training) 압축 가능성 입증
- **SANA 1.5의 혁신**: Diffusion Transformer에 처음 적용, 입출력 유사성 기반 중요도 계산

**DiT-MoE: Scaling Diffusion Transformers to 16 Billion Parameters (Fei et al., 2024):** [arxiv](https://arxiv.org/abs/2511.05535)
- Mixture-of-Experts를 통한 희소 확장 (16.5B)
- 전문가 선택이 공간 위치와 시간 스텝에 의존
- **비교**: SANA는 **밀집 모델로 더 효율적 접근**, 추론 시간 스케일링으로 보상

#### 3. 추론 시간 스케일링

**Large Language Monkeys: Scaling Inference with Repeated Sampling (Brown et al., 2024):** [academic.oup](https://academic.oup.com/bib/article/26/Supplement_1/i44/8378055)
- LLM에서 샘플 반복을 통해 추론 계산으로 능력 확장 가능 증명
- 로그-선형 스케일링 패턴 발견
- **SANA 1.5의 기여**: 이미지 생성에 **처음 적용**, VLM 판정자로 검증 강화

**Inference-Time Scaling for Diffusion Models Beyond Scaling Denoising Steps (Ma et al., 2025):** [academic.oup](https://academic.oup.com/bib/article/26/Supplement_1/i24/8378044)
- Feynman-Kac 프레임워크로 입자 기반 샘플링 제시
- 보상 함수 기반 유연한 조종(steering) 가능
- **SANA 1.5와의 관계**: 동시 발전 연구로, SANA가 실용적 VILA-Judge 활용

#### 4. 텍스트-이미지 정렬 평가 및 벤치마크

**GenEval: An Object-Focused Framework (Ghosh et al., 2023):** [e-journal.unair.ac](https://e-journal.unair.ac.id/JESTT/article/view/47782)
- 객체 중심 평가 (공동 발생, 위치, 개수, 색상)
- 83% 인간-어그리먼트 달성
- **SANA 1.5의 중요성**: GenEval에서 **0.96 새로운 최고 성능**

**GenEval 2: Addressing Benchmark Drift (Kamath et al., 2025):** [arxiv](https://arxiv.org/pdf/2401.10061.pdf)
- GenEval의 **포화 문제** 지적 (벤치마크 드리프트 17.7% 오차)
- 구성성 강조 (3-10 "atoms" per prompt)
- Soft-TIFA 메트릭 제안 (AUROC 94.5% 인간 정렬)
- **시사점**: SANA 1.5의 0.81 → 0.96 개선은 **진정한 능력 향상** 증명

**On the Scalability of Diffusion-based Text-to-Image Generation (Li et al., 2024):** [arxiv](http://arxiv.org/pdf/2312.04557.pdf)
- 0.4B~4B 범위에서 크로스 어텐션과 블록 수의 영향 분석
- 텍스트 정렬에는 **블록 깊이가 채널 수보다 효과적**
- **SANA 1.5의 검증**: 블록 중요도 기반 가지치기로 이론 실증

#### 5. 다른 효율적 생성 모델들과의 비교

| 모델 | 파라미터 | 처리량 (img/s) | 레이턴시 (s) | GenEval |
|-----|---------|--------------|-----------|---------|
| FLUX-dev | 12.0B | 0.04 | 23.0 | 0.67 |
| Playground v3 | 24.0B | 0.06 | 15.0 | 0.76 |
| SANA-1.5 4.8B | 4.8B | **0.26** | **4.2** | **0.81** |
| SANA-1.5 + Inference | 4.8B | 0.001 | - | **0.96** |

SANA-1.5는 **파라미터 효율성 (24B 대비 1/5), 속도 (5.5배), 품질 (GenEval 0.81)**에서 우수합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0fc575fd-e6f1-4416-9aff-8f7495cea32b/2501.18427v4.pdf)

### 일반화 성능의 한계 및 미래 연구 방향

#### 현재 한계

1. **텍스트 렌더링**: 복잡한 장면에서 정확한 텍스트 생성 여전히 미흡
2. **추론 시간 스케일링 계산 비용**: 2048샘플 생성 + VILA-Judge = 149 GFLOPs (모델 1회 인퍼런스의 ~30배)
3. **공간 관계 이해**: 다중 객체 간의 3D 공간 관계에서 약점
4. **색상 정확도**: 특정 RGB 값의 미세한 색상 제어 제한

#### 앞으로의 연구 고려사항

**1. 추론 스케일링 효율화:**
- 더 가벼운 검증 모델 개발 (VILA-Judge보다 작은 모델)
- 캐싱 및 조기 종료 메커니즘 도입
- 적응형 샘플 수 결정 (쉬운 프롬프트는 적게, 어려운 프롬프트는 많게)

**2. 일반화 성능 확대:**
- **도메인 확장**: 의료 영상, 3D 생성으로의 이전
- **제로샷 능력**: 새로운 스타일/개념에 대한 일반화
- **적대적 입력**: 모자이크된 또는 노이즈 있는 프롬프트 강건성

**3. 멀티모달 일반화:**
- 비디오 생성으로 자연스러운 확장 (이미 SANA-Video 연구 진행 중)
- 이미지-텍스트 쌍방향 생성
- 조건부 생성의 정밀 제어

**4. 블록 선택의 동적화:**
- 입력 프롬프트의 복잡도에 따라 **동적으로 블록 활성화** 조정
- 조기 종료(early exit) 메커니즘으로 불필요한 계산 회피
- 각 프롬프트에 최적화된 블록 조합 학습

**5. 최적화 궤적 이론:**
- 왜 부분 보존 초기화가 더 나은 솔루션을 발견하는가에 대한 이론적 분석
- 로스 랜드스케이프(loss landscape) 시각화
- 신경 탱젠트 커널(NTK) 이론과의 연결

**6. 안전성 강화:**
- NSFW 내용 필터링 (현재 ShieldGemma-2B 적용)
- 편향 평가 및 공정성 분석
- 저작권 보호된 콘텐츠 생성 방지

#### 산업적 임플리케이션

SANA 1.5의 효율성 개선은 다음을 가능하게 합니다:

- **엣지 디바이스 배포**: RTX 4090에서 4.8B 모델 미세조정 가능
- **기업 맞춤화**: 리소스 제약 환경에서의 도메인 특화 모델 개발
- **민주화**: 오픈소스 커뮤니티의 고품질 모델 접근성 대폭 향상

### 결론

SANA 1.5는 텍스트-이미지 생성의 스케일링에 대한 근본적 재고를 제시합니다. **"더 큰 모델이 항상 더 낫다"는 기존 믿음에 도전**하며, 지능적 설계(깊이 성장, 블록 가지치기, 추론 시간 스케일링)를 통해 더 작은 모델이 더 큰 모델을 능가할 수 있음을 입증합니다. 

세 가지 기술의 조화로운 통합으로, SANA 1.5는:
- **60% 훈련 시간 단축**
- **1.6배 성능 향상** (GenEval 0.66 → 0.81)
- **5.5배 속도 향상** (FLUX 대비)
- **2배 처리량 증가** 

을 동시에 달성했습니다. 이러한 효율성 개선이 고품질 이미지 생성을 더 많은 연구자와 실무자에게 접근 가능하게 함으로써, 생성 AI의 민주화에 기여합니다.

***

### 참고 문헌

<span style="display:none">[^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86][^1_87][^1_88][^1_89][^1_90][^1_91][^1_92][^1_93][^1_94][^1_95][^1_96][^1_97][^1_98]</span>

<div align="center">⁂</div>

[^1_1]: 2501.18427v4.pdf

[^1_2]: https://arxiv.org/abs/2502.12048

[^1_3]: https://www.semanticscholar.org/paper/1a65219f0d3852b55d1fadf58e1ca75c1090805e

[^1_4]: https://jst.tnu.edu.vn/jst/article/view/13790

[^1_5]: https://arxiv.org/abs/2511.05535

[^1_6]: https://academic.oup.com/bib/article/26/Supplement_1/i44/8378055

[^1_7]: https://academic.oup.com/bib/article/26/Supplement_1/i24/8378044

[^1_8]: https://e-journal.unair.ac.id/JESTT/article/view/47782

[^1_9]: https://arxiv.org/pdf/2401.10061.pdf

[^1_10]: http://arxiv.org/pdf/2312.04557.pdf

[^1_11]: https://www.sciltp.com/journals/hm/articles/2504000541

[^1_12]: https://www.mdpi.com/2076-3417/15/20/11150

[^1_13]: https://jisem-journal.com/index.php/journal/article/view/6615

[^1_14]: https://www.frontiersin.org/articles/10.3389/fenvs.2025.1659344/full

[^1_15]: https://ieeexplore.ieee.org/document/11147513/

[^1_16]: https://pubs.aip.org/pof/article/37/11/117119/3371491/Fine-structure-investigation-of-turbulence-induced

[^1_17]: https://iopscience.iop.org/article/10.1149/MA2025-031244mtgabs

[^1_18]: https://iopscience.iop.org/article/10.1149/MA2025-02121107mtgabs

[^1_19]: https://pubs.aip.org/pof/article/37/11/117120/3371493/Fine-structure-investigation-of-turbulence-induced

[^1_20]: https://biss.pensoft.net/article/181733/

[^1_21]: https://arxiv.org/html/2410.02098

[^1_22]: https://arxiv.org/html/2404.09976

[^1_23]: https://arxiv.org/html/2501.18427v3

[^1_24]: https://arxiv.org/html/2407.11633v1

[^1_25]: http://arxiv.org/pdf/2404.02883.pdf

[^1_26]: http://arxiv.org/pdf/2212.09748v2.pdf

[^1_27]: https://arxiv.org/abs/2410.13925v1

[^1_28]: https://arxiv.org/abs/2301.09474

[^1_29]: https://openreview.net/pdf?id=iIGNrDwDuP

[^1_30]: https://liner.com/review/scaling-laws-synthetic-images-for-model-training-for-now

[^1_31]: https://liner.com/review/inferencetime-scaling-for-diffusion-models-beyond-scaling-denoising-steps

[^1_32]: https://neurips.cc/virtual/2025/poster/117664

[^1_33]: https://proceedings.iclr.cc/paper_files/paper/2025/file/f8e7248f3e659cfe70c6debcdae1b023-Paper-Conference.pdf

[^1_34]: https://arxiv.org/abs/2501.09732

[^1_35]: https://kimjy99.github.io/논문리뷰/stable-diffusion-3/

[^1_36]: https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Scaling_Down_Text_Encoders_of_Text-to-Image_Diffusion_Models_CVPR_2025_paper.pdf

[^1_37]: https://velog.io/@jojo0217/논문리뷰-Inference-Time-Scaling-for-Diffusion-Modelsbeyond-Scaling-Denoising-Steps

[^1_38]: https://velog.io/@guts4/Scaling-Rectified-Flow-Transformers-for-High-Resolution-Image-SynthesisStableDiffusion3-2024-arXiv

[^1_39]: https://arxiv.org/html/2410.13863v1

[^1_40]: https://arxiv.org/abs/2507.08390

[^1_41]: https://openaccess.thecvf.com/content/ICCV2025/papers/Hou_Dita_Scaling_Diffusion_Transformer_for_Generalist_Vision-Language-Action_Policy_ICCV_2025_paper.pdf

[^1_42]: https://www.amazon.science/publications/on-the-scalability-of-diffusion-based-text-to-image-generation

[^1_43]: https://huggingface.co/papers/2501.09732

[^1_44]: https://arxiv.org/abs/2411.17470

[^1_45]: https://arxiv.org/abs/2503.09443

[^1_46]: https://arxiv.org/abs/2501.06848

[^1_47]: https://arxiv.org/html/2505.15270v2

[^1_48]: https://arxiv.org/pdf/2001.08361.pdf

[^1_49]: https://arxiv.org/abs/2505.22524

[^1_50]: https://arxiv.org/html/2512.01426v1

[^1_51]: https://arxiv.org/abs/2503.00307

[^1_52]: https://arxiv.org/html/2410.15959v3

[^1_53]: https://arxiv.org/html/2312.04567v1

[^1_54]: https://arxiv.org/html/2510.24711v1

[^1_55]: https://arxiv.org/html/2410.08184v1

[^1_56]: https://openaccess.thecvf.com/content/CVPR2025/papers/Ma_Scaling_Inference_Time_Compute_for_Diffusion_Models_CVPR_2025_paper.pdf

[^1_57]: https://arxiv.org/abs/2503.07265

[^1_58]: https://www.semanticscholar.org/paper/e8f84138900a916be4476abbeb474fd89ce49e45

[^1_59]: https://arxiv.org/abs/2510.02987

[^1_60]: https://arxiv.org/abs/2506.08835

[^1_61]: https://arxiv.org/abs/2508.06152

[^1_62]: https://arxiv.org/abs/2505.21347

[^1_63]: https://arxiv.org/abs/2412.18150

[^1_64]: https://arxiv.org/abs/2310.11513

[^1_65]: https://arxiv.org/abs/2409.10695

[^1_66]: https://arxiv.org/abs/2410.05664

[^1_67]: https://arxiv.org/pdf/2310.11513.pdf

[^1_68]: http://arxiv.org/pdf/2403.04321.pdf

[^1_69]: https://arxiv.org/pdf/2307.06350.pdf

[^1_70]: https://arxiv.org/html/2503.21745v1

[^1_71]: https://arxiv.org/html/2406.03070

[^1_72]: https://arxiv.org/html/2412.18150

[^1_73]: https://arxiv.org/abs/2406.13743

[^1_74]: http://arxiv.org/pdf/2503.07265.pdf

[^1_75]: https://www.emergentmind.com/topics/geneval-2

[^1_76]: https://openaccess.thecvf.com/content/CVPR2024/papers/Lin_VILA_On_Pre-training_for_Visual_Language_Models_CVPR_2024_paper.pdf

[^1_77]: https://openaccess.thecvf.com/content/CVPR2025/papers/Zhu_DiG_Scalable_and_Efficient_Diffusion_Models_with_Gated_Linear_Attention_CVPR_2025_paper.pdf

[^1_78]: https://www.themoonlight.io/en/review/geneval-2-addressing-benchmark-drift-in-text-to-image-evaluation

[^1_79]: https://developer.nvidia.com/blog/vision-language-model-prompt-engineering-guide-for-image-and-video-understanding/

[^1_80]: https://arxiv.org/abs/2405.18428

[^1_81]: https://proceedings.neurips.cc/paper_files/paper/2023/file/a3bf71c7c63f0c3bcb7ff67c67b1e7b1-Paper-Datasets_and_Benchmarks.pdf

[^1_82]: https://arxiv.org/html/2312.07533v2

[^1_83]: https://kimjy99.github.io/논문리뷰/sana/

[^1_84]: https://liner.com/review/geneval-an-objectfocused-framework-for-evaluating-texttoimage-alignment

[^1_85]: https://github.com/NVlabs/VILA

[^1_86]: https://liner.com/ko/review/dig-scalable-and-efficient-diffusion-models-with-gated-linear-attention

[^1_87]: https://velog.io/@kirby_id/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-VILA-On-Pre-training-for-Visual-Language-Models

[^1_88]: https://www.themoonlight.io/en/review/dig-scalable-and-efficient-diffusion-models-with-gated-linear-attention

[^1_89]: https://arxiv.org/html/2512.22374

[^1_90]: https://www.semanticscholar.org/paper/GenEval-2:-Addressing-Benchmark-Drift-in-Evaluation-Kamath-Chang/e8f84138900a916be4476abbeb474fd89ce49e45

[^1_91]: https://arxiv.org/html/2507.23682v3

[^1_92]: https://arxiv.org/html/2310.11513

[^1_93]: https://arxiv.org/html/2503.16726v1

[^1_94]: https://arxiv.org/abs/2512.16853

[^1_95]: https://arxiv.org/html/2411.12915v2

[^1_96]: https://arxiv.org/abs/2509.24695

[^1_97]: https://arxiv.org/html/2409.04429v3

[^1_98]: https://arxiv.org/html/2509.24899v2
