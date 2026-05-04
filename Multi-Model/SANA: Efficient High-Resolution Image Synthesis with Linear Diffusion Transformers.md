# SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers

SANA 논문은 “선형 Diffusion Transformer + 강한 압축 오토인코더 + LLM 텍스트 인코더 + 효율적인 flow 기반 샘플러”를 조합해, 4K까지의 고해상도 이미지를 수십 배 빠른 속도로 생성하면서도 기존 거대 모델(SD3, FLUX 등)에 필적하는 품질을 달성하는 효율적 텍스트-투-이미지 프레임워크를 제안합니다. 특히 토큰 수를 16배 줄이는 AE-F32, 선형 어텐션 DiT, Gemma 기반 텍스트 인코더, Flow-DPM-Solver 덕분에 0.6B~1.6B 파라미터 규모로 12B급 모델과 경쟁하며 4K 해상도까지 일반화되는 것이 핵심 주장입니다.[^1][^2][^3]

***

## 1. 핵심 주장과 주요 기여

- **딥 컴프레션 오토인코더(AE-F32C32)**
기존 LDM 계열이 주로 다운샘플링 배율 $F=8$인 오토인코더(AE-F8)를 사용한 데 비해, SANA는 $F=32$ 오토인코더를 설계해 토큰 수를 16배 감소시키면서도 SDXL의 AE-F8과 비슷한 재구성 품질을 유지합니다. 이로써 4K까지의 초고해상도에서도 훈련·추론 비용이 크게 줄어듭니다.[^2][^1]
- **선형 DiT(Linear Diffusion Transformer)**
기존 DiT의 자가어텐션 $O(N^2)$ 복잡도를 ReLU 기반 선형 어텐션으로 모두 교체해, 토큰 수에 선형으로 스케일하는 $O(N)$ 구조를 제안합니다. Mix-FFN(3×3 depth-wise conv + GLU) 덕분에 위치 임베딩 없이(NoPE)도 품질을 유지하면서 고해상도에서 1.7× 이상의 가속을 얻습니다.[^1]
- **Decoder-only LLM 텍스트 인코더 + CHI**
T5 대신 Gemma-2 계열 decoder-only LLM을 텍스트 인코더로 사용하고, Complex Human Instruction(CHI) 프롬프트와 in-context learning을 활용해 텍스트 이해·지시 따르기 능력과 텍스트-이미지 정렬을 강화합니다. RMSNorm과 작은 스케일 파라미터로 LLM 임베딩을 안정화해 학습 발산 문제도 해결합니다.[^2][^1]
- **Flow 기반 학습·추론(Flow-DPM-Solver)**
Rectified Flow/EDM 계열의 데이터·속도(velocity) 예측 목표를 사용하고, DPM-Solver++를 Rectified Flow에 맞게 수정한 Flow-DPM-Solver를 제안해, 14–20 스텝 내 수렴하면서 Flow-Euler 대비 더 좋은 FID/CLIP을 달성합니다. 이로써 동일 품질 기준에서 필요한 샘플링 스텝 수를 대폭 줄입니다.[^1]
- **고속·온디바이스 추론**
Sana-0.6B는 4096×4096 생성에서 FLUX-dev 대비 100× 이상 빠른 처리량(throughput)을 보이고, 1024×1024에서는 39× 이상 빠르며, 8-bit 정수 양자화(W8A8) 후에도 랩톱 GPU에서 1K 이미지를 0.37초에 생성할 수 있습니다.[^3][^2][^1]

***

## 2. 이 논문이 해결하려는 문제

고해상도 텍스트-투-이미지 모델은 최근 품질·상업적 가치 모두에서 큰 진전을 보였지만, 모델 크기와 계산비용이 폭발적으로 증가했습니다. FLUX(12B), Playground v3(24B), SD3 등은 뛰어난 품질을 제공하지만, 4K 해상도에서 훈련 및 추론 비용이 매우 크고, 일반 사용자·엣지 디바이스에서는 사용이 어렵습니다.[^3][^2][^1]

특히 DiT 기반 모델은 토큰 수에 대해 $O(N^2)$로 증가하는 어텐션 비용 때문에 고해상도에서 심각한 메모리·시간 병목을 겪습니다. 또한 기존 오토인코더는 $F=8$ 수준의 다운샘플만 수행해 고해상도에서 토큰 수가 지나치게 많고, 텍스트 인코더(T5, CLIP)의 지시 따르기 능력 및 텍스트 이해력도 한계가 있었습니다.[^1]

SANA가 묻는 핵심 질문은 “4K 고해상도에서도 고품질을 유지하면서, 작은 모델로 빠르게 동작하고 엣지 디바이스에도 배치 가능한 텍스트-투-이미지 모델을 설계할 수 있는가?”입니다.[^2][^1]

***

## 3. 제안 방법과 수식 중심 설명

### 3.1 딥 컴프레션 오토인코더 (AE-F32C32P1)

기존 LDM은 오토인코더가 이미지를 $F=8$로 다운샘플해 잠복공간(latent space)에서 diffusion을 수행합니다. SANA는 autoencoder를[^1]

$$
E : \mathbb{R}^{H \times W \times 3} \rightarrow \mathbb{R}^{\frac{H}{32} \times \frac{W}{32} \times C}, \quad C=32
$$

로 설계하여 $F=32$로 길이·너비를 압축하고, patch size $P=1$인 DiT에 바로 입력합니다. 즉 토큰 수는[^1]

$$
N = \frac{H}{32} \cdot \frac{W}{32}
$$

이며, 기존 AE-F8C4P2 구조 대비 토큰 수를 4배, AE-F8C16P4 대비 16배 줄입니다.[^1]

다양한 조합(AE-F8C16P4, AE-F16C32P2, AE-F32C32P1)을 동일 토큰 수(예: 1024×1024 → 32×32 tokens) 기준으로 비교한 결과, AE-F8C16이 재구성 품질(rFID)이 가장 좋지만, 텍스트 조건 생성 품질(FID)은 AE-F32C32P1이 가장 우수했습니다. 이는 “압축은 AE가 전담하고, DiT는 순수하게 노이즈 제거에 집중하게 하는” 분업이 더 낫다는 설계 철학을 뒷받침합니다.[^1]

### 3.2 선형 어텐션 기반 Linear DiT

SANA의 핵심은 DiT에서 모든 full self-attention을 ReLU 기반 선형 어텐션으로 치환한 것입니다. 일반적인 softmax attention은[^1]

$$
\mathrm{Attn}(Q, K, V)_i = \sum_{j=1}^N \mathrm{softmax}\big(Q_i K_j^\top\big)_j V_j
$$

로 $O(N^2)$ 복잡도를 갖습니다. SANA는 $\phi(x) = \mathrm{ReLU}(x)$를 사용해 다음과 같이 정의합니다.[^1]

$$
O_i = \frac{\sum_{j=1}^N \phi(Q_i)\,\phi(K_j)^\top V_j}{\sum_{j=1}^N \phi(Q_i)\,\phi(K_j)^\top}
$$

이를 전개하면, 쿼리 $Q_i$에 공통인 항을 분리하여

$$
S_V = \sum_{j=1}^N \phi(K_j)^\top V_j, \quad S_K = \sum_{j=1}^N \phi(K_j)^\top
$$

를 미리 한 번만 계산하고,

$$
O_i = \frac{\phi(Q_i)\, S_V}{\phi(Q_i)\, S_K}
$$

로 표현할 수 있어, $S_V, S_K$ 계산이 $O(N)$, 각 쿼리 업데이트도 $O(d)$에 가능해 전체 복잡도가 $O(N)$이 됩니다. ReLU 대신 일반적인 kernel $\phi$를 사용한 선형 어텐션(Linformer, Performer 계열)과 유사한 구조지만, 고해상도 이미지 생성에서의 효과를 실증한 점이 주요 공헌입니다.[^1]

또한 FFN을 **Mix-FFN**으로 교체합니다.[^1]

- Inverted residual block
- 3×3 depth-wise convolution
- Gated Linear Unit(GLU)

조합을 사용해, 선형 어텐션의 상대적으로 약한 로컬 정보 포착 능력을 보완하고, 3×3 conv가 위치 정보를 암묵적으로 인코딩하게 만들어 별도의 positional embedding 없이도 좋은 성능(NoPE)을 얻습니다.[^1]

### 3.3 Flow 기반 학습과 Flow-DPM-Solver

SANA는 SD3에서 사용한 Rectified Flow(RF) 계열 formulation을 채택합니다. 일반적인 확산 모델은[^4][^1]

$$
x_t = \alpha_t x_0 + \sigma_t \epsilon
$$

형태의 forward process를 공유하며, DDPM은 노이즈 예측을 목표로 합니다.[^1]

- **DDPM(노이즈 예측)**:

$$
\epsilon_\theta(x_t, t) \approx \epsilon
$$
- **EDM(데이터 예측)**:

$$
x_\theta(x_t, t) \approx x_0
$$
- **Rectified Flow(속도 예측)**:

$$
v_\theta(x_t, t) \approx \epsilon - x_0
$$

SANA는 velocity 예측을 사용하면서, 샘플러 단계에서 DPM-Solver++를 변형하여 Flow-DPM-Solver를 정의합니다. 핵심 아이디어는 (1) 스케일링 계수 $\alpha_t$를 $1-\sigma_t$로 치환하여 RF에 맞는 시간 스케줄을 사용하고, (2) 네트워크 출력 $v_\theta$를 데이터 예측 형태 $x_\theta$로 변환해 안정적인 고차 ODE solver를 적용하는 것입니다.[^1]

예를 들어, 초기 상태에서 데이터 복원을

$$
x_0 \approx x_T - \sigma_T \, v_\theta(x_T, t_T)
$$

형태로 얻고, 이 $x_\theta$를 사용하는 수정된 DPM-Solver++ 알고리즘(Flow-DPM-Solver)을 제안해, 14–20 스텝에서 안정적으로 수렴합니다. 실험적으로 Flow-Euler(28–50 스텝 필요)보다 더 적은 스텝에서 더 좋은 FID/CLIP을 보입니다.[^1]

### 3.4 멀티 캡션 레이블링과 CLIP-score 샘플링

텍스트-이미지 정렬을 개선하기 위해, 각 이미지에 대해 VILA-3B/13B, InternVL2-8B/26B 등 네 가지 VLM으로 다중 캡션을 생성합니다. 학습 시 하나를 무작위로 고르는 대신, CLIP score 기반의 소프트 샘플링을 사용합니다.[^1]

$$
P(c_i) = \frac{\exp(c_i / \tau)}{\sum_{j=1}^N \exp(c_j / \tau)}
$$

여기서 $c_i$는 캡션 $c_i$의 CLIP score, $\tau$는 temperature입니다. $\tau \to 0$이면 최고 점수 캡션만 선택되고, 큰 $\tau$는 더 균일한 샘플링을 유도합니다. Multi-caption + CLIP 샘플링은 FID에는 큰 변화를 주지 않으면서 CLIP-score를 개선해 의미 정렬을 향상시키는 것으로 보고됩니다.[^1]

### 3.5 Gemma 텍스트 인코더와 CHI

텍스트 인코더로 Gemma-2 2B 등의 decoder-only LLM을 사용하고, 마지막 레이어의 hidden state를 텍스트 임베딩으로 사용합니다. 하지만 LLM 임베딩은 T5 대비 분산이 몇 자릿수 더 커서, 그대로 cross-attention에 넣으면 학습이 쉽게 NaN으로 발산하는 문제가 있습니다.[^2][^1]

이를 해결하기 위해 SANA는

1. RMSNorm으로 텍스트 임베딩의 분산을 1.0으로 정규화
2. 작은 학습 가능한 스케일 팩터(예: 0.01)를 곱해 점진적으로 텍스트 조건의 세기를 키움

전략을 사용해 안정성을 확보하고, 수렴 속도도 개선합니다.[^1]

또한 LiDiT에서 영감을 받아 **Complex Human Instruction(CHI)**를 도입해,

- LLM이 프롬프트를 “설명/리라이팅”하도록 돕는 복잡한 시스템 프롬프트를 설계하고
- in-context learning 예시를 제공해, 짧은 프롬프트(예: “a cat”)에도 세밀한 디테일이 포함된 임베딩을 생성

하도록 합니다. 실험적으로 CHI를 추가하면 GenEval 점수가 약 2포인트 상승하고, 짧은 프롬프트에서 안정적인 생성이 가능해집니다.[^1]

***

## 4. 모델 구조

SANA의 전체 파이프라인은 다음과 같습니다.[^1]

1. **이미지 → latent**:
오토인코더 encoder가 입력 이미지를 $F=32$로 다운샘플하여 latent $z \in \mathbb{R}^{H/32 \times W/32 \times 32}$를 생성합니다.[^1]
2. **latent → 토큰**:
patch size $P=1$로 flatten하여 $N = (H/32) (W/32)$개의 토큰으로 만든 뒤, Linear DiT에 입력합니다.[^1]
3. **텍스트 인코딩**:
Gemma-2 LLM이 텍스트를 임베딩하고 RMSNorm + 스케일링을 거쳐 cross-attention의 key/value로 사용됩니다.[^1]
4. **Linear DiT 블록**:
각 블록은
    - ReLU-linear attention
    - Mix-FFN (inverted residual + 3×3 depth-wise conv + GLU)
    - LayerNorm / RMSNorm
로 구성되며, base 1K 모델 수준에서는 positional embedding 없이 사용됩니다(NoPE).[^1]
5. **디코더**:
diffusion 모델이 예측한 깨끗한 latent를 오토인코더 decoder가 업샘플링해 최종 이미지를 생성합니다.[^1]

네트워크 규모는 Sana-0.6B(폭 1152, 깊이 28, 약 590M 파라미터)와 Sana-1.6B(폭 2240, 깊이 20, 약 1.6B 파라미터) 두 가지 변형이 주요하며, 20–30층 사이가 효율성과 품질의 균형점임을 보고합니다.[^1]

***

## 5. 성능 향상과 한계

### 5.1 효율성과 품질

- **효율성**
1024×1024 해상도에서 Sana-0.6B는 FLUX-dev 대비 약 39×, Sana-1.6B는 약 23× 빠른 throughput을 보입니다. 4096×4096에서는 Sana-0.6B가 FLUX-dev 대비 100× 이상 빠른 throughput, 106× 빠른 latency 개선을 달성합니다. 0.6B 모델만으로도 1K 이미지를 A100에서 0.9초, 소비자용 RTX 4090에서는 양자화 후 0.37초에 생성합니다.[^2][^1]
- **품질 지표**
MJHQ-30K 데이터셋 기준 FID, CLIPScore에서 PixArt-Σ(0.6B), SDXL(2.6B)와 동급 혹은 더 우수한 성능을 보이며, GenEval·DPG-Bench·ImageReward 등 다양한 벤치마크에서도 1–3B 파라미터급 모델 중 상위권에 위치합니다. FLUX-dev(12B) 대비 GenEval은 약간 낮지만 DPG-Bench는 비슷한 수준이며, 훨씬 작은 모델 크기와 압도적인 속도를 고려하면 “효율 대 품질” 측면에서 강한 trade-off를 달성합니다.[^1]


### 5.2 한계

- **안전성과 제어 가능성**
저자들은 생성 이미지의 안전·제어 가능성(유해 콘텐츠, 스타일·콘텐츠 제어 등)을 완전히 보장하지 못하며, 이는 향후 과제임을 명시합니다.[^1]
- **복잡한 텍스트 렌더링, 얼굴/손**
텍스트 렌더링과 복잡한 손·얼굴 등은 여전히 어려운 케이스로, 예시에서도 일부 아티팩트와 오타, 비자연스러운 손가락이 나타납니다.[^1]
- **고압축 AE의 잠재적 정보 손실**
F32 오토인코더는 AE-F8 대비 재구성 품질을 상당 부분 회복했지만(rFID 개선), 이론적으로는 세밀한 로컬 텍스처 정보를 일부 손실할 위험이 있습니다. 논문은 FID/CLIP 기준으로는 병목이 아니라고 주장하지만, 세부 텍스처나 특수 도메인(의료영상 등)에 대한 일반화는 추가 검증이 필요합니다.[^1]

***

## 6. 일반화 성능 향상 가능성에 대한 분석

### 6.1 NoPE와 토큰 길이/해상도 일반화

SANA는 base 1K 모델에서 명시적인 positional embedding을 완전히 제거(NoPE)하고, 3×3 depth-wise conv가 암묵적인 위치 정보를 인코딩하게 만듭니다. 이는 LLM 분야에서 NoPE가 길이 일반화를 개선한다는 결과와 부합하며, 토큰 수 증가(해상도 상승)에 대해 구조적으로 더 잘 generalize할 수 있는 설계입니다.[^1]

실제로 SANA는 동일 아키텍처를 512 → 1024 → 2K → 4K cascade fine-tuning으로 점진적으로 확장하면서도, 4K에서 안정적으로 수렴하고 디테일이 개선되는 결과를 보입니다. 이는 위치 임베딩 보간에 의존하는 기존 DiT 계열보다 해상도 일반화 측면에서 구조적인 이점을 시사합니다.[^1]

### 6.2 Flow 기반 목적과 샘플링 안정성

Rectified Flow/EDM 계열의 데이터·velocity 예측은, 특히 $t \approx T$ 근방에서 노이즈 예측에 비해 누적 오차와 불안정을 줄이는 것으로 알려져 있습니다. SANA는 이 이점을 활용해 Flow-DPM-Solver를 설계하고, 14–20 스텝에서 안정적으로 수렴하는 샘플러를 얻습니다.[^4][^1]

스텝 수가 줄어들면, (1) 다양한 guidance 강도·스케줄에 대한 민감도가 낮아지고, (2) 샘플링 스케줄 변경·이식(예: 다른 해상도, 다른 도메인) 시에도 robust하게 동작할 가능성이 커집니다. 이는 모델이 새로운 조건·도메인으로 전이될 때 **샘플러 측면의 일반화**를 어느 정도 확보해 준다고 해석할 수 있습니다.[^4][^1]

### 6.3 딥 컴프레션 AE와 표현 일반화

AE-F32는 고해상도 이미지의 redundant한 정보를 latent에서 강하게 압축해, DiT가 상대적으로 “정보 밀도가 높은” latent 토큰 위에서 학습하도록 만듭니다. 이는[^1]

- 고해상도 디테일을 모두 직접 모델링하기보다,
- 전역 구조·중요한 시맨틱 요소에 집중하게 만들어,

텍스트 조건에 따른 전역 구조·구성(composition) 측면의 일반화를 돕는 측면이 있습니다. 동시에, AE가 충분히 표현력이 없으면 세부 텍스처 일반화가 희생될 수 있으므로, AE 설계·학습이 generalization의 병목이 되지 않도록 하는 것이 중요합니다.[^1]

### 6.4 LLM 텍스트 인코더 + CHI, 멀티 캡션이 주는 효과

Gemma 기반 텍스트 인코더와 CHI, 멀티 캡션 + CLIP 샘플링은 텍스트 도메인에서의 일반화에 직접적으로 기여합니다.[^1]

- 짧고 모호한 프롬프트를 LLM이 풍부하고 구조화된 프롬프트로 확장해 주므로, 학습 시 다양한 세부 묘사에 노출됩니다.[^1]
- 여러 VLM이 생성한 캡션과 CLIP-score 기반 샘플링을 통해, 단일 noisy 캡션에 과적합하는 것을 줄이고, 다양한 표현에 대해 의미 정렬을 학습합니다.[^1]

흥미롭게도, SANA는 학습 시 영어 프롬프트만 사용했는데도, 추론 시 중국어/이모지 프롬프트에 대해 의미 있는 이미지를 생성할 수 있음을 보여 줍니다. 이는 LLM 텍스트 인코더의 다국어·이모지 generalization을 이미지 생성으로 일부 전이했다는 점에서, 텍스트 측 일반화의 강점을 방증합니다.[^1]

### 6.5 양자화 후 성능 유지와 강인성

W8A8 양자화 후에도 CLIP Score와 ImageReward가 거의 유지되면서(예: CLIPScore 약 28.5 → 28.3), latency는 2.4× 개선됩니다. 이는 표현이 양자화 잡음에 비교적 강인하며, latent·텍스트 표현이 안정적으로 학습되었음을 시사합니다. 양자화·프루닝·지연 평가 등 다양한 배포 세팅에서 품질 유지가 가능하다는 점은, 일반화된 표현 학습의 한 정량적인 지표로 볼 수 있습니다.[^1]

***

## 7. 2020년 이후 관련 최신 연구 비교 분석

### 7.1 주요 관련 논문 (제목·저자·링크·요약)

- **SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers** – Enze Xie et al., arXiv 2024.[^5][^3][^2]
Linear DiT, AE-F32, Gemma 텍스트 인코더, Flow-DPM-Solver를 결합해 4K까지 고해상도 이미지를 효율적으로 생성하는 텍스트-투-이미지 프레임워크를 제안합니다.
- **PixArt-Σ: Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation** – Junsong Chen et al., arXiv/ECCV 2024.[^6][^7][^8][^9]
0.6B 파라미터 DiT로 4K 이미지를 직접 생성하며, 고품질 데이터와 효율적인 토큰 압축, weak-to-strong training으로 고해상도·고품질을 달성합니다.
- **Scaling Rectified Flow Transformers for High-Resolution Image Synthesis (Stable Diffusion 3)** – Patrick Esser et al., arXiv 2024.[^10][^11][^12][^4]
Rectified Flow 기반의 multi-modal DiT(MM-DiT)를 제안해, 고해상도 텍스트-투-이미지에서 기존 diffusion formulation을 능가하는 품질과 샘플링 효율을 보여 줍니다.
- **High-Resolution Image Synthesis with Latent Diffusion Models** – Robin Rombach et al., CVPR 2022.[^1]
LDM 프레임워크를 제안해, autoencoder로 이미지 공간을 latent로 압축한 뒤 diffusion을 수행함으로써 고해상도 이미지를 효율적으로 생성하는 길을 열었습니다. SANA의 AE-F32 설계는 이 계열의 직접적인 후속입니다.[^1]
- **PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis** – Junsong Chen et al., ICLR 2024.[^1]
DiT 기반 텍스트-투-이미지에서 fast training 기법을 제안하며, PixArt-Σ의 기반이 되는 모델입니다. SANA는 PixArt 계열과 비교하여 더 강한 압축 및 선형 어텐션을 도입합니다.[^8][^1]
- **Efficient Diffusion Transformers with Linear Compressed Attention (EDiT)** – Haochen Lu et al., arXiv 2025.[^13][^14]
클래식 DiT와 MM-DiT의 $O(N^2)$ 문제를 해결하기 위해, locally modulated linear attention과 hybrid attention(이미지 간은 선형, 텍스트 관련은 softmax)을 도입하며, PixArt-Σ와 SD3.5-Medium에 적용해 2.2× 가속과 유사 품질을 달성합니다. SANA를 선행연구로 명시적으로 인용하며, linear-time DiT 계열의 확장 사례입니다.[^14]


### 7.2 방법론적 비교 (요약 표)

| 모델 | 연도 | 핵심 아이디어 | 최대 해상도(보고) | 어텐션 복잡도 | 압축 방식 | 텍스트 인코더 | 비고 |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| **LDM** (Rombach et al.)[^1] | 2022 | Autoencoder로 이미지 공간을 latent로 압축 후 diffusion 수행 | 수백~수천 해상도 (VAE-F8) | U-Net + conv, 사실상 $O(N)$ | AE-F8 | CLIP | latent diffusion 패러다임 확립 |
| **PixArt-Σ**[^6][^7][^8][^9] | 2024 | DiT + 효율적 토큰 압축 + weak-to-strong training | 직접 4K | DiT softmax $O(N^2)$ + token compression | AE-F8 + KV 압축 | T5-XXL | 0.6B 파라미터로 4K, 고품질 |
| **SD3 (Rectified Flow Transformers)**[^4][^10][^11] | 2024 | Rectified Flow + MM-DiT, 노이즈 샘플링 스케줄 최적화 | 1K 이상 고해상도 | DiT softmax $O(N^2)$ | AE-F8 계열 | T5 + CLIP 조합 | 대규모 RF 기반 텍스트-투-이미지 |
| **SANA**[^1][^5][^2][^3] | 2024 | AE-F32 + Linear DiT + Gemma 텍스트 인코더 + Flow-DPM-Solver | 4K (4096×4096) | 완전 선형 어텐션 $O(N)$ | AE-F32C32P1 | Gemma-2 decoder-only | FLUX 대비 100× 빠르고, 비슷한 품질 |
| **EDiT**[^13][^14] | 2025 | Linear compressed attention + hybrid attention (이미지/텍스트 분리) | PixArt-Σ, SD3.5 기반으로 상속 | 선형/softmax 혼합 | 기존 AE 활용 | 기존 텍스트 인코더 상속 | SANA 이후 선형 DiT 계열 확장 |

### 7.3 SANA의 위치와 향후 영향

- **지금까지의 흐름 속에서의 위치**
LDM이 latent space diffusion을 열었고, PixArt-Σ와 SD3가 고해상도·Rectified Flow·DiT 아키텍처를 통해 품질을 끌어올렸습니다. SANA는 이 흐름 위에서 **“고해상도 + 효율성(선형 시간·메모리) + LLM 텍스트 인코더”**를 동시에 달성한 첫 공개 연구 중 하나로, 고해상도 텍스트-투-이미지에서 효율성을 새로운 1급 목표로 설정하는 전환점을 제공합니다.[^7][^3][^4][^2][^1]
- **후속 연구에 미치는 영향**
이미 Efficient Diffusion Transformers with Linear Compressed Attention(EDiT)와 같은 후속 연구가 SANA를 인용하며, linear-time DiT를 PixArt-Σ, SD3.5와 결합해 추가적인 속도 향상을 보여 주고 있습니다. 또한 “SANA 1.5: Efficient Scaling of Training-Time and Inference-Time Compute in Linear Diffusion Transformer”에서는 깊이 성장 파라다임과 8-bit 옵티마이저, 반복적 샘플링 스케일링 등을 통해 SANA 계열을 수십억 파라미터로 확장하면서도 효율성을 유지하는 방식을 연구하고 있습니다.[^15][^13][^14]


### 7.4 앞으로 연구 시 고려할 점

- **압축–품질 trade-off 탐색**
AE-F32가 보여 준 것처럼, autoencoder 압축 비율을 크게 높여도 적절한 설계·학습으로 품질을 유지할 수 있습니다. 향후 연구에서는 도메인별(의료, 위성 등) 최적 압축 배율, 비선형/어댑티브 압축, 토큰 중요도 기반 비균일 압축 등 보다 정교한 설계를 탐색할 필요가 있습니다.[^1]
- **선형 어텐션·NoPE 설계 공간**
SANA와 EDiT는 서로 다른 선형 어텐션 설계(ReLU vs convolution-modulated linear attention)를 제안합니다. 향후 연구에서는[^13][^14][^1]
    - 이미지/텍스트 토큰에 다른 커널을 쓰는 hybrid 설계,
    - NoPE/상대 위치/implicit conv 위치 인코딩의 조합,
    - 초고해상도(8K 이상)에서의 length generalization
을 체계적으로 비교·분석하는 것이 중요합니다.
- **LLM 텍스트 인코더와 멀티모달 일반화**
Gemma 기반 텍스트 인코더와 CHI, multi-caption pipeline은 텍스트 이해·지시 따르기·다국어 generalization에 큰 영향을 줍니다. 향후에는[^1]
    - LLM을 직접 멀티모달(텍스트+이미지)로 미세조정,
    - instruction tuning과 safety alignment를 결합,
    - text-video·text-3D 등으로 확장
하는 방향이 유망합니다.
- **Flow 기반 샘플러와 이론적 분석**
Rectified Diffusion/Rectified Flow에 대한 이론적 분석이 진행 중이며, SANA의 Flow-DPM-Solver는 여기에 실무적 변형을 더한 사례입니다. 향후에는[^16][^4][^1]
    - 다양한 flow 계열 목적(velocity vs data vs hybrid) 비교,
    - adaptive step size, guidance-aware ODE solver,
    - domain shift 상황에서의 안정성 분석
이 중요한 연구 주제가 될 것입니다.
- **안전성·제어·책임 있는 사용**
SANA 저자들이 지적하듯, 안전성·제어 가능성은 아직 해결되지 않은 과제입니다. 고해상도·실사 품질이 높아질수록[^1]
    - 안전한 프롬프트 필터링,
    - controllable generation(레이아웃, 스타일, 인물 비식별화),
    - 데이터 저작권·편향 문제
를 중심으로 한 “책임 있는 고해상도 생성” 연구가 병행되어야 합니다.

위 내용은 원 논문 *“SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers”* 본문과 보충 자료, 그리고 관련 오픈 액세스 논문들(*PixArt-Σ*, *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis* (SD3), *Efficient Diffusion Transformers with Linear Compressed Attention (EDiT)* 등)을 바탕으로 정리했습니다.[^11][^12][^9][^5][^6][^7][^10][^14][^8][^13][^3][^4][^2][^1]
<span style="display:none">[^17][^18][^19][^20][^21][^22][^23][^24][^25][^26][^27][^28][^29][^30]</span>

<div align="center">⁂</div>

[^1]: 2410.10629v3.pdf

[^2]: https://arxiv.org/html/2410.10629v3

[^3]: https://huggingface.co/papers/2410.10629

[^4]: https://arxiv.org/abs/2403.03206

[^5]: https://arxiv.org/abs/2410.10629

[^6]: https://arxiv.org/html/2403.04692v2

[^7]: https://arxiv.org/abs/2403.04692

[^8]: https://ar5iv.labs.arxiv.org/html/2403.04692

[^9]: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04633.pdf

[^10]: https://arxiv.org/pdf/2403.03206.pdf

[^11]: https://encord.com/blog/stable-diffusion-3-text-to-image-model/

[^12]: https://huggingface.co/posts/kadirnar/764069086101800

[^13]: https://arxiv.org/abs/2503.16726

[^14]: https://arxiv.org/html/2503.16726v2

[^15]: https://arxiv.org/html/2501.18427v3

[^16]: https://arxiv.org/html/2410.07303v1

[^17]: https://www.semanticscholar.org/paper/24cf7478fbba1d698e80f196254b050e13d64e6b

[^18]: https://arxiv.org/abs/2512.12595

[^19]: https://arxiv.org/abs/2410.22655

[^20]: http://arxiv.org/pdf/2410.10629.pdf

[^21]: http://arxiv.org/pdf/2409.19589.pdf

[^22]: https://www.semanticscholar.org/paper/SANA:-Efficient-High-Resolution-Image-Synthesis-Xie-Chen/c0f9315d8175b76097eadff3a2686ef203e6ee6b

[^23]: https://www.semanticscholar.org/paper/PixArt-Σ:-Weak-to-Strong-Training-of-Diffusion-for-Chen-Ge/f6632f0c4633ea981684a16a05f5d7d46d1d586c

[^24]: https://arxiv.org/html/2403.03206v1

[^25]: https://arxiv.org/html/2508.07246v1

[^26]: https://arxiv.org/pdf/2403.04692.pdf

[^27]: https://arxiv.org/html/2509.06068v1

[^28]: https://www.themoonlight.io/en/review/sana-efficient-high-resolution-image-synthesis-with-linear-diffusion-transformers

[^29]: https://lemuria.es/post/3969712

[^30]: https://kimjy99.github.io/논문리뷰/stable-diffusion-3/

