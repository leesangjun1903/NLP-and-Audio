
# SANA-Video: Efficient Video Generation with Block Linear Diffusion Transformer

> **논문 정보**
> - **제목**: SANA-Video: Efficient Video Generation with Block Linear Diffusion Transformer
> - **저자**: Junsong Chen 외 19명 (NVIDIA, MIT, Tsinghua University 등)
> - **arXiv**: [2509.24695](https://arxiv.org/abs/2509.24695) (2025년 9월 29일 제출, 10월 13일 v2)
> - **GitHub**: [NVlabs/Sana](https://github.com/NVlabs/Sana)
> - **프로젝트 페이지**: [nvlabs.github.io/Sana/Video](https://nvlabs.github.io/Sana/Video)

---

## 1. 핵심 주장과 주요 기여 요약

SANA-Video는 최대 720×1280 해상도, 분(minute) 단위 길이의 영상을 효율적으로 생성할 수 있는 소형 Diffusion 모델입니다. 고해상도·고품질·긴 영상을 강력한 텍스트-영상 정합 능력과 함께 매우 빠른 속도로 생성하며, RTX 5090 GPU에서 구동 가능합니다.

**핵심 기여 두 가지:**

① **Linear DiT**: 영상 생성에서 처리해야 하는 방대한 토큰 수를 고려해, 기존의 Vanilla Attention보다 효율적인 **Linear Attention**을 핵심 연산으로 채택하였습니다. ② **상수 메모리 KV 캐시 (Constant-Memory KV Cache for Block Linear Attention)**: Linear Attention의 누적 특성(cumulative properties)에서 유래한 상수 메모리 상태를 활용한 **블록 단위 자기회귀(block-wise autoregressive)** 방식으로 긴 영상 생성을 설계하였습니다.

또한 효과적인 데이터 필터링 및 학습 전략을 탐구하여, 학습 비용을 64개의 H100 GPU 기준 12일로 줄였으며, 이는 MovieGen 대비 단 1%에 해당하는 수준입니다. 이 저비용으로 Wan 2.1-1.3B, SkyReel-V2-1.3B 등 최신 소형 Diffusion 모델 대비 경쟁력 있는 성능을 달성하면서 측정 지연(latency) 기준 **16배 빠른 속도**를 달성했습니다.

---

## 2. 문제 정의 · 제안 방법 · 모델 구조 · 성능 및 한계

### 2.1 해결하고자 하는 문제

현대 영상 생성 모델이 직면한 두 가지 핵심 병목은 다음과 같습니다:

**① 계산 복잡도 문제**

Transformer 기반 생성 모델의 핵심인 표준 Softmax Attention은 시퀀스 길이 $N$에 대해 $O(N^2)$의 계산 비용이 소요됩니다. SANA가 처리하는 큰 잠재 공간(latent space)과 텍스트 조건화의 결합에서 이 이차적 스케일링은 병목이 됩니다.

**② 긴 영상 생성 시 메모리 폭증 문제**

영상을 생성할 때 이미지에 비해 토큰 수가 극적으로 증가하며, 이는 합리적인 처리 시간을 유지하기 위해 Linear Attention을 필수적으로 만듭니다.

---

### 2.2 제안 방법 및 수식

#### (A) Linear Attention

기존 Softmax Attention은 다음과 같이 정의됩니다:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V$$

이 연산은 $O(N^2)$ 복잡도를 가집니다.

Linear Attention은 전체 Attention 행렬을 근사하지만 시퀀스 길이에 대해 선형적으로( $O(N)$ ) 스케일링됩니다.

Linear Attention의 일반적인 형태는 커널 함수 $\phi$를 사용해:

$$\text{LinearAttn}(Q, K, V) = \frac{\phi(Q)\left(\phi(K)^\top V\right)}{\phi(Q)\left(\phi(K)^\top \mathbf{1}\right)}$$

으로 표현되며, 분모는 정규화를 위한 항입니다. 연산 순서를 바꾸면 $\phi(K)^\top V$를 먼저 계산해 $O(Nd^2)$의 선형 복잡도를 달성합니다.

#### (B) Constant-Memory Block KV Cache

긴 영상 생성을 위해 Linear Attention의 누적 특성에서 파생된 상수 메모리 상태를 활용하는 Block-wise Autoregressive 방식을 설계하였습니다. 이 KV 캐시는 고정 메모리 비용으로 Linear DiT에 전역(global) 컨텍스트를 제공하며, 전통적인 KV 캐시의 필요성을 없애 효율적인 분 단위 영상 생성을 가능하게 합니다.

블록 $t$에 대한 Recurrent State의 업데이트는 다음과 같이 표현할 수 있습니다:

$$S_t = S_{t-1} + \sum_{i \in \text{block}_t} \phi(k_i) v_i^\top$$

$$\text{output}_t = \frac{\phi(q_t) S_t}{\phi(q_t) z_t}$$

여기서 $z_t = z_{t-1} + \sum_{i \in \text{block}_t} \phi(k_i)$는 정규화 누적 벡터입니다. 이 구조는 블록이 아무리 늘어나도 $S_t$의 메모리 크기가 $O(d^2)$로 **고정**되는 것이 핵심입니다.

#### (C) Self-Forcing Block Training (노출 편향 해소)

자기회귀 모듈을 먼저 학습한 후, 개선된 Self-Forcing Block Training으로 노출 편향(exposure bias)을 해결하는 두 단계 방식을 채택하였습니다. 이 과정은 긴 영상 생성을 위한 고품질·효율적 모델을 만들어냅니다.

#### (D) DC-AE-V (Deep Compression AutoEncoder for Video)

기존 오토인코더가 이미지를 8배 압축하는 데 반해, SANA의 DC-AE는 32배 압축을 수행합니다. 이 압축은 처리해야 할 잠재 토큰 수를 줄여 시스템을 더 효율적으로 만들며, 영상 생성에서 각 프레임의 처리가 필요하기 때문에 이 압축의 중요성은 더욱 증가합니다.

---

### 2.3 모델 구조

SANA-Video는 SANA를 기반 아키텍처로 채택하고, T2V(Text-to-Video) 과제의 고유한 도전을 처리하기 위해 Linear Diffusion Transformer 블록을 혁신적으로 커스터마이징 하였습니다.

**학습 파이프라인은 3단계로 구성됩니다:**

| 단계 | 내용 |
|------|------|
| Stage 1: VAE 적응 | 480P 영상에는 Wan-VAE를, 720P 고해상도 영상에는 더 높은 압축비를 제공하는 자체 개발 DCAE-V를 적용합니다. |
| Stage 2: T2I에서 T2V로 Pre-training 계속 | 사전 학습된 T2I 모델에서 영상 Linear DiT를 초기화함으로써, 잘 학습된 시각·텍스트 의미 지식을 효율적·효과적으로 활용합니다. |
| Stage 3: 자기회귀 블록 학습 | 블록 단위 Autoregressive 학습 후 Self-Forcing으로 노출 편향 수정 |

텍스트 인코더로는 디코더 전용(decoder-only) LLM을 사용하여 더 나은 텍스트-영상 정합을 달성하고, Block Causal Linear Attention과 Causal Mix-FFN으로 긴 영상 생성에 효율적인 Attention 및 피드포워드 연산을 구성하며, Flow-DPM-Solver로 샘플링 스텝을 감소시켰습니다.

---

### 2.4 성능 향상

시스템은 5초짜리 720p 영상을 60초의 지연(latency)으로 생성할 수 있으며, RTX 5090 GPU에서 NVFP4 정밀도를 사용하면 이를 29초로 단축할 수 있습니다.

Wan 2.1-1.3B, SkyReel-V2-1.3B 등 최신 소형 Diffusion 모델 대비 경쟁력 있는 성능을 달성하면서 측정 지연 기준 **16배 빠른 속도**를 달성하고, NVFP4 정밀도로 RTX 5090에서 5초짜리 720p 영상 생성 속도를 71초에서 29초로 단축(2.4배 가속)하였습니다.

학습 과정은 64개의 H100 GPU에서 12일이 소요되며, 이는 MovieGen 같은 대규모 모델 학습 비용의 1%에 불과하여 연구자와 기관이 이 기술에 보다 쉽게 접근할 수 있게 합니다.

---

### 2.5 한계점

논문에서 직접 명시된 한계는 아래와 같이 추론할 수 있습니다:

1. **Linear Attention의 표현력 한계**: DiT의 Self-Attention을 Linear Attention으로 단순히 교체하면 FID 및 Clip Score 성능 손실이 발생하며, Mix-FFN을 추가해야 이 성능 손실을 보상할 수 있습니다.
2. **480P에서의 고압축 VAE 적용 한계**: 480P 영상에서는 고압축비 VAE가 전반적인 성능을 제한하므로 Wan-VAE를 채용해야 합니다.
3. **하드웨어 의존성**: RTX 5090, H100 등 고사양 GPU 기반 배포 최적화에 집중되어 있어 보다 저사양 환경에서의 범용 적용성 검증이 제한적입니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 VAE 빠른 적응에서의 일반화 증거

두 VAE(Wan-VAE, DCAE-V) 모두에 대한 적응이 5,000~10,000 학습 스텝 내에 수렴하며, 이는 **Linear DiT의 강력한 일반화 능력**을 잘 보여줍니다.

### 3.2 T2I → T2V 전이 학습을 통한 일반화

사전 학습된 T2I 모델로부터 영상 Linear DiT를 초기화하는 것은, 잘 학습된 시각·텍스트 의미 지식을 활용하는 효율적이고 효과적인 방법입니다. 이는 모달리티 간 전이(transfer) 학습이 일반화 성능 향상에 직접 기여함을 의미합니다.

### 3.3 다운스트림 태스크 파인튜닝 일반화

영상 생성 모델은 자율주행 시나리오의 좋은 시뮬레이터로도 활용될 수 있으며, SANA-Video는 전방 카메라 데이터(30 FOV)를 포함한 내부 주행 데이터로 파인튜닝하여 다양하고 현실적인 주행 장면을 생성하는 데 활용되었습니다.

게임 생성 분야에서도 다운스트림 응용을 탐구하여 인터랙티브 비디오 게임을 생성하는 데 활용하였고, 구체적으로 Minecraft 플레이 화면 레코딩 데이터인 VPT를 학습 데이터로 사용하였습니다.

### 3.4 하이브리드 아키텍처로의 확장 가능성

후속 연구인 SANA-WM에서는 LTX2 VAE 스왑이 품질을 유지하면서 효율성을 크게 향상시키고, 하이브리드 백본이 기준 SANA-Video 대비 장거리 일관성(long-range consistency)과 I2V 차원을 향상시킴을 보여줍니다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

| 모델 | 연도 | 핵심 기법 | 비교 포인트 |
|------|------|-----------|------------|
| **DDPM** | 2020 | Denoising Score Matching | 확산 모델의 기초; $O(N^2)$ Attention |
| **DiT** (Peebles & Xie) | 2023 | Diffusion + Transformer | Softmax Attention 기반; SANA-Video의 아키텍처 출발점 |
| **Open-Sora** | 2024 | Efficient Video Production | 민주화된 효율적 영상 제작을 목표로 하나, 학습 비용이 높음 |
| **MovieGen** (Meta) | 2024 | 대규모 미디어 기반 모델 | 고품질이지만 SANA-Video 대비 학습 비용이 100배 이상 |
| **Wan 2.1-1.3B** | 2025 | 소형 Diffusion | 동급 성능이나 속도에서 SANA-Video에 비해 뒤처짐 |
| **SANA (T2I)** | 2025 | Linear DiT + DC-AE | ICLR 2025 Oral; SANA-Video의 직접적 기반 모델 |
| **SANA-Video** | 2025 | Block Linear DiT + Const. KV Cache | 경쟁 모델 대비 16배 빠른 latency |
| **SANA-WM** | 2025 | Hybrid Linear DiT | 2.6B 파라미터 제어 가능 월드 모델, 720p 1분 영상, 6-DoF 카메라 제어 |

효율적인 장문 컨텍스트 모델링은 순수 Softmax Attention을 넘어서 Linear Attention, Kernelized Attention, Gated Linear Attention, State-Space 모델, 합성곱 믹서, 테스트 타임 학습 레이어, Delta-rule 순환 등으로 발전하였습니다. 최신 장문 컨텍스트 언어 아키텍처는 순환·선형·상태공간 레이어를 일부 정확한 Attention이나 희소 모듈과 결합하여 대부분의 레이어를 효율적으로 유지합니다.

---

## 5. 앞으로의 연구에 미치는 영향과 고려할 점

### 5.1 연구에 미치는 영향

**① 효율적 영상 생성의 민주화**

SANA-Video는 MovieGen 같은 대형 모델 학습에 필요한 계산 자원의 1%만을 필요로 합니다. 이 낮은 진입 장벽은 제한된 계산 예산을 가진 연구자와 기관이 이 기술에 더 쉽게 접근할 수 있게 합니다.

**② Linear Attention의 영상 생성 적용 가능성 확인**

SANA와 SANA-Video는 이미지 및 영상 확산 생성에 Linear Attention 백본을 사용하고, DC-AE, DC-VideoGen, LTX-style VAE 등의 고압축 토크나이저로 생성기가 처리하는 시각 토큰 수를 줄이는 방향으로 발전하고 있음을 보여줍니다.

**③ 파인튜닝 생태계 확장**

SANA-Video는 Cosmos-RL과 협업하여 SANA-Image 및 SANA-Video 모두에 대한 완전한 RL 인프라(SFT/RL)를 제공하게 되었습니다.

**④ 세계 모델(World Model)로의 확장 경로 제시**

SANA-WM으로의 발전은 2.6B 파라미터의 제어 가능한 세계 모델로, 720p 1분 분량의 영상 월드를 6-DoF 카메라 제어와 함께 생성하는 방향으로 이어졌습니다.

---

### 5.2 앞으로 연구 시 고려할 점

| 고려 사항 | 설명 |
|-----------|------|
| **Linear Attention의 표현력 보완** | Softmax Attention을 Linear Attention으로 단순 교체 시 성능 저하가 발생하므로, Mix-FFN이나 희소 Attention 혼합 같은 보완 설계가 필요합니다. |
| **저해상도(480P) VAE 고압축 한계 극복** | 480P에서는 고압축비 VAE가 전반적 성능을 제한하므로 해상도별 최적 압축 전략 연구가 필요합니다. |
| **노출 편향(Exposure Bias) 해결** | 블록 단위 자기회귀 학습에서 훈련-추론 분포 차이를 더욱 정교하게 해소하는 방법론 연구가 필요합니다. |
| **다양한 도메인 일반화** | 자율주행, 게임 생성 등 전문 도메인 파인튜닝 연구가 시작되었으나, 의료 영상·산업 영상 등 더 다양한 도메인으로의 일반화 검증이 필요합니다. |
| **양자화(Quantization) 및 경량화** | NVFP4로 2.4배 추가 가속이 가능하지만, INT4·INT8 혼합 정밀도나 지식 증류(Knowledge Distillation)와 결합한 더 극단적인 경량화 연구의 여지가 있습니다. |
| **음향 생성과의 통합** | 현재 SANA-Video는 시각 영상만을 생성하며, 오디오-영상 공동 생성으로의 확장이 향후 과제입니다. |
| **RL 기반 정렬(Alignment) 학습** | Flow-GRPO, Diffusion-NFT 등의 알고리즘을 통한 Post-Training 인프라가 구축되어 있어, 인간 선호도 정렬 연구에 활용 가능성이 높습니다. |

---

## 참고 자료 (출처)

1. **arXiv 논문 원문**: Junsong Chen et al., "SANA-Video: Efficient Video Generation with Block Linear Diffusion Transformer," arXiv:2509.24695, Sep. 2025. [https://arxiv.org/abs/2509.24695](https://arxiv.org/abs/2509.24695)
2. **NVIDIA Research 프로젝트 페이지**: [https://research.nvidia.com/labs/eai/publication/sana-video/](https://research.nvidia.com/labs/eai/publication/sana-video/)
3. **논문 HTML 전문**: [https://arxiv.org/html/2509.24695v1](https://arxiv.org/html/2509.24695v1)
4. **OpenReview**: [https://openreview.net/forum?id=mzAchylAtf](https://openreview.net/forum?id=mzAchylAtf)
5. **HuggingFace Papers**: [https://huggingface.co/papers/2509.24695](https://huggingface.co/papers/2509.24695)
6. **GitHub (NVlabs/Sana)**: [https://github.com/NVlabs/Sana](https://github.com/NVlabs/Sana)
7. **SANA (T2I 기반 논문)**: Enze Xie et al., "SANA: Efficient High-Resolution Text-to-Image Synthesis with Linear Diffusion Transformers," ICLR 2025 Oral. [https://arxiv.org/html/2410.10629v3](https://arxiv.org/html/2410.10629v3)
8. **SANA-WM (후속 연구)**: "SANA-WM: Efficient Minute-Scale World Modeling with Hybrid Linear Diffusion Transformer," arXiv:2605.15178. [https://arxiv.org/html/2605.15178v1](https://arxiv.org/html/2605.15178v1)
9. **sanavideo.org 프로젝트 설명**: [https://sanavideo.org/](https://sanavideo.org/)
10. **SANA MIT HAN Lab 프로젝트**: [https://hanlab.mit.edu/projects/sana](https://hanlab.mit.edu/projects/sana)
11. **ResearchGate PDF**: [https://www.researchgate.net/publication/395969314](https://www.researchgate.net/publication/395969314)
