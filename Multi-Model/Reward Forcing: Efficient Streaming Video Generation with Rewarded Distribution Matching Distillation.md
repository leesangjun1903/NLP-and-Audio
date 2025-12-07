
# Reward Forcing: Efficient Streaming Video Generation with Rewarded Distribution Matching Distillation

## 1. 핵심 주장 및 주요 기여 요약 (Executive Summary)
본 논문은 실시간 스트리밍 비디오 생성 시 발생하는 **'초기 프레임 복제(Stagnation)'** 및 **'화질 저하(Drift)'** 문제를 동시에 해결하는 **Reward Forcing** 프레임워크를 제안합니다. 핵심 기여는 다음과 같습니다.

1.  **EMA-Sink 메커니즘**: 슬라이딩 윈도우에서 버려지는 토큰 정보를 지수 이동 평균(EMA)으로 압축하여 싱크 토큰(Sink Token)에 업데이트함으로써, 계산 비용 증가 없이 장기 문맥(Long-term Context)을 유지하고 초기 프레임에 대한 과도한 의존을 방지합니다.
2.  **Re-DMD (Rewarded Distribution Matching Distillation)**: 기존의 분포 매칭 증류(DMD)가 모든 샘플을 동일하게 취급하여 낮은 동적인(Static) 비디오를 생성하는 문제를 해결하기 위해, 비전-언어 모델(VLM) 기반의 보상 함수를 도입하여 고품질의 동적 움직임을 가진 샘플에 가중치를 부여하는 새로운 증류 방식을 제안합니다.
3.  **SOTA 달성**: H100 GPU 단일 칩에서 **23.1 FPS**의 실시간 생성 속도를 달성하며, VBench 등 주요 벤치마크에서 기존 SOTA 모델(LongLive, SkyReels-V2 등)을 상회하는 성능을 입증했습니다.

***

## 2. 상세 분석: 문제 정의, 제안 방법, 성능 및 한계

### 2.1 해결하고자 하는 문제 (Problem Definition)
기존의 스트리밍 비디오 생성 모델(예: Sliding Window Attention 기반)은 두 가지 주요 딜레마에 직면해 있습니다.
*   **Error Accumulation (오류 누적)**: 이전 프레임의 작은 오류가 다음 프레임으로 전파되어 시간이 지날수록 화질이 붕괴되는 현상.
*   **Over-Attention / Stagnation (과도한 의존 및 정체)**: 이를 막기 위해 초기 프레임을 고정된 싱크(Sink) 토큰으로 사용하면, 모델이 초기 프레임에 과도하게 집중하여 이후 프레임이 초기 이미지를 그대로 복사하거나 움직임이 사라지는(Static) 현상이 발생합니다.
*   **DMD의 한계**: 기존의 DMD(Distribution Matching Distillation)는 생성된 샘플이 실제 분포와 유사하기만 하면 되므로, 움직임이 적은(안전한) 샘플을 생성해도 페널티를 주지 못해 동적 품질이 저하됩니다.

### 2.2 제안하는 방법 (Methodology)

#### A. EMA-Sink: 동적 상태 패키징 (State Packaging)
슬라이딩 윈도우 밖으로 밀려나는(evicted) 토큰 정보를 버리지 않고, 싱크 토큰에 지수 이동 평균(Exponential Moving Average) 방식으로 융합합니다. 이를 통해 **최근의 동적 정보**와 **장기적인 문맥**을 동시에 유지합니다.

$$ S^K_i = \alpha \cdot S^K_{i-1} + (1 - \alpha) \cdot K_{i-w} $$
$$ S^V_i = \alpha \cdot S^V_{i-1} + (1 - \alpha) \cdot V_{i-w} $$

*   $$S^K, S^V$$: 압축된 싱크 토큰의 Key, Value 상태
*   $$K_{i-w}, V_{i-w}$$: 윈도우에서 방출되는 i-w 시점의 Key, Value
*   $$\alpha$$: 과거 정보의 보존 비율을 결정하는 모멘텀 계수 (논문에서는 0.99 수준 사용)

#### B. Re-DMD: 보상 기반 분포 매칭 증류
모델이 더 높은 동적 품질(Motion Dynamics)을 가진 비디오를 생성하도록 유도하기 위해, VLM이 평가한 보상(Reward) 점수를 증류 손실 함수에 가중치로 반영합니다.

$$ \nabla_{\theta} J_{\text{Re-DMD}} \approx -\mathbb{E}_t \left[ \mathbb{E}_{\epsilon} \left[ \underbrace{\frac{\exp(r(x_t)/\beta)}{Z(c)}}_{\text{Reward Weight}} \cdot (\underbrace{s_{\text{real}}(\Psi(G_{\theta}(\epsilon), t), t) - s_{\text{fake}}(\Psi(G_{\theta}(\epsilon), t), t)}_{\text{Score Difference}}) \frac{d G_{\theta}(\epsilon)}{d \theta} d\epsilon \right] \right] $$

*   $$r(x_t)$$: 생성된 비디오 $$x_t$$에 대한 보상 점수 (VLM 측정)
*   $$\beta$$: 보상 민감도를 조절하는 온도 파라미터 (Temperature parameter)
*   $$s_{\text{real}}, s_{\text{fake}}$$: Teacher 모델(Real)과 Student 모델(Fake)의 점수 함수(Score function)

이 수식은 보상이 높은($$r(x_t)$$가 큰) 샘플 쪽으로 모델의 업데이트 방향(Gradient)을 더 강하게 유도합니다.

### 2.3 모델 구조 (Model Architecture)
*   **Base Model**: Wan2.1-T2V-1.3B (최신 오픈소스 비디오 생성 모델 기반)
*   **Inference**: Autoregressive 방식의 KV Cache 활용, 5초 단위의 짧은 클립 학습을 통해 무한 길이 생성으로 확장.
*   **Chunk Processing**: 한 번의 스텝에서 3개의 Latent Frame을 처리하여 효율성 극대화.

### 2.4 성능 향상 및 한계 (Performance & Limitations)
*   **성능**: **23.1 FPS** (H100 기준)로 실시간 생성이 가능하며, VBench 평가에서 Total Quality, Motion Dynamics, Temporal Consistency 모든 부문에서 SOTA를 달성했습니다. 특히 장기 생성 시 발생하는 화질 저하(Drift)를 크게 억제했습니다.
*   **한계**:
    *   **보상 모델 의존성**: Re-DMD의 성능은 보상 함수(VLM)의 정확도에 비례합니다. 보상 모델이 잘못된 지표를 학습했다면 생성 모델도 편향될 수 있습니다.
    *   **평가 지표 불일치**: 학습에 사용된 보상 함수와 최종 평가 지표(VBench) 간의 미세한 정렬 불일치가 존재할 수 있습니다.

***

## 3. 모델의 일반화 성능 향상 가능성 (Generalization Capabilities)

이 논문의 방법론은 특히 **"시간적(Temporal) 일반화"** 측면에서 탁월한 가능성을 보여줍니다.

1.  **무한 길이로의 일반화 (Extrapolation)**:
    *   대다수 모델은 5초 내외의 짧은 클립으로 학습되므로, 추론 시 길이가 길어지면 문맥을 잃거나 화질이 급격히 떨어집니다.
    *   **EMA-Sink**는 고정된 메모리 크기 내에서 과거의 모든 정보를 압축하여 보존하므로, 학습된 길이를 초과하는 **무한 스트리밍 생성**에서도 초기 프레임의 스타일과 객체 일관성을 잃지 않도록 일반화됩니다. 이는 OOD(Out-of-Distribution) 시간 영역에서의 강건성을 보장합니다.

2.  **동적 움직임의 일반화**:
    *   일반적인 증류 모델은 평균적인(안전한) 분포로 수렴하려는 경향 때문에 정적인 비디오를 생성하기 쉽습니다.
    *   **Re-DMD**는 학습 데이터에 없더라도 보상 함수가 선호하는 "역동적인 움직임" 영역으로 모델의 분포를 강제로 이동(Forcing)시킵니다. 이는 모델이 단순히 학습 데이터를 암기하는 것을 넘어, **"고품질의 움직임이란 무엇인가"**에 대한 일반적인 기준을 학습하게 만듭니다.

***

## 4. 향후 연구 영향 및 고려사항 (Future Impact)

### 4.1 향후 연구에 미치는 영향
*   **Real-time Interactive World Simulators**: 23FPS의 속도와 무한 생성 능력은 사용자의 입력에 실시간으로 반응하는 '게임 같은' 비디오 생성 모델(World Simulator)의 실현을 앞당길 것입니다.
*   **Reward-centric Generative Models**: LLM에서 RLHF가 표준이 된 것처럼, 비디오 생성 분야에서도 단순히 데이터 분포를 따라가는 것을 넘어 **보상 모델(Reward Model)을 통해 원하는 특성(예: 물리 법칙 준수, 역동성)을 주입하는 연구**가 가속화될 것입니다.

### 4.2 연구 시 고려할 점
*   **Reward Hacking 방지**: Re-DMD 사용 시, 모델이 보상 점수만 높이고 실제 시각적 품질은 기괴해지는 'Reward Hacking' 현상을 방지하기 위해 KL Divergence 제약 조건($$\beta$$)을 신중하게 튜닝해야 합니다.
*   **효율적인 Memory Management**: EMA-Sink는 효율적이지만, 압축 과정에서 손실되는 정보가 정밀한 텍스트 제어(Instruction Following)에 어떤 영향을 미치는지에 대한 후속 연구가 필요합니다.

***

## 5. 2020년 이후 관련 최신 연구 탐색

본 논문의 위치를 파악하기 위해 2020년 이후, 특히 2024-2025년의 핵심 연구 흐름을 정리합니다.

| 연도 | 모델/연구명 | 주요 특징 및 본 논문과의 관계 |
| :--- | :--- | :--- |
| **2025** | **SkyReels-V2** | **Diffusion Forcing**을 적용하여 무한 길이 생성을 시도했으나, 속도(0.49 FPS)면에서 본 논문(23.1 FPS)에 비해 현저히 느림. |
| **2025** | **LongLive** | **Attention Sink**와 KV-recache를 도입하여 장기 일관성을 확보했으나, 초기 프레임에 과도하게 의존하여 움직임이 정체되는 한계를 보임 (본 논문이 해결한 지점). |
| **2025** | **Wan 2.1** | 본 논문의 **Base Model**. 강력한 성능을 가진 최신 Video DiT 모델이지만, 기본적으로 양방향(Bidirectional) 구조라 스트리밍 생성에는 무거움. |
| **2024** | **DMD (Distribution Matching Distillation)** | 본 논문의 방법론적 기초. 이미지/비디오 생성 단계를 1~4 스텝으로 줄이는 증류 기법이나, 동적 품질 저하 문제가 존재함. |
| **2024** | **VideoReward / InstructVideo** | 비디오 생성을 위한 **인간 피드백 기반 강화학습(RLHF)** 연구들. Re-DMD가 VLM 보상을 활용하는 이론적 토대가 됨. |

이 연구는 기존의 **효율성(Distillation)** 연구와 **품질 향상(Reward Learning)** 연구, 그리고 **장기 생성(Attention Sink)** 연구를 하나의 프레임워크로 성공적으로 통합했다는 점에서 중요한 의의를 가집니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/56fee686-3302-46b4-b4d8-9e79eb6a455a/2512.04678v1.pdf)
[2](http://arxiv.org/pdf/2402.03681.pdf)
[3](https://arxiv.org/html/2502.01719v3)
[4](https://arxiv.org/pdf/2501.13918.pdf)
[5](https://arxiv.org/abs/2412.15689)
[6](https://arxiv.org/pdf/2312.12490.pdf)
[7](https://arxiv.org/html/2404.14735v1)
[8](http://arxiv.org/pdf/2402.03746.pdf)
[9](https://arxiv.org/html/2412.21059)
[10](https://arxiv.org/html/2512.04678v1)
[11](https://clippie.ai/blog/ai-video-creation-trends-2025-2026)
[12](https://fal.ai/models/fal-ai/wan-i2v)
[13](https://filmart.ai/skyreels-v2-open-source-ai-video-generator/)
[14](https://www.linkedin.com/posts/yukang-chen-35aaa2151_longvideogeneration-longvideounderstanding-activity-7384440538072170496-ri0h)
[15](https://huggingface.co/papers/2512.04678)
[16](https://www.futuremarketinsights.com/reports/video-streaming-market)
[17](https://www.datacamp.com/tutorial/wan-2-1)
[18](https://www.capcut.com/resource/skyreels-v2-tutorial)
[19](https://huggingface.co/papers/2509.22622)
