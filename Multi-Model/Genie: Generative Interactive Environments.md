
# Genie: Generative Interactive Environments

> **논문 정보**
> - **제목**: Genie: Generative Interactive Environments
> - **저자**: Jake Bruce, Michael Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, et al. (Google DeepMind)
> - **발표**: ICML 2024 (Oral), Proceedings of the 41st International Conference on Machine Learning, PMLR 235:4603–4623
> - **arXiv**: [arXiv:2402.15391](https://arxiv.org/abs/2402.15391) (2024년 2월 23일)

---

## 1. 핵심 주장 및 주요 기여 요약

### 🔑 핵심 주장

Genie는 레이블이 없는 인터넷 동영상으로부터 비지도 방식으로 학습된 최초의 생성적 인터랙티브 환경(Generative Interactive Environment)입니다. 이 모델은 텍스트, 합성 이미지, 사진, 심지어 스케치로 프롬프트를 주면 무한히 다양한 액션 제어 가능한 가상 세계를 생성할 수 있으며, 110억(11B) 개의 파라미터를 가진 파운데이션 월드 모델로 볼 수 있습니다.

### 🏆 주요 기여 (4가지)

| # | 기여 내용 |
|---|-----------|
| 1 | **비지도 학습 기반 인터랙티브 환경 생성**: 액션 레이블 없이 인터넷 동영상만으로 학습 |
| 2 | **잠재 액션 모델(LAM)**: 행동 정보 없이 비지도로 잠재 액션 공간 학습 |
| 3 | **파운데이션 월드 모델**: 11B 파라미터 규모의 범용 월드 모델 구현 |
| 4 | **범용 에이전트 학습 경로 개척**: 학습된 잠재 액션으로 미관찰 환경에서 에이전트 모방 학습 가능 |

모델은 시공간 비디오 토크나이저(spatiotemporal video tokenizer), 자기회귀 다이나믹스 모델(autoregressive dynamics model), 잠재 액션 모델(latent action model)로 구성됩니다.

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

잠재 액션 모델은 제어 가능한 비디오 생성을 위해 설계되었으며, 인터넷에서 수집된 비디오에 대한 액션 레이블 주석은 비용이 많이 들기 때문에 비지도 방식으로 잠재 액션을 학습합니다. 기존 월드 모델 문헌에서는 일반적으로 **그라운드 트루스 액션 레이블**이나 **도메인 특화 요구 사항**이 필수였으나, Genie는 이러한 그라운드 트루스 액션 레이블이나 도메인 특화 조건 없이 프레임 단위로 생성된 환경에서 사용자가 행동할 수 있게 합니다.

또한, 기존 연구에서는 게임 환경이 AI 에이전트 개발에 효과적인 테스트베드였지만, 사용 가능한 게임 수에 한계가 있었습니다. Genie를 통해 미래의 AI 에이전트는 끝없이 생성되는 새로운 환경에서 훈련될 수 있습니다.

---

### 2.2 제안하는 방법 및 수식

#### 📐 전체 파이프라인

Genie는 최신 비디오 생성 모델의 아이디어를 기반으로 하며, 핵심 설계 선택으로 모든 모델 구성 요소에 ST-Transformer(공간-시간 변환기)를 사용합니다. 새로운 비디오 토크나이저를 활용하고, 인과적(causal) 액션 모델을 통해 잠재 액션을 추출하며, 비디오 토큰과 잠재 액션은 MaskGIT을 이용해 다음 프레임을 자기회귀적으로 예측하는 다이나믹스 모델에 전달됩니다.

---

#### 🔧 구성요소 1: 시공간 비디오 토크나이저 (ST Video Tokenizer)

비디오 프레임 $x_t \in \mathbb{R}^{H \times W \times 3}$을 이산 토큰 시퀀스로 인코딩합니다:

$$z_t = \text{Tokenizer}(x_t) \in \mathbb{Z}^{h \times w}$$

비디오 토크나이저는 비디오 프레임을 이산 토큰으로 인코딩하며, 이 토큰들이 다이나믹스 모델의 입력으로 사용됩니다.

모델 전체에 걸쳐 공간-시간(ST) 블록을 사용하며, 기본 비전 트랜스포머의 높은 메모리 복잡도를 고려하여 ST 블록은 선택된 토큰들 간의 어텐션을 수행하여 메모리 수요를 줄입니다. 각 ST 블록은 공간 어텐션 레이어(spatial attention layer)와 시간 어텐션 레이어(temporal attention layer)로 분할됩니다.

$$\text{ST-Block}(X) = \text{FFN}\Big(\text{Temporal-Attn}\big(\text{Spatial-Attn}(X)\big)\Big)$$

트랜스포머의 2차(quadratic) 메모리 비용은 최대 $O(10^4)$ 토큰을 포함할 수 있는 비디오에 문제가 되므로, 모델 용량과 계산 제약을 균형 있게 맞추는 메모리 효율적인 ST-Transformer 아키텍처를 채택합니다.

---

#### 🔧 구성요소 2: 잠재 액션 모델 (Latent Action Model, LAM)

LAM의 유일한 목표는 이산 잠재 액션의 코드북(codebook)을 형식화하는 것입니다. 이 코드북은 해석 가능한 액션(예: MOVE_RIGHT)을 장려하기 위해 의도적으로 작게 설계됩니다. 이러한 코드북을 학습하기 위해 LAM은 VQ-VAE 모델로 구축됩니다.

VQ-VAE 목적 함수:

$$\mathcal{L}_{\text{LAM}} = \underbrace{\|x_{t+1} - \hat{x}_{t+1}\|^2}_{\text{재구성 손실}} + \underbrace{\|\text{sg}[z_e] - e\|^2}_{\text{코드북 손실}} + \underbrace{\beta\|z_e - \text{sg}[e]\|^2}_{\text{commitment loss}}$$

여기서 $z_e$는 인코더 출력, $e$는 코드북 벡터, $\text{sg}[\cdot]$는 stop-gradient 연산자, $\beta$는 커밋먼트 손실 가중치입니다.

인코딩 단계에서 인코더는 모든 이전 프레임과 다음 프레임을 입력으로 받아 연속 잠재 액션 집합을 출력합니다. 그런 다음 디코딩 단계에서 디코더는 이전 프레임과 잠재 액션을 받아 다음 프레임을 예측합니다. VQ-VAE 기반 목적 함수를 사용하여 총 액션 수를 소규모의 이산 코드 집합으로 제한합니다. 액션 어휘 크기를 8로 제한하여 가능한 잠재 액션 수를 적게 유지합니다. 이를 통해 액션은 과거와 미래 사이의 가장 의미 있는 변화를 인코딩하게 됩니다.

---

#### 🔧 구성요소 3: 자기회귀 다이나믹스 모델 (Autoregressive Dynamics Model)

다이나믹스 모델은 과거 프레임 토큰들과 잠재 액션을 입력으로 받아 다음 프레임 토큰을 예측합니다:

$$\hat{z}_t = \text{DynamicsModel}(z_1, z_2, \ldots, z_{t-1},\; a_1, a_2, \ldots, a_{t-1})$$

다이나믹스 모델은 환경의 시간적 진화를 이해하고 예측하는 역할을 담당하며, 디코더 전용 MaskGIT을 사용하여 과거 토큰 프레임과 액션을 입력으로 받아 다음 프레임을 예측합니다.

MaskGIT 기반 마스크 예측 목적 함수:

$$\mathcal{L}_{\text{Dynamics}} = -\mathbb{E}\left[\sum_{i \in \mathcal{M}} \log p_\theta(z_t^{(i)} \mid z_t^{(\lnot\mathcal{M})}, z_{<t}, a_{<t})\right]$$

여기서 $\mathcal{M}$은 마스킹된 토큰의 인덱스 집합, $z_t^{(\lnot\mathcal{M})}$은 마스킹되지 않은 토큰들입니다.

다이나믹스 모델에 비디오 프레임 시퀀스 $(z_1, \ldots, z_{t-1})$와 액션 $(a_1, \ldots)$을 입력하여 다음 비디오 프레임 토큰 $z_t$를 얻습니다.

---

### 2.3 모델 구조 전체 요약

```
입력 이미지/스케치/텍스트
        ↓
[ST-Video Tokenizer]  →  이산 토큰 z_t ∈ ℤ^(h×w)
        ↓
[Latent Action Model (VQ-VAE)]  →  잠재 액션 a_t ∈ {0,...,7}
        ↓
[Autoregressive Dynamics Model (MaskGIT)]  →  다음 프레임 토큰 ẑ_{t+1}
        ↓
디코더  →  픽셀 프레임 x̂_{t+1}
```

아키텍처는 $L$개의 시공간 블록으로 구성되며, 각 블록은 공간 레이어(spatial layer), 시간 레이어(temporal layer), 피드포워드 레이어(feed-forward layer)를 포함합니다.

**데이터**: 저자들은 20만 시간 이상의 공개된 인터넷 2D 플랫포머 게임 영상을 수집하고, 저품질 영상을 필터링한 후 총 680만 개의 16초 비디오(30,000시간)를 학습에 사용하였습니다. 데이터셋에는 별도의 수정이 이루어지지 않았습니다.

---

### 2.4 성능 향상

CoinRun(절차적으로 생성된 2D 플랫포머 환경)에서 오라클 행동 복제(BC) 모델(전문가 액션에 접근 가능) 및 랜덤 에이전트와 비교 평가한 결과, LAM 기반 정책은 전문가 샘플을 200개만 적응시켜도 오라클과 동일한 점수를 달성했으며, Genie는 CoinRun을 훈련 중 거의 본 적이 없음에도 불구하고 이런 성과를 보였습니다.

---

### 2.5 한계점

**속도 문제**: Genie는 현재 초당 약 1프레임으로 실행되어 부드럽고 실시간 플레이에는 너무 느립니다.

**기억 한계**: 약 16프레임 정도의 짧은 히스토리만 유지하므로, 긴 시퀀스에서 일관성을 잃을 수 있습니다.

**환각(Hallucination)**: 다른 생성 모델과 마찬가지로 때때로 이상하거나 비현실적인 프레임을 생성할 수 있습니다.

**데이터 수집 비용**: Genie의 접근 방식은 환경 탐색에 대한 인간 시연에 의존하며, 플랫포머 게임의 온라인 플레이쓰루 비디오를 수집 및 정제하여 대규모 데이터셋을 구축하는 방식인데, 이러한 데이터셋은 구축하기 어렵고 다른 유형의 환경으로 전환 시 또 다른 비용이 발생합니다.

**복합 오류(Compounding Error)**: 월드 모델은 "복합 오류" 문제에 직면합니다. 작은 오류가 시간이 지남에 따라 누적되어 장기 예측이 불량해질 수 있습니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재 일반화 능력

Genie는 실제 사진이나 스케치처럼 이전에 보지 못한 이미지로 프롬프트를 줄 수 있으며, 이를 통해 사람들이 상상한 가상 세계와 상호작용할 수 있습니다. 이는 액션 레이블 없이 학습되었음에도 가능합니다. 대신 Genie는 대규모 인터넷 비디오 데이터셋으로 학습됩니다.

대조적으로, Genie는 인터넷 비디오에서 학습된 잠재 액션을 사용하여 임의 환경에 대한 정책을 추론할 수 있으며, 비용이 많이 들고 일반화되지 않을 수 있는 그라운드 트루스 액션의 필요성을 회피합니다.

### 3.2 잠재 액션의 전이 가능성

학습된 잠재 액션 공간은 에이전트가 미관찰 비디오에서 행동을 모방하도록 훈련하는 것을 용이하게 하며, 미래의 범용 에이전트 학습을 향한 경로를 열어줍니다.

논문에서는 Genie가 학습한 잠재 액션이 실제 인간이 설계한 환경으로 전이될 수 있다는 개념 증명(proof of concept)을 제시하고 있으나, 이는 표면적인 수준에 불과합니다.

### 3.3 스케일에 따른 일반화

11B 모델은 분포 외(out-of-distribution) 일반화 능력이 뛰어나 명확한 캐릭터 움직임을 보여줍니다. 더 작은 2.5B 모델은 로보틱스 데이터셋에서 훈련되어 로봇 팔의 제어뿐만 아니라 다양한 물체의 상호작용 및 변형도 학습합니다.

### 3.4 일반화 향상을 위한 후속 연구 (GenieRedux)

GenieRedux 연구는 Genie의 접근 방식을 재검토하여, Genie가 강력한 결과를 달성하지만 비용이 많이 드는 인간 데이터와 제한된 랜덤 에이전트 탐색에 의존한다는 점을 지적하고, RL 기반 탐색이 복잡한 환경에서 세계 모델의 일반화 가능성과 효율성을 향상시키는 확장 가능하고 효과적인 대안임을 입증하였습니다.

AutoExplore Agent는 월드 모델의 불확실성에만 의존하는 탐색 에이전트로, 가장 효과적으로 학습할 수 있는 다양한 데이터를 제공합니다.

---

## 4. 최신 관련 연구 비교 분석 (2020년 이후)

| 모델 | 연도 | 핵심 특징 | Genie와의 차이 |
|------|------|-----------|----------------|
| **DreamerV3** | 2023 | RSSM 기반 잠재 공간 동역학 모델, Minecraft 다이아몬드 획득 | 특정 환경에 특화, 액션 레이블 필요 |
| **IRIS** (Δ-IRIS) | 2022/2024 | Transformer 기반 월드 모델, 100k 인터랙션으로 Atari 달성 | 단일 게임 도메인, 소규모 |
| **Sora** (OpenAI) | 2024 | 대규모 비디오 생성, 물리적 시뮬레이션 | 인터랙티브 제어 불가, 액션 없음 |
| **Genie 2** (DeepMind) | 2024 | 3D 환경 생성, OOD 일반화 | Genie 1의 후속: 3D + 더 높은 일반화 |
| **iVideoGPT** | 2024 | 인터랙티브 비디오 GPT, 확장 가능한 월드 모델 | Genie와 유사하나 다른 아키텍처 |
| **GenieRedux** | 2024 | Genie 재구현 + RL 기반 탐색 에이전트 | Genie의 한계(인간 데이터 의존) 개선 |

지금까지 월드 모델은 주로 좁은 도메인 모델링에 국한되어 왔습니다. Genie 1에서는 다양한 2D 세계 생성 접근법을 소개하였고, Genie 2는 일반성 측면에서 상당한 도약을 나타냅니다.

Genie의 접근 방식은 MaskGIT 및 토크나이즈된 이미지에 대한 ST-Transformer를 사용한다는 점에서 Phenaki, TECO, MaskViT와 같은 최근 트랜스포머 기반 모델과 가장 유사합니다.

월드 모델은 RL 에이전트를 보조하는 대략적인 상상 모델에서 액션에 조건화된 독립적인 현실적 비디오 생성 모델로 발전해 왔습니다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5.1 연구에 미치는 영향

**① 파운데이션 월드 모델 패러다임 확립**

최근의 연구들은 태스크 특화 월드 모델에서 광범위하게 일반화할 수 있는 다양한 데이터로 훈련된 대규모 시스템인 파운데이션 월드 모델로 이동하는 경향을 보이고 있습니다.

**② 범용 RL 에이전트 훈련의 새로운 경로**

Genie를 에이전트 훈련에 활용하는 능력을 간략히 다루었지만, 풍부하고 다양한 환경의 부족이 RL의 핵심 한계 중 하나인 점을 고려하면, 더 일반적으로 유능한 에이전트를 만드는 새로운 경로를 열 수 있습니다.

**③ 크리에이티브 콘텐츠 생성의 민주화**

Genie는 어린이를 포함한 누구나 인간이 설계한 시뮬레이션 환경처럼 생성된 세계를 꿈꾸고, 만들고, 들어갈 수 있게 하는 새로운 형태의 생성 AI입니다.

**④ Genie 2로의 발전**

Genie 2는 미래 에이전트가 무한한 커리큘럼의 새로운 세계에서 훈련 및 평가될 수 있도록 하며, 인터랙티브 경험의 프로토타이핑을 위한 새로운 창의적 워크플로우를 개척할 수 있습니다.

---

### 5.2 앞으로 연구 시 고려할 점

| 고려 사항 | 설명 |
|-----------|------|
| **① 실시간 추론 최적화** | 미래 연구는 효율성을 향상시키고 실시간 속도로 복잡한 인터랙티브 환경을 생성하는 모델의 능력을 확장하는 데 집중할 수 있습니다. |
| **② 장기 일관성 유지** | 현재 약 16프레임의 짧은 히스토리만 유지하는 한계를 극복하여 긴 시퀀스에서도 일관성을 유지하는 메커니즘 연구가 필요합니다. |
| **③ 다양한 도메인 확장** | 이 연구는 아직 초기 단계이며 에이전트와 환경 생성 능력 모두에서 상당한 개선의 여지가 있으므로, 2D 플랫포머를 넘어 3D, 로보틱스 등 다양한 도메인으로의 확장이 필요합니다. |
| **④ 데이터 수집 자동화** | RL 기반 탐색이 복잡한 환경에서 세계 모델의 일반화 가능성과 효율성을 향상시키는 확장 가능하고 효과적인 대안임이 입증되었으므로, 인간 시연 의존도를 줄이는 자동화된 데이터 수집 방법 연구가 중요합니다. |
| **⑤ 물리적 일관성** | 생성된 환경의 물리 법칙 준수 및 인과성 유지를 위한 물리 기반 제약 조건 통합 연구가 필요합니다. |
| **⑥ 안전 및 윤리** | 이 기술이 기존 인간 게임 생성과 창의성을 증폭시키고 관련 산업이 Genie를 활용하여 차세대 플레이 가능 세계 개발을 가능하게 하는 데 활용할 가능성을 탐색하는 것이 중요합니다. |

---

## 📚 참고 자료 (출처)

1. **Bruce, J., et al. (2024)**. "Genie: Generative Interactive Environments." *Proceedings of the 41st International Conference on Machine Learning (ICML 2024)*, PMLR 235:4603–4623. https://arxiv.org/abs/2402.15391

2. **Google DeepMind 공식 논문 페이지**: https://deepmind.google/research/publications/60474/

3. **ICML 2024 Oral 발표 페이지**: https://icml.cc/virtual/2024/oral/35508

4. **Genie 공식 프로젝트 페이지**: https://sites.google.com/view/genie-2024/home

5. **Genie 2 공식 블로그 (Google DeepMind)**: https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/

6. **Savov, N., et al. (2024/2025)**. "Exploration-Driven Generative Interactive Environments (GenieRedux)." *CVPR 2025*. https://insait-institute.github.io/GenieRedux/

7. **Open-Genie PyTorch 구현**: https://github.com/myscience/open-genie

8. **Hugging Face 논문 페이지**: https://huggingface.co/papers/2402.15391

9. **Hugging Face 블로그 (Vlad Bogolin)**: https://huggingface.co/blog/vladbogo/genie-generative-interactive-environments

10. **ACM Digital Library**: https://dl.acm.org/doi/10.5555/3692070.3692255

11. **Awesome-Embodied-World-Model (GitHub)**: https://github.com/tsinghua-fib-lab/Awesome-Embodied-World-Model

12. **Medium — Arjun Agarwal, "Understanding Genie"** (Mar 2026): https://medium.com/@arjunagarwal899/understanding-genie-generative-interactive-environments-for-world-modeling-fbd90b446b3b

13. **rewire.it Blog — "What Are World Models?"**: https://rewire.it/blog/what-are-world-models-ai-path-to-understanding-reality/
