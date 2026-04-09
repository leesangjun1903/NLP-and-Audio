
# VLOGGER: Multimodal Diffusion for Embodied Avatar Synthesis

> **논문 정보**
> - **저자**: Enric Corona, Andrei Zanfir, Eduard Gabriel Bazavan, Nikos Kolotouros, Thiemo Alldieck, Cristian Sminchisescu (Google Research)
> - **arXiv**: [2403.08764](https://arxiv.org/abs/2403.08764) (2024년 3월 13일)
> - **학회**: CVPR 2025 (IEEE Xplore 수록 확인)
> - **프로젝트 페이지**: https://enriccorona.github.io/vlogger/

---

## 1. 핵심 주장 및 주요 기여 요약

VLOGGER는 단 한 장의 입력 이미지로부터 텍스트 및 오디오 기반의 talking human 비디오를 생성하는 방법으로, 최근 생성적 확산 모델의 성공 위에 구축됩니다. 핵심 방법론은 ① 확률론적 human-to-3D-motion diffusion 모델, ② 텍스트-이미지 모델에 시간적(temporal) 및 공간적(spatial) 제어를 추가하는 새로운 diffusion 기반 아키텍처의 두 가지로 구성됩니다.

### 🔑 주요 기여 (4가지)

| 기여 | 내용 |
|---|---|
| **①** 2단계 파이프라인 | 오디오→3D 모션 예측 → 시공간 제어 비디오 생성 |
| **②** MENTOR 데이터셋 | 800,000 정체성, 3D 포즈·표정 주석 포함 대규모 데이터셋 신규 구축 |
| **③** 공정성(Fairness) | 다양한 인종·성별·연령에서 편향 없는 일반화 성능 |
| **④** 실용 응용 | 비디오 편집, 개인화(personalization), 영상 번역 등 다양한 다운스트림 태스크 |

이전 연구와 달리 VLOGGER는 각 개인별 학습이 필요 없고, 얼굴 감지·크롭에 의존하지 않으며, 얼굴이나 입술만이 아닌 전신 이미지를 생성하고, 상체가 보이는 경우나 다양한 피사체 정체성 등 광범위한 시나리오를 처리합니다.

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 🎯 해결하고자 하는 문제

멀티모달 포토리얼리스틱 인간 합성은 데이터 수집, 자연스러운 표정 구현, 표정-오디오 동기화, 가림(occlusion), 전신 동작 표현 등의 복합적 도전 과제로 인해 매우 복잡합니다.

특히 VLOGGER의 목표는, 신원(identity)이나 포즈 제어 없이 동적 영상만 생성 가능한 최근 비디오 합성 방법론과, 제어 가능한 이미지 생성 방법론 사이의 간극을 해소하는 것입니다. 이를 위해 먼저 입력 오디오 신호에 따라 신체 동작과 표정을 예측하는 확산 기반 네트워크를 사용하는 2단계 방식을 제안합니다. 이러한 확률론적 접근은 음성과 포즈, 시선, 표정 사이의 미묘한 일대다(one-to-many) 매핑을 모델링하는 데 필수적입니다.

---

### 2.2 🏗️ 제안 방법 및 수식

#### 전체 파이프라인

$$
\text{Audio Waveform} + I_{ref} \xrightarrow{\text{Stage 1: Motion Diffusion}} \mathbf{M}_{1:T} \xrightarrow{\text{Stage 2: Video Diffusion}} \hat{V}_{1:T}
$$

**Stage 1 — Stochastic Human-to-3D-Motion Diffusion Model**

첫 번째 네트워크는 오디오 파형을 입력으로 받아 목표 비디오 길이에 걸친 시선(gaze), 표정(facial expressions), 포즈(pose)를 책임지는 중간 신체 모션 제어 신호를 생성합니다.

오디오로부터 3D 모션 표현으로의 확산 모델 목표 함수(DDPM):

$$
\mathcal{L}_{motion} = \mathbb{E}_{t, \mathbf{m}_0, \epsilon}\left[\left\|\epsilon - \epsilon_\theta\left(\mathbf{m}_t, t, \mathbf{a}\right)\right\|^2\right]
$$

여기서:
- $\mathbf{m}_t$: 타임스텝 $t$에서 노이즈가 가해진 모션 시퀀스
- $\mathbf{a}$: 오디오 특징(audio features)
- $\epsilon_\theta$: 노이즈 예측 네트워크
- $\epsilon$: 추가된 가우시안 노이즈

**Classifier-Free Guidance (CFG):**

에블레이션 연구에서 temporal loss와 classifier-free guidance가 이미지 품질 및 LME에서 최상의 성능을 이끌어냄을 보였습니다.

$$
\hat{\epsilon}_\theta = \epsilon_\theta(\mathbf{m}_t, t, \emptyset) + w \cdot \left(\epsilon_\theta(\mathbf{m}_t, t, \mathbf{a}) - \epsilon_\theta(\mathbf{m}_t, t, \emptyset)\right)
$$

여기서 $w$는 guidance scale.

**Stage 2 — Temporal Diffusion Model for Video Generation**

두 번째 네트워크는 대형 이미지 확산 모델을 확장한 시간적(temporal) 이미지-이미지 변환 모델로, 예측된 신체 모션 제어를 입력받아 해당하는 프레임을 생성합니다. 특정 정체성에 조건을 부여하기 위해 네트워크는 해당 인물의 참조 이미지도 입력으로 받습니다.

비디오 생성 목표 함수:

$$
\mathcal{L}_{video} = \mathbb{E}_{t, \mathbf{x}_0, \epsilon}\left[\left\|\epsilon - \epsilon_\theta\left(\mathbf{x}_t, t, \mathbf{M}_{1:N}, I_{ref}\right)\right\|^2\right]
$$

여기서:
- $\mathbf{x}_t$: 노이즈가 가해진 비디오 프레임 시퀀스
- $\mathbf{M}_{1:N}$: Stage 1에서 예측된 3D 모션 제어 시퀀스
- $I_{ref}$: 참조 이미지 (정체성 조건)

**Temporal Outpainting for Variable-Length Generation:**

제안된 시간적 확산 모델은 고정된 프레임 수 $N$만 생성하도록 훈련되므로 가변 길이 비디오로 확장하는 방법이 자명하지 않습니다. 대부분의 이전 확산 기반 비디오 생성 방법들은 짧은 클립으로 제한되어 있어, 이 논문에서는 temporal outpainting이라는 아이디어를 탐구합니다: 먼저 $N$개의 프레임을 생성하고, 이전 $N - N'$개의 프레임을 기반으로 반복적으로 $N' < N$개의 프레임을 outpaint합니다.

$$
\hat{\mathbf{x}}_{N+1:N+N'} = \text{OutpaintDiffusion}\left(\mathbf{x}_{N-N'+1:N}, \mathbf{M}_{N+1:N+N'}, I_{ref}\right)
$$

최종 파이프라인은 신체 제어를 적용하며, 제안된 temporal outpainting(전체 모델에서 50% 오버랩)이 최상의 시간적 일관성을 달성합니다.

---

### 2.3 🏛️ 모델 구조 세부 사항

아키텍처는 공간적·시간적 프레임워크를 제공하는 제어 기능을 통합합니다. 예를 들어, 통계적 3D 신체 모델을 사용하여 목표 정체성의 기하학적 측면을 인코딩하며, 다양한 신체 부위 가시성 수준에 적응할 수 있는 강건한 인간 합성을 가능하게 합니다. 비디오 생성 구성 요소는 최신 확산 방법론을 확장한 temporal diffusion 모델을 사용하며, 예측된 모션 제어와 원본 이미지를 모두 입력받아 포토리얼리스틱한 프레임 시퀀스를 생성합니다.

VLOGGER는 temporal outpainting 기법을 통합하여 임의 길이의 비디오를 생성하며, 초기 비디오 출력 후 품질을 정제하기 위한 super-resolution 레이어를 포함합니다.

구조 요약:

```
[입력]
 ├─ 오디오 파형 (audio waveform)
 └─ 참조 이미지 I_ref

[Stage 1: Motion Diffusion Network]
 └─ Audio Feature Extractor → Transformer-based Diffusion
    → 3D 모션 M_{1:T} (얼굴 랜드마크, 포즈, 시선)

[Stage 2: Temporal Image Diffusion]
 ├─ ControlNet 기반 공간 제어 (2D Body Controls)
 ├─ Temporal Attention (시간적 일관성)
 ├─ Identity Reference Encoder (I_ref)
 └─ Super-Resolution Diffusion Model

[출력]
 └─ 고해상도 포토리얼리스틱 비디오 (가변 길이)
```

네트워크는 먼저 단일 프레임에서 새로운 제어 레이어를 학습하는 단계와, 이후 시간적 구성 요소를 추가하여 비디오로 학습하는 두 단계로 훈련됩니다. 이를 통해 첫 번째 단계에서 대형 배치 크기를 사용하고 head reenactment 태스크를 더 빠르게 학습할 수 있으며, 이미지 모델은 학습률 $5 \times 10^{-5}$로 배치 크기 128에서 400k 스텝 동안 두 단계 모두 훈련됩니다.

---

### 2.4 📊 성능 향상

VLOGGER는 3개의 공개 벤치마크에서 이미지 품질, 정체성 보존, 시간적 일관성을 고려하면서 상체 제스처 생성까지 포함하여 최신 방법들을 능가합니다. 복수의 다양성 지표에 대한 성능 분석에서 아키텍처 선택과 MENTOR 사용이 공정하고 편향 없는 대규모 모델 훈련에 기여함을 보입니다.

평가 지표 및 결과:

VLOGGER는 FID(Fréchet Inception Distance), NIQE(Natural Image Quality Evaluator)를 통해 다른 모델들을 능가하는 이미지 품질을 보였으며, LME(Landmark Error)와 jitter 측정값 등 메트릭에서 오디오와 얼굴 동작 간 높은 동기화를 달성합니다. VLOGGER는 인구통계학적 변수의 스펙트럼에 걸쳐 낮은 편향성을 보여, 정체성 보존과 다양한 출력 생성에서 경쟁자들을 능가합니다.

| 지표 | 의미 | VLOGGER 성과 |
|---|---|---|
| **FID↓** | 이미지 품질 (낮을수록 좋음) | SOTA 능가 |
| **LME↓** | 립싱크 정확도 | 높은 동기화 |
| **Jitter↓** | 시간적 일관성 | Temporal outpainting으로 최적화 |
| **NIQE↓** | 자연 이미지 품질 | SOTA 능가 |

---

### 2.5 ⚠️ 한계

VLOGGER는 합성의 소스로 단안(monocular) 입력 이미지만 사용하며, 그럴듯한 합성을 생성할 수 있지만 가려진 부분에 접근할 수 없어 결과 비디오가 해당 인물의 세밀한 분석에서는 사실적이지 않을 수 있습니다. 참조 이미지에 눈이 감긴 상태로 표시될 때와 같이, 추가 데이터로 확산 모델을 파인튜닝함으로써 정체성을 더 잘 포착할 수 있음을 보여줍니다.

추가적인 한계:

- 생성된 비디오는 짧고, 자세히 보면 어색한 부분이 있습니다.
- 제안된 시간적 확산 모델은 고정된 프레임 수 $N$만 생성하도록 훈련되어, 가변 길이 비디오 생성에 대한 직접적인 확장이 자명하지 않습니다.
- 단일 이미지만으로는 가려진 신체 부위의 세밀한 재현이 어려움
- 실시간(real-time) 생성보다는 오프라인 생성에 초점

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 MENTOR 데이터셋을 통한 일반화

논문에서는 MENTOR라는 새롭고 다양한 데이터셋을 구성했으며, 3D 포즈 및 표정 주석을 포함하여 이전 데이터셋보다 한 자릿수 더 큰 800,000개의 정체성과 동적 제스처를 포함하고 있으며, 이를 통해 주요 기술적 기여를 훈련하고 검증합니다.

VLOGGER 개발의 핵심은 800,000개 이상의 다양한 정체성과 총 2,200시간 이상의 비디오를 포함하는 MENTOR 데이터셋입니다. 이 대규모·고정밀 데이터셋으로 훈련함으로써 VLOGGER는 다양한 민족, 나이, 의상, 포즈, 주변 환경의 비디오를 편향 없이 생성할 수 있습니다.

### 3.2 공정성과 무편향 일반화

모델은 대형 사전 훈련된 확산 모델의 사전 지식(priors)과 제안된 대규모 데이터셋을 활용하여, 다른 방법들과 달리 피부 톤, 성별, 나이 등 모든 범주에서 일관된 성능을 보이며 편향이 거의 없습니다. 또한 단단히 크롭된 이미지 대신 다양한 시점(viewpoint)의 이미지에서 인간을 애니메이션할 수 있음을 보여줍니다.

### 3.3 훈련 전략에 의한 일반화

훈련 중 네트워크는 연속 프레임 시퀀스와 인물의 임의 참조 이미지 $I_{ref}$를 입력으로 받으므로, 이론적으로 어떤 비디오 프레임이든 참조로 할당할 수 있습니다. 실제로는 참조를 목표 클립으로부터 (시간적으로) 더 멀리 샘플링하는데, 가까운 예시는 훈련을 너무 쉽게 만들어 일반화 잠재력을 저해하기 때문입니다.

### 3.4 개인화(Personalization)를 통한 일반화 확장

단일 사용자 비디오로 모델을 파인튜닝하면 광범위한 표현에 걸쳐 더욱 정확한 합성을 지원합니다.

이는 기반 모델(foundation model)이 범용 일반화를 담당하고, 개인화 파인튜닝으로 특정 인물에 대한 세밀한 일반화를 추가하는 2단계 일반화 전략을 구현하고 있음을 의미합니다.

### 3.5 일반화 성능 향상 가능성 요약

$$
\text{일반화 성능} = \underbrace{\text{대규모·다양성 데이터}}_{\text{MENTOR}} + \underbrace{\text{사전학습 Diffusion Prior}}_{\text{T2I 모델}} + \underbrace{\text{3D 표현 기반 제어}}_{\text{신체·표정 구조 인식}} + \underbrace{\text{개인화 파인튜닝}}_{\text{선택적 적용}}
$$

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

### 4.1 비교 테이블

| 모델 | 연도 | 주요 특징 | 생성 범위 | 학습 방식 |
|---|---|---|---|---|
| **Wav2Lip** (Prajwal et al.) | 2020 | 립싱크 전문 | 입술 영역만 | 개인별 불필요 |
| **SadTalker** (Zhang et al.) | 2023 | 3DMM 기반 | 얼굴 중심 | 개인별 불필요 |
| **DreamTalk** (Ma et al.) | 2023 | Diffusion 기반 표정 제어 | 얼굴 | 개인별 불필요 |
| **EMO** (Tian et al.) | 2024 | Weak condition 오디오→비디오 | 얼굴+상체 | 개인별 불필요 |
| **VASA-1** (Xu et al., MS) | 2024 | face latent space diffusion | 얼굴 | 개인별 불필요 |
| **VLOGGER** (Corona et al.) | 2024 | 3D 모션 + 전신 확산 | **얼굴+상체+손** | 개인별 불필요 |

VASA-1은 오디오와 정교하게 동기화된 입술 움직임을 생성할 뿐만 아니라 광범위한 표정 미묘함과 자연스러운 머리 동작을 생성할 수 있으며, 핵심 혁신으로 face latent space에서 작동하는 확산 기반 전체적 얼굴 역학 및 머리 움직임 생성 모델과 표현력 있고 해제된(disentangled) 얼굴 latent space의 개발을 포함합니다.

그러나 오디오와 초상화 동작 사이의 약한 상관관계로 인해, end-to-end 오디오 기반 방법들은 합성 비디오의 시간적 안정성을 확보하기 위해 공간적 동작과 관련된 추가 조건을 도입합니다. 얼굴 로케이터 및 속도 레이어 같은 조건들은 초상화 움직임의 범위와 속도를 제한하여 최종 출력의 표현력을 잠재적으로 감소시킵니다.

### 4.2 VLOGGER의 차별점

VLOGGER는 단 하나의 참조 이미지와 오디오 입력만으로 오디오 기반 인간 비디오를 생성하는 새로운 프레임워크입니다. 이 방법은 입술 움직임이나 얼굴 애니메이션만 생성하는 것이 아니라 상체 및 손 제스처도 포함하여 더 전체론적인 커뮤니케이션 표현을 목표로 합니다.

---

## 5. 앞으로의 연구에 미치는 영향과 고려 사항

### 5.1 🚀 연구 영향

**① 전신 Avatar 생성 표준 제시**
머리 동작, 시선, 눈 깜빡임, 입술 움직임은 물론 이전 방법과 달리 상체 및 손 제스처까지 생성함으로써 오디오 기반 합성을 한 단계 더 발전시켰습니다.

**② Embodied Conversational Agent 가능성 제시**
VLOGGER는 오디오 및 애니메이션 시각 표현을 갖춘 체화된 대화 에이전트(embodied conversational agent)에 대한 멀티모달 인터페이스로, 복잡한 표정과 점증하는 신체 동작을 특징으로 하며, 프레젠테이션, 교육, 내레이션, 저대역폭 온라인 커뮤니케이션, 텍스트 전용 HCI의 인터페이스로 활용될 수 있습니다.

**③ 공정성(Fairness) 인식 연구 방향**
연구팀은 MENTOR의 공개가 커뮤니티가 중요한 공정성 문제를 해결하는 데 기여하기를 희망합니다.

**④ 후속 연구 촉발**
GAN 및 확산 모델의 급격한 발전과 함께 인간 비디오 합성은 최근 몇 년간 상당한 주목을 받으며 실용적 활용 임계값에 점차 근접하고 있습니다.

### 5.2 🔬 앞으로 연구 시 고려할 점

**① 실시간 처리 최적화**
VASA-1이 512×512 비디오의 온라인 생성을 지원하는 것과 달리, VLOGGER는 아직 실시간 처리가 명시적으로 지원되지 않습니다. 후속 연구에서는 추론 속도 최적화(예: Consistency Model, 증류 기법 적용)가 핵심 과제입니다.

**② 단일 이미지의 한계 극복**
VLOGGER는 합성 소스로 단안 입력 이미지만을 사용하며, 가려진 부분에 접근할 수 없어 특정 인물의 세밀한 분석에서 완벽하지 않을 수 있습니다. 이를 극복하기 위한 멀티뷰 입력, 3D 재건 통합, NeRF/Gaussian Splatting 기반 표현 결합이 유망한 연구 방향입니다.

**③ 멀티민족·다언어 일반화**
기존 공개 talking-face 비디오 데이터셋은 주로 백인 얼굴로 구성되며 다른 민족에 대해서는 상당한 long-tail 분포를 보이므로, 이러한 데이터셋으로 훈련된 모델은 다른 민족 그룹, 특히 아시아 얼굴을 구동할 때 성능 저하를 보일 수 있습니다.

**④ Deepfake 악용 방지 대책**
VLOGGER는 AI 분야의 급격한 진보를 강력히 보여주는 동시에, 실제와 가짜를 구별하는 데 점점 더 큰 어려움이 발생할 것임을 예고합니다. 따라서 detection, watermarking, 사용 제한 등 책임 있는 AI 활용 방안이 반드시 병행되어야 합니다.

**⑤ 감정 표현의 명시적 제어**
현재 VLOGGER는 오디오로부터 감정을 암묵적으로 추론하지만, EmoTalker, DreamTalk 등의 연구 흐름처럼 감정 레이블이나 텍스트로 감정을 명시적으로 제어하는 방향이 필요합니다.

**⑥ 장기 시퀀스에서의 일관성**
2020년경부터 zero-shot 오디오 기반 초상화 합성 연구가 급증하였으며, 장기(long-term) 시퀀스에서의 모션 일관성 유지는 여전히 핵심 미해결 과제입니다.

---

## 📚 참고 자료 및 출처

| 번호 | 제목 / 출처 |
|---|---|
| 1 | **VLOGGER: Multimodal Diffusion for Embodied Avatar Synthesis** — arXiv:2403.08764, Corona et al. (2024) https://arxiv.org/abs/2403.08764 |
| 2 | **VLOGGER HTML 논문 전문** — https://arxiv.org/html/2403.08764v1 |
| 3 | **VLOGGER Paper PDF** — https://enriccorona.github.io/vlogger/paper.pdf |
| 4 | **VLOGGER 공식 프로젝트 페이지** — https://enriccorona.github.io/vlogger/ |
| 5 | **IEEE Xplore / CVPR 2025 수록본** — https://openaccess.thecvf.com/content/CVPR2025/papers/Corona_VLOGGER |
| 6 | **Hugging Face Papers** — https://huggingface.co/papers/2403.08764 |
| 7 | **VentureBeat 분석 기사** — https://venturebeat.com/ai/google-researchers-unveil-vlogger |
| 8 | **Moonlight Literature Review** — https://www.themoonlight.io/en/review/vlogger-multimodal-diffusion |
| 9 | **VASA-1: Lifelike Audio-Driven Talking Faces** — arXiv:2404.10667, NeurIPS 2024 Oral https://openreview.net/forum?id=5zSCSE0k41 |
| 10 | **Loopy: Taming Audio-Driven Portrait Avatar** — arXiv:2409.02634 https://arxiv.org/html/2409.02634v1 |
| 11 | **CyberHost: Taming Audio-driven Avatar Diffusion Model** — arXiv:2409.01876 https://arxiv.org/html/2409.01876v1 |
| 12 | **Sonic: Shifting Focus to Global Audio Perception** — arXiv:2411.16331 https://arxiv.org/html/2411.16331v1 |
| 13 | **Awesome-Talking-Face (GitHub)** — https://github.com/JosephPai/Awesome-Talking-Face |
| 14 | **GIGAZINE 보도** — https://gigazine.net/gsc_news/en/20240319-google-ai-vlogger |
| 15 | **MM Asia 2025 Challenge** — https://mmasia2025.org/multimodal_multiethnic |
