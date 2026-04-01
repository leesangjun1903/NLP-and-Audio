# EMO: Emote Portrait Alive -- Generating Expressive Portrait Videos with Audio2Video Diffusion Model under Weak Conditions

---

## 1. 핵심 주장 및 주요 기여 요약

EMO는 오디오 신호와 얼굴 움직임 사이의 동적이고 미묘한 관계에 초점을 맞추어 Talking Head 비디오 생성의 사실성과 표현력을 향상시키는 것을 목표로 한다. 이를 위해 중간 3D 모델이나 얼굴 랜드마크 없이 오디오에서 비디오로 직접 합성하는 새로운 프레임워크 EMO를 제안한다.

**주요 기여:**
1. 디퓨전 모델의 생성 능력을 활용하여 단일 이미지와 오디오 클립으로부터 직접 캐릭터 헤드 비디오를 합성하며, 중간 표현이나 복잡한 전처리 없이 시각적·감정적 충실도가 높은 Talking Head 비디오를 생성한다.
2. 오디오-표정 매핑의 모호성으로 인한 불안정성을 완화하기 위해 Speed Controller와 Face Region Controller라는 안정적 제어 메커니즘을 통합하였다.
3. 실험 결과 EMO는 설득력 있는 말하기 비디오뿐만 아니라 다양한 스타일의 노래 비디오도 생성할 수 있으며, 기존 SOTA 방법론들을 표현력과 사실성 면에서 크게 능가하였다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 기법들은 인간 표정의 전체 스펙트럼과 개인 얼굴 스타일의 고유성을 포착하는 데 종종 실패한다. 특히 3D 메쉬 기반 방법들은 표현 용량이 제한되어 생성 비디오의 전반적 표현력과 사실성이 제한된다. 오디오 신호와 얼굴 표정 간의 매핑에 내재된 모호성은 생성 비디오에서 얼굴 왜곡이나 프레임 간 불일치를 초래할 수 있다.

### 2.2 제안 방법 및 수식

EMO는 Stable Diffusion 기반의 디퓨전 모델을 활용한다. 디퓨전 모델의 기본 학습 목적 함수는 DDPM의 단순화된 노이즈 예측 손실을 따른다:

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{x_0, \epsilon, t} \left[ \left\| \epsilon - \epsilon_\theta(x_t, t) \right\|^2 \right]
$$

여기서:
- $x_0$: 원본 데이터(비디오 프레임의 잠재 표현)
- $\epsilon \sim \mathcal{N}(0, \mathbf{I})$: 순방향 과정에서 추가된 가우시안 노이즈
- $x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon$: 시간 $t$에서의 노이즈가 가해진 잠재 표현
- $\epsilon_\theta$: 학습 가능한 디노이징 네트워크 (Backbone UNet)

EMO에서는 이 기본 목적 함수에 **오디오 조건 $a$**, **참조 이미지 특징 $f_{\text{ref}}$**, 그리고 **약한 조건 신호(Face Region Mask $m$, Speed Signal $s$)** 가 추가 조건으로 들어간다:

$$
\mathcal{L}_{\text{EMO}} = \mathbb{E}_{x_0, \epsilon, t, a, f_{\text{ref}}, m, s} \left[ \left\| \epsilon - \epsilon_\theta\!\left(x_t, t, a, f_{\text{ref}}, m, s\right) \right\|^2 \right]
$$

**"Weak Conditions"의 의미:**
지정된 얼굴 영역과 할당된 속도는 강한 제어 조건을 구성하지 않는다. Face Locator의 경우, 전체 비디오 클립의 합집합 영역이므로 얼굴 움직임이 허용되는 상당히 넓은 영역을 나타내며, 헤드가 정적 자세로 제한되지 않는다. Speed Layers의 경우, 인간 머리 회전 속도의 정확한 추정이 어려워 예측된 속도 시퀀스가 본질적으로 노이즈를 포함한다.

### 2.3 모델 구조

프레임워크는 두 단계로 구성된다. 초기 Frames Encoding 단계에서 ReferenceNet이 참조 이미지와 모션 프레임으로부터 특징을 추출한다. 이후 Diffusion Process 단계에서 사전학습된 오디오 인코더가 오디오 임베딩을 처리하고, 얼굴 영역 마스크가 다중 프레임 노이즈와 통합되어 얼굴 영상 생성을 지배한다. Backbone Network에서 Reference-Attention과 Audio-Attention 두 가지 어텐션 메커니즘이 적용되며, Temporal Module이 시간 차원을 조작하고 모션 속도를 조절한다.

#### 주요 구성 요소:

| 구성 요소 | 설명 |
|-----------|------|
| **Backbone Network** | SD 1.5와 유사한 UNet 구조로, 다중 프레임 노이즈 잠재 입력을 받아 각 타임스텝에서 연속 비디오 프레임으로 디노이즈한다. |
| **ReferenceNet** | Backbone Network와 동일한 구조를 가지며 입력 이미지로부터 상세 특징을 추출한다. 동일한 SD 1.5 UNet 아키텍처에서 유래했으므로, 두 구조에서 생성된 특징 맵이 유사성을 보여 Backbone이 ReferenceNet 특징을 통합하기 용이하다. |
| **Audio Encoder** | 사전학습된 wav2vec 네트워크를 사용하여 오디오 특징을 추출하고, 이를 연결하여 최종 오디오 표현 임베딩을 생성한다. |
| **Face Locator** | 얼굴 바운딩 박스 영역을 포함하는 마스크를 활용하며, 경량 합성곱 레이어로 바운딩 박스 마스크를 인코딩하여 노이즈 잠재 표현에 추가한 후 Backbone에 입력한다. |
| **Temporal Module** | AnimateDiff의 아키텍처 개념을 적용하여 프레임 간 특징에 self-attention temporal layer를 적용한다. |
| **Speed Layers** | 말하는 캐릭터의 모션을 제어 가능하고 안정적으로 만들기 위한 약한 조건을 제공한다. |

#### 어텐션 메커니즘:
- **Reference-Attention**: 대상 캐릭터의 이미지를 ReferenceNet에 입력하여 self-attention 레이어에서 참조 특징 맵을 추출하고, Backbone 디노이징 과정에서 대응 레이어의 특징이 추출된 특징 맵과 Reference-Attention 레이어를 통해 상호작용한다.
- **Audio-Attention**: 음성 특징을 생성 절차에 주입하기 위해, Backbone Network의 각 Reference-Attention 레이어 뒤에 잠재 코드와 오디오 특징 간 교차 어텐션을 수행하는 Audio-Attention 레이어를 추가한다.

#### 학습 전략:
학습 시 Temporal 부분은 AnimateDiff에서 가중치를 초기화한다. 1단계에서 Backbone은 단일 프레임을 입력받고 ReferenceNet은 결과 비디오에서 무작위로 선택된 프레임을 처리한다. 2단계에서 Temporal 모듈과 오디오 모듈이 도입되며, 생성된 비디오 클립에서 사전 정의된 수의 프레임이 샘플링된다.

### 2.4 성능 향상

HDTF 데이터셋에서 광범위한 실험 및 비교를 수행하였으며, DreamTalk, Wav2Lip, SadTalker 등 현재 SOTA 방법들을 FID, SyncNet, F-SIM, FVD 등 다양한 메트릭에서 능가하였다. 정량적 평가 외에도 포괄적인 사용자 연구 및 정성적 평가를 수행하여 EMO 방법이 매우 자연스럽고 표현력 있는 말하기 및 노래 비디오를 생성할 수 있음을 확인하였다.

**학습 데이터:**
EMO 모델 학습을 위해 250시간 이상의 영상과 1억 5천만 개 이상의 이미지로 구성된 방대하고 다양한 오디오-비디오 데이터셋이 구축되었으며, 이는 다양한 콘텐츠 유형과 언어적 다양성을 포함한다.

### 2.5 한계

디퓨전 모델 기반이 아닌 방법들에 비해 생성 과정이 상대적으로 시간이 많이 소요되며, 비얼굴 신체 부위에 대한 명시적 제어 신호의 부재로 인해 아티팩트가 생성될 수 있다. 모델의 성능이 학습 데이터의 품질과 다양성에 민감할 수 있으며, 데이터셋이 다양한 감정 표현이나 오디오-비주얼 상관관계를 충분히 포착하지 못하면 의도된 감정 톤을 완전히 반영하지 못할 수 있다. 또한 현재 버전은 단일 개인의 초상화 비디오 생성에 초점을 맞추고 있어, 다중 화자나 더 복잡한 장면으로의 확장이 향후 연구 과제이다.

---

## 3. 모델의 일반화 성능 향상 가능성

EMO의 일반화 능력과 관련된 핵심 요소들은 다음과 같다:

### 3.1 중간 표현 우회를 통한 일반화
EMO 프레임워크는 디퓨전 모델의 생성 능력을 활용하여 이미지와 오디오 클립으로부터 직접 캐릭터 헤드 비디오를 합성하며, 중간 표현이나 복잡한 전처리가 필요 없다. 이는 3D 메쉬나 랜드마크 등 특정 표현에 의존하지 않으므로, 다양한 스타일의 초상화(회화, 3D 모델, AI 생성 콘텐츠)에 대해 더 나은 일반화가 가능하다.

### 3.2 다양한 입력 스타일 지원
EMO의 접근법은 노래 오디오 입력뿐 아니라 다양한 언어의 음성 오디오도 처리할 수 있으며, 과거 시대의 초상화, 회화, 3D 모델 및 AI 생성 콘텐츠에 생동감 있는 모션과 사실성을 부여할 수 있다.

### 3.3 약한 조건(Weak Conditions)의 활용
약한 조건의 핵심 이점은 모델에 과도한 제약을 부여하지 않아, 다양한 헤드 포즈와 표정을 자유롭게 생성할 수 있다는 점이다. 이는 아래와 같이 정리된다:

- **Face Locator**: 비디오 클립 전체의 바운딩 박스 합집합으로 넓은 영역 확보 → 다양한 헤드 모션 허용
- **Speed Layers**: 노이즈가 포함된 속도 추정을 사용하여 과적합 방지

### 3.4 사전학습 가중치 전이
ReferenceNet과 Backbone Network 모두 원래 SD UNet에서 가중치를 상속받는다. 이로 인해 Stable Diffusion이 대규모 이미지 데이터셋에서 학습한 풍부한 시각적 사전 지식을 활용할 수 있어 일반화 성능이 크게 향상된다.

### 3.5 향후 일반화 개선을 위한 가능성
- 비얼굴 신체 부위에 대한 제어 신호를 통합하여 더 세밀한 신체 부위 조작을 가능하게 하는 것이 향후 연구 방향이다.
- 더 다양한 언어, 감정, 문화적 맥락의 데이터 확대
- DiT(Diffusion Transformer) 등 최신 아키텍처로의 전환을 통한 스케일링

---

## 4. 연구 영향 및 향후 연구 시 고려할 점

### 4.1 연구에 미치는 영향

EMO 프레임워크는 오디오 입력으로부터 표현력 있는 초상화 비디오를 생성하는 분야에서 중요한 발전을 대표하며, 디퓨전 모델의 역량과 혁신적 제어 메커니즘을 활용하여 높은 수준의 사실성과 표현력을 달성하고 Talking Head 비디오 생성 분야의 새로운 벤치마크를 수립하였다.

EMO 이후의 영향:
- **패러다임 전환**: 3D 모델 기반 → 직접 Audio-to-Video 디퓨전 방식이 주류로 자리잡는 데 기여
- **후속 연구 촉발**: Hallo, V-Express, EchoMimic, AniPortrait 등 수많은 후속 연구가 EMO의 프레임워크를 참조하거나 발전시킴
- **응용 확장**: 가상 비서, 애니메이션, 멀티미디어 창작, 다국어/다문화 콘텐츠 제작 등

### 4.2 향후 연구 시 고려할 점

| 고려 사항 | 상세 내용 |
|-----------|-----------|
| **추론 속도** | 디퓨전 기반 방식의 느린 생성 속도 → 증류(distillation), 비자기회귀(non-autoregressive) 방식 등 연구 필요 |
| **신체 부위 제어** | 얼굴 이외 상체/손 등의 자연스러운 움직임 생성을 위한 명시적 제어 |
| **장기 시간적 일관성** | 긴 비디오에서의 ID 보존 및 시간적 일관성 (Hallo2 등에서 다뤄짐) |
| **다중 화자 지원** | 단일 인물 제한 → 다중 화자 상호작용 장면 |
| **윤리적 고려** | 딥페이크 악용 방지를 위한 감지 기술 및 윤리 가이드라인 필요 |
| **해상도 확장** | 고해상도(1024×1024 이상) 비디오 생성 품질 향상 |
| **감정 제어성** | 오디오에서 자동 추출되는 감정 외, 사용자 지정 감정 제어 기능 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 접근 | EMO 대비 특징 |
|------|------|-----------|---------------|
| **Wav2Lip** (Prajwal et al.) | 2020 | GAN 기반 입술 동기화 | 입술 동기화에 특화되나 전체 얼굴 표현력 제한 |
| **SadTalker** (Zhang et al.) | 2023 (CVPR) | 3DMM 계수 기반 + ExpNet | 3D 메쉬의 제한된 표현 용량으로 표현력과 사실성이 제약됨 |
| **DreamTalk** (Ma et al.) | 2023 | 디퓨전으로 3DMM 계수 생성 | 디퓨전 모델을 사용하지만 이미지 프레임에 직접 적용하지 않고 3DMM 계수 생성에 사용하여 자연스러운 얼굴 비디오 생성에 한계 |
| **DiffTalk** (Shen et al.) | 2023 (CVPR) | 디퓨전 기반 초상화 애니메이션 | 오디오 기반 디퓨전이나 얼굴 일부에 집중 |
| **EMO** (Tian et al.) | 2024 (ECCV) | 직접 Audio→Video 디퓨전 | 중간 표현 없이 종단간 생성, 약한 조건 제어 |
| **AniPortrait** (Wei et al.) | 2024 | 오디오 → 랜드마크 → 비디오 2단계 | 랜드마크 기반이므로 중간 표현 의존 |
| **VASA-1** (Microsoft) | 2024 | 얼굴 잠재 공간에서 작동하는 디퓨전 기반 holistic facial dynamics 및 헤드 움직임 생성 모델, 표현적이고 분리된 얼굴 잠재 공간 개발 | 실시간 생성 가능, 분리된 잠재 공간으로 제어성 우수 |
| **Hallo** (Xu et al.) | 2024 | 계층적 오디오 기반 시각 합성 | EMO와 유사한 디퓨전 파이프라인, 오픈소스 공개 |
| **Hallo2** (Cui et al.) | 2024 (ICLR 2025) | 장시간/고해상도 초상화 애니메이션 | EMO의 한계인 장기 비디오 생성 문제 해결 시도 |
| **V-Express** (Wang et al.) | 2024 | 조건부 드롭아웃 기반 점진적 학습 | 약한 조건 학습의 안정화 기법 |
| **EchoMimic** (Chen et al.) | 2024 (AAAI 2025) | 편집 가능한 랜드마크 조건 | 랜드마크를 통한 세밀한 제어와 자연스러움 균형 |
| **LivePortrait** (2024) | 2024 | Stitching & Retargeting 제어 | 효율적 추론에 초점 |
| **X-Portrait** (2024) | 2024 | 계층적 모션 어텐션을 활용한 조건부 디퓨전 모델로 표현적이고 시간적으로 일관된 초상화 애니메이션 생성 | 모션 전이 기반 접근 |
| **HunyuanPortrait** | 2025 (CVPR) | 암묵적 조건 제어 | 최신 대규모 모델 기반 |
| **OmniHuman-1** | 2025 | 전신 애니메이션 스케일링 | EMO의 얼굴 한정 → 전신으로 확장 |

### 핵심 비교 축 분석:

**1) 중간 표현 의존도**
- EMO, Hallo, VASA-1: 최소화 (직접 생성 또는 잠재 공간 활용)
- SadTalker, DreamTalk, AniPortrait: 3DMM/랜드마크 등 중간 표현 의존

**2) 생성 품질 vs. 속도 트레이드오프**
- VASA-1: 실시간 생성 가능 (512×512, 40 FPS)
- EMO: 높은 품질이나 느린 추론

**3) 제어성**
- EMO: 약한 조건을 통한 암묵적 제어
- VASA-1: 분리된 잠재 공간을 통한 명시적 제어
- EchoMimic: 편집 가능한 랜드마크를 통한 제어

---

## 참고자료 및 출처

1. **Tian, L., Wang, Q., Zhang, B., Bo, L.** (2024). "EMO: Emote Portrait Alive — Generating Expressive Portrait Videos with Audio2Video Diffusion Model under Weak Conditions." *arXiv:2402.17485* / *ECCV 2024, LNCS vol 15141, Springer*. — [arxiv.org/abs/2402.17485](https://arxiv.org/abs/2402.17485), [arxiv.org/html/2402.17485v1](https://arxiv.org/html/2402.17485v1)
2. **EMO 프로젝트 페이지**: [humanaigc.github.io/emote-portrait-alive](https://humanaigc.github.io/emote-portrait-alive/)
3. **EMO GitHub 리포지토리**: [github.com/HumanAIGC/EMO](https://github.com/HumanAIGC/EMO)
4. **Springer 출판본**: [doi.org/10.1007/978-3-031-73010-8_15](https://link.springer.com/chapter/10.1007/978-3-031-73010-8_15)
5. **Metaphysic.ai 분석 기사**: [blog.metaphysic.ai](https://blog.metaphysic.ai/plausible-stable-diffusion-video-from-a-single-image/)
6. **Emergent Mind 분석**: [emergentmind.com/papers/2402.17485](https://www.emergentmind.com/papers/2402.17485)
7. **AIModels.fyi 분석**: [aimodels.fyi/papers/arxiv/emo-emote-portrait-alive](https://www.aimodels.fyi/papers/arxiv/emo-emote-portrait-alive-generating-expressive-portrait)
8. **GPT Review Portal 분석**: [gptreviewportal.com/emote-portrait-alive](https://gptreviewportal.com/emote-portrait-alive/)
9. **VASA-1 논문**: [arxiv.org/html/2404.10667v2](https://arxiv.org/html/2404.10667v2)
10. **Hallo GitHub**: [github.com/fudan-generative-vision/hallo](https://github.com/fudan-generative-vision/hallo)
11. **Hallo2 GitHub**: [github.com/fudan-generative-vision/hallo2](https://github.com/fudan-generative-vision/hallo2)
12. **Awesome Talking Head Generation**: [github.com/harlanhong/awesome-talking-head-generation](https://github.com/harlanhong/awesome-talking-head-generation)
13. **ACM Computing Surveys — Talking Head Synthesis Survey**: [dl.acm.org/doi/10.1145/3785656](https://dl.acm.org/doi/10.1145/3785656)
14. **Emergent Mind — Audio-Driven Talking Head Generation Topic**: [emergentmind.com/topics/audio-driven-talking-head-generation](https://www.emergentmind.com/topics/audio-driven-talking-head-generation-ad-thg)
15. **Analytics Vidhya EMO 분석**: [analyticsvidhya.com/blog/2024/02/emo-ai-by-alibaba](https://www.analyticsvidhya.com/blog/2024/02/emo-ai-by-alibaba-an-audio-driven-portrait-video-generation-framework/)
16. **Ho, J., Jain, A., Abbeel, P.** (2020). "Denoising Diffusion Probabilistic Models." *NeurIPS 2020.*

> **참고**: 본 분석에서 EMO 논문의 내부 수식은 논문 원문의 디퓨전 모델 학습 목적 함수와 조건부 확장을 기반으로 재구성하였습니다. 논문의 세부 구현 수식(예: temporal module의 정확한 feature reshape 수식, speed layer의 정확한 encoding 방식 등)은 원문 PDF의 직접 확인을 권장합니다.
