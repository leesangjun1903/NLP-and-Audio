# UniReal: Universal Image Generation and Editing via Learning Real-world Dynamics

## 1. 핵심 주장과 주요 기여 (요약)

UniReal은 텍스트-이미지 생성, 지시 기반 편집(instructive editing), 객체 커스터마이징, 객체 삽입(composition), 분할/깊이 추정 등 다양한 이미지 task를 **하나의 디퓨전 트랜스포머(5B 파라미터)** 안에서 처리하는 통합 프레임워크입니다. 가장 중요한 발상은 "**입력/출력 이미지를 비디오의 비연속 프레임(discontinuous frames)으로 취급**"하여, Sora류 비디오 생성 모델이 프레임 간 일관성과 변화를 동시에 학습하는 원리를 이미지 task에 그대로 적용한 점입니다.

주요 기여는 다음 네 가지로 정리됩니다. 첫째, 입력 이미지를 *canvas / asset / control* 세 가지 역할로 분리하고 텍스트 토큰("IMG1", "RES1" 등)과 이미지 토큰을 인덱스 임베딩으로 묶는 **텍스트-이미지 연결 메커니즘**을 제안했습니다. 둘째, base prompt + context prompt + image prompt로 구성된 **계층적 프롬프트(hierarchical prompt)**를 통해 다중 task 학습 시 발생하는 모호성을 해소합니다. 셋째, 기존의 task-specific 데이터 큐레이션 방식 대신 **비디오 프레임 쌍을 보편적 supervision 신호로 활용**하는 자동 데이터 구축 파이프라인을 제시했습니다. 넷째, 단일 모델로 instructive editing(EMU Edit / MagicBrush), customized generation(DreamBench), object insertion 등에서 task-specific SOTA 모델들과 동등하거나 더 나은 성능을 달성하면서, 학습된 적 없는 novel application에 대한 emergent ability까지 보입니다.

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조

### 2.1 문제 정의

논문은 "이미지 생성·편집 분야가 task별로 지나치게 파편화되어 있어 (1) 일반화 가능한 지식을 학습하기 어렵고, (2) task별로 별도 모델·데이터를 구축해야 하는 부담이 크다"는 점을 핵심 문제로 지적합니다. 그러나 InstructPix2Pix류 편집, DreamBooth류 커스터마이징, AnyDoor류 객체 삽입은 모두 *입출력 사이의 일관성 유지 + 통제된 시각적 변화*라는 동일한 본질을 공유합니다. 저자들은 이 공통 구조가 비디오 모델의 프레임 간 관계 모델링과 정확히 일치한다고 보고, 비디오 생성기를 이미지 task의 통합 backbone으로 재해석합니다.

### 2.2 모델 구조

전체 파이프라인은 다음과 같이 구성됩니다.

**(a) Latent 인코딩.** $N$개의 입력 이미지 $\{I_i\}_{i=1}^{N}$와 $M$개의 출력 이미지 슬롯이 모두 VAE 인코더 $\mathcal{E}$를 통과해 latent로 변환되고 patchify되어 visual token이 됩니다.

$$
z_i = \text{Patchify}(\mathcal{E}(I_i)), \qquad i = 1,\dots,N
$$

출력 슬롯에는 노이즈 latent $z^{\text{noise}}_j$가 들어갑니다.

**(b) 토큰 합성.** 각 visual token에는 다음 4가지 임베딩이 더해집니다.

$$
\tilde{z}_i = z_i + \text{PE}_i + \text{IndexEmbed}(\text{IMG}_i) + \text{ImgPrompt}(\text{role}_i)
$$

여기서 $\text{role}_i \in \{\text{canvas, asset, control}\}$ 이고, 노이즈 토큰에는 추가로 timestep 임베딩 $\text{TE}(t)$가 더해집니다. *Canvas*는 편집 대상이 되는 배경, *Asset*은 보존해야 할 참조 객체, *Control*은 mask/edge/depth 등 레이아웃 규제 신호 역할을 합니다.

**(c) 텍스트 처리.** Base prompt(예: "put this dog on a grassland"), context prompt(예: "realistic style, with reference object"), image prompt를 합친 텍스트가 T5 인코더로 들어가 텍스트 토큰을 생성합니다. "IMG1", "RES1" 같은 referring word는 T5 토크나이저의 special token으로 등록되어, 같은 인덱스의 이미지 토큰과 학습 가능한 임베딩을 통해 결합됩니다.

**(d) Full attention 트랜스포머.** 모든 텍스트·이미지·노이즈 토큰을 1D 시퀀스로 concat한 뒤 full attention transformer에 통과시킵니다. 표준 어텐션은 다음과 같습니다.

$$
\text{Attn}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

이 full attention 덕분에 각 출력 픽셀이 모든 입력 이미지/텍스트 토큰을 동시에 참조할 수 있어, 다중 입력 task(예: "IMG1의 강아지가 IMG2의 그릇에서 우유를 마신다")가 자연스럽게 처리됩니다.

**(e) 학습 목적함수.** 논문은 Lipman et al. (2023)의 flow matching loss를 그대로 사용한다고만 명시했습니다. 표준 conditional flow matching의 형태는 다음과 같습니다.

$$
x_t = (1-t)\,x_0 + t\,x_1, \qquad x_0 \sim \mathcal{N}(0,I),\; x_1 \sim p_{\text{data}}
$$

$$
\mathcal{L}_{\text{FM}} \;=\; \mathbb{E}_{t,\,x_0,\,x_1,\,c}\!\left[\,\bigl\| v_\theta(x_t, t, c) - (x_1 - x_0) \bigr\|_2^2 \,\right]
$$

여기서 $c$는 입력 이미지·텍스트로 구성된 조건이고, $v_\theta$는 트랜스포머가 예측하는 속도장(velocity field)입니다. 이 형태는 Lipman et al. 2023 및 Diff2Flow의 식(5)에서 그대로 인용한 표준 정의입니다.

### 2.3 데이터 구축

이 부분이 UniReal의 차별화된 핵심입니다. 원본 비디오에서 두 프레임을 무작위로 뽑아 *before/after* 편집 쌍으로 사용하고, 비디오 캡션 모델(grounding caption은 Kosmos-2)과 SAM2로 마스크 트랙릿을 만들어 다음 데이터셋을 자동 구축합니다(논문 Table 1 기준): Video Frame2Frame(8M, instructive editing), Video Multi-object(5M, customization), Video Object Insertion(1M), Video ObjectAdd(1M), Video SEG(5M), Video Control(3M) 등 총 약 23M 비디오 기반 샘플. 여기에 InstructP2P, UltraEdit, VTON-HD, RefCOCO 등 공개 데이터 및 in-house T2I·편집 데이터(약 302M)를 결합합니다.

학습은 256→512→1024 해상도로 점진적으로 진행되며, 각 단계에서 learning rate 1e-5와 warm-up을 사용합니다.

### 2.4 성능 향상 및 한계

정량적으로 UniReal은 EMU Edit 테스트셋에서 $\text{CLIP}\_{\text{dir}} = 0.127$, $\text{CLIP}\_{\text{out}} = 0.285$로 EMU Edit, UltraEdit, OmniGen, PixWizard 대비 최상위 instruction-following 성능을 보였고, MagicBrush에서도 $\text{CLIP}\_{\text{dir}} = 0.151$, $\text{CLIP}\_{\text{out}} = 0.308$, $\text{CLIP}_{\text{im}} = 0.903$으로 거의 모든 핵심 지표에서 1위를 기록했습니다. DreamBench에서는 CLIP-T 0.326으로 OmniGen, SuTI, BootPIG 등 zero-shot customization 모델 중 최고 수준입니다. 다만 입력 이미지와의 유사도(L1, DINO)는 다소 낮은데, 저자들은 "지시를 충실히 따라 큰 변화를 만들기 때문"이라 해석하며 이는 instruction-following과의 trade-off로 볼 수 있습니다.

한계는 명확합니다. 첫째, 입력 이미지 수가 5장을 넘으면 안정성이 떨어지고 계산량이 급증합니다(저자 권장: 3-4장). 둘째, full attention의 quadratic complexity 때문에 고해상도·다중 이미지 task에서 추론 비용이 큽니다. 셋째, 비디오 데이터 단독으로는 stylization 같은 일부 sub-task를 충분히 커버하지 못해 task-specific 데이터를 일부 보강해야 합니다.

---

## 3. 모델의 일반화 성능 향상 가능성 (중점)

UniReal의 일반화 잠재력은 세 층위에서 두드러집니다.

**첫째, Universal supervision으로서의 비디오.** 논문 Fig. 9의 ablation에서, *Video Frame2Frame 데이터만으로 학습한 모델이 add/remove/속성 변경/커스터마이징을 모두 어느 정도 수행*했고, 심지어 **단일 입력 이미지로만 학습되었음에도 다중 입력이 필요한 reference-based object insertion까지 가능**했습니다(저자 표현으로 "not stable" 단서가 붙기는 합니다). 이는 비디오의 자연스러운 시간적 변화 안에 add/remove/속성 변화/구조 변화/물리적 상호작용 등 거의 모든 편집 의미가 내재되어 있다는 강력한 경험적 증거이며, 데이터 확장성(scale-up) 측면에서 task-specific 합성 데이터(InstructPix2Pix류)의 한계를 돌파할 수 있는 길을 제시합니다.

**둘째, 계층적 프롬프트의 task 합성성.** Context prompt가 "task embedding"이 아닌 **공유 가능한 자연어 키워드**로 설계된 점이 일반화에 결정적입니다. 예컨대 "with reference object"라는 키워드는 customization과 object insertion 모두에서 공유되며, 추론 시에는 *학습되지 않은 조합*("with reference object" + "perception task")으로 새로운 동작을 유도할 수 있습니다. 이것이 Fig. 10 우측의 emergent ability — multi-object insertion + pose editing, hairstyle local transfer, layer-aware editing, 이동·크기 조절 등 — 가 가능한 메커니즘으로 보입니다.

**셋째, multi-task 학습이 task-specific 학습보다 더 잘 일반화된다는 정량적 증거.** Table 4에서 *full multi-task 학습 모델이 expert-only 학습 모델보다 거의 모든 지표에서 우세*했습니다(MagicBrush DINO 0.837 vs 0.788, DreamBench CLIP-T 0.326 vs 0.309 등). 저자들은 expert 데이터만으로는 케이스 다양성이 부족하고 합성 데이터의 아티팩트가 누적되는 반면, 비디오 기반 realistic 데이터가 이를 보완한다고 분석합니다. 이는 **"이질적 task가 서로의 부족한 케이스를 보완하여 일반화 성능을 끌어올린다"**는 멀티태스크 학습의 고전적 가설을 이미지 생성 분야에서 다시 한 번 강하게 뒷받침합니다.

다만 일반화의 한계도 분명합니다. (1) 비디오에는 의도적 stylization 변화가 거의 없으므로 watercolor·sketch 변환 같은 task는 여전히 expert 데이터가 필요합니다. (2) Kosmos-2/SAM2/GPT-4o mini 등 자동 라벨링 모델의 오류가 데이터에 누적되어, 정밀한 instruction-following 학습에 노이즈로 작용할 수 있습니다. (3) Full attention의 비용 때문에 입력 이미지 수가 늘어나는 long-horizon 일반화에는 구조적 제약이 존재합니다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

UniReal은 2024년 후반에 등장한 일련의 "통합 이미지 생성·편집 모델" 흐름의 한 분기점에 위치합니다. 비교 대상은 다음과 같습니다.

| 모델 | 발표 | 통합 방식 | 입력 처리 | 데이터 전략 | UniReal과의 차이 |
|---|---|---|---|---|---|
| InstructPix2Pix (Brooks et al., CVPR 2023) | 2023 | Single-task (instructive editing) | 1 source + prompt | GPT-3 + Stable Diffusion으로 합성 | UniReal: 비디오 프레임 쌍을 자연 supervision으로 사용 |
| DreamBooth (Ruiz et al., CVPR 2023) | 2023 | Single-task (customization) | 3-5장 reference + finetuning | per-subject finetune 필요 | UniReal: zero-shot, finetuning 불필요 |
| AnyDoor (Chen et al., CVPR 2024) | 2024 | Single-task (object insertion) | reference+target+mask 필수 | object-level paired data | UniReal: mask 불필요, 자동 pose/조명 적응 |
| Emu Edit (Sheynin et al., CVPR 2024) | 2024 | Multi-task editing (single image) | 1 source + task embedding | task-labeled 합성 데이터 | UniReal: 다중 입력 이미지, 자연 영상 supervision |
| OmniGen (Xiao et al., 2024.09) | 2024 | Unified gen+edit | interleaved text/image, causal+bidirectional attention | mixed task data | UniReal: 비디오 기반 dynamics 학습, video-as-frames 관점 |
| ACE (Han et al., 2024.10) | 2024 | Unified gen+edit (LCU 단위) | Long-context Condition Unit (텍스트+이미지+마스크) | MLLM으로 instruction 라벨링 | UniReal: image role(canvas/asset/control) 명시적 분리 + 비디오 supervision |
| PixWizard (Lin et al., 2024.09) | 2024 | Unified vision tasks | task embedding 기반 | open-language instruction | UniReal: task embedding 대신 공유 가능한 자연어 keyword |
| Show-o / Transfusion / SEED-X / Emu2 (2024) | 2024 | Multimodal understanding+gen | autoregressive 또는 hybrid token | 이해 중심 | UniReal: 생성·편집 품질에 집중, 이해는 부수적 |

이 비교에서 드러나는 UniReal의 고유 위치는 두 가지입니다. 첫째, **비디오를 supervision으로 끌어온 첫 번째 본격 시도**라는 점입니다. OmniGen과 ACE 모두 task별 데이터를 모아 합치는 방식인 데 반해, UniReal은 "시간 변화가 자연스러운 편집 신호다"라는 강한 가정에서 출발해 8M+ 비디오 쌍을 자동 생성합니다. 둘째, **입력 이미지의 역할을 의미적으로 명시 분리(canvas/asset/control)** 하는 설계입니다. ACE의 LCU는 "텍스트+이미지+마스크"를 하나의 단위로 묶지만, UniReal은 각 이미지가 어떤 의미적 역할을 하는지를 학습 가능한 임베딩으로 표시하여 multi-image task에서의 모호성을 줄입니다.

학습 알고리즘 측면에서는 ACE, FLUX 기반 ACE++(2025.01) 등이 UniReal과 마찬가지로 flow matching $\mathcal{L}_{\text{FM}}$ 또는 rectified flow 계열을 채택하는 추세이며, "diffusion weighting of v-MSE + cosine schedule"이 flow matching weighting과 등가라는 최근 결과(Diffusion Meets Flow Matching, 2024)에 따라 두 패러다임이 점점 수렴하고 있습니다.

---

## 5. 향후 연구에 미치는 영향과 고려할 점

**영향 측면.** UniReal이 제시한 두 가지 발상 — *비디오 = 보편적 편집 supervision*, *이미지 task = 비연속 프레임 생성* — 은 후속 연구의 데이터 패러다임을 바꿀 가능성이 큽니다. 이미 VIVID-10M(2024) 같은 비디오 기반 편집 데이터셋, Diffusion Self-Distillation(2024) 같은 zero-shot customization 연구가 비슷한 방향으로 등장하고 있습니다. 또한 image role embedding(canvas/asset/control) 개념은 ACE++의 context-aware content filling 등에서 변형된 형태로 채택되는 흐름이 보입니다. 더 넓게는, 비디오 생성 backbone(Sora, CogVideoX, Movie Gen 등)이 이미지 task의 사실상 표준 백본이 되는 *수렴 현상*이 가속화될 것으로 보입니다.

**향후 연구 시 고려할 점.** 첫째, **자동 라벨링 노이즈의 정량적 영향 분석**이 필요합니다. UniReal은 GPT-4o mini, Kosmos-2, SAM2 등의 출력을 그대로 학습 신호로 쓰는데, 이 라벨이 instruction-following 정밀도의 상한선을 결정할 수 있습니다. 둘째, **5장 이상 입력에서의 안정성 문제**를 풀려면 full attention을 대체하는 효율적 구조(예: sparse attention, grouped attention, MoE)가 필요합니다. 셋째, **stylization, fine-grained 텍스트 렌더링, 정밀한 조명·물리 시뮬레이션** 등 비디오 데이터에 자연 분포하지 않는 task는 여전히 별도 supervision이 필요하므로, 어떤 task는 비디오로 충분하고 어떤 task는 보강이 필요한지에 대한 *체계적 task taxonomy* 연구가 가치가 큽니다. 넷째, 비디오에서 추출된 두 프레임은 종종 캡션이 포착하지 못한 글로벌 변화(카메라 모션, 배경 이동)를 포함하므로, 본 논문이 도입한 "static/dynamic scenario" context prompt 같은 *분포 라벨링 기법*을 더 정교하게 자동화하는 연구가 후속 가치가 있습니다. 다섯째, 평가 측면에서 "instruction-following을 잘하면 source-target 유사도가 떨어지는" trade-off는 기존 metric(L1, DINO, CLIP-I)이 universal model 평가에 부적절함을 시사하며, 새로운 universal evaluation protocol이 요구됩니다.

마지막으로 한 가지 신중한 단서를 덧붙이자면, 본 논문은 모델 코드와 가중치가 공개되지 않은 상태에서 평가된 결과이므로(2024년 12월 v2 기준) — Hugging Face 페이지에도 코드 공개 여부에 대한 질문이 게시되어 있습니다 — Table 2-4의 정량 수치는 제3자 재현으로 검증되기 전까지는 저자 보고 값으로 받아들이는 것이 안전합니다. 또한 본 답변에서 명시한 flow matching loss 수식과 attention 수식은 논문이 직접 적은 식이 아니라 논문이 인용한 표준 정의(Lipman et al. 2023)를 그대로 표기한 것임을 밝힙니다.

---

## 참고자료

논문 원문: Chen, X., Zhang, Z., Zhang, H., et al. "UniReal: Universal Image Generation and Editing via Learning Real-world Dynamics." arXiv:2412.07774v2, 2024년 12월 11일.

비교 분석에 활용한 자료:
- arXiv 페이지: https://arxiv.org/abs/2412.07774
- 프로젝트 페이지: https://xavierchen34.github.io/UniReal-Page/
- HTML 전문: https://arxiv.org/html/2412.07774v1
- Hugging Face 논문 페이지(코드 공개 관련 댓글 포함): https://huggingface.co/papers/2412.07774
- Moonlight Literature Review: https://www.themoonlight.io/en/review/unireal-universal-image-generation-and-editing-via-learning-real-world-dynamics
- ChatPaper 요약: https://www.chatpaper.ai/paper/f5712328-6768-4f68-a40b-fb6bc3c51850
- OmniGen (비교): Xiao et al., arXiv:2409.11340 (https://arxiv.org/abs/2409.11340)
- ACE (비교): Han et al., arXiv:2410.00086 (https://arxiv.org/abs/2410.00086)
- ACE++ (후속 흐름): Mao et al., arXiv:2501.02487 (https://ali-vilab.github.io/ACE_plus_page/)
- Flow Matching 수식 표준 정의: Lipman et al., ICLR 2023 (논문 [30]에서 인용); Diff2Flow CVPR 2025 paper Eq. (5) (https://www.openaccess.thecvf.com/content/CVPR2025/papers/Schusterbauer_Diff2Flow_Training_Flow_Matching_Models_via_Diffusion_Model_Alignment_CVPR_2025_paper.pdf)
- "Diffusion Meets Flow Matching" 등가성 분석: https://diffusionflow.github.io/
- 논문 내부 인용 핵심: InstructPix2Pix [1], MagicBrush [69], EMU Edit [48], DreamBooth [47], AnyDoor [10], UltraEdit [72], PixWizard [29], OmniGen [60], ACE [15], Kosmos-2 [39], SAM2 [45], T5 [44], Lipman et al. flow matching [30].
