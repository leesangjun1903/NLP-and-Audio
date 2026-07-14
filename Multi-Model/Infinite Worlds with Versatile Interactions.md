# Infinite Worlds with Versatile Interactions (LingBot-World 2.0 / LingBot-World-Infinity)

---

## 1. 핵심 주장과 주요 기여 요약

이 논문은 Ant Group의 Robbyant 팀이 발표한 **LingBot-World 2.0**(일명 **LingBot-World-Infinity**)을 제시합니다. 이 모델은 LingBot-World의 발전된 버전으로, 네 가지 핵심 개선점을 특징으로 합니다. 첫째, 신중하게 설계된 causal pretraining paradigm을 통해 일관된 출력 품질을 유지하면서 무한한 상호작용 지평(unbounded interaction horizon)을 달성합니다. 둘째, 기본 모델로부터 실시간 변형 모델을 증류하여 720p 비디오 스트림을 60fps로 구동할 수 있는 빠른 응답 시간을 보장합니다.

셋째, 이전 버전과 비교해 공격, 궁술, 마법 시전, 사격 등 더 넓은 범위의 행동과 더 풍부한 텍스트 기반 이벤트를 포함하는 매우 다양한 상호작용 요소를 도입했습니다. 넷째, 파일럿 에이전트가 캐릭터 행동을 계획·실행하고 디렉터 에이전트가 장면 진행에 따라 새로운 환경 요소를 합성하는, 월드 모델링 분야 최초의 agentic harness 통합을 시도했습니다. 또한 여러 플레이어가 동시에 세계에 몰입할 수 있는 공유 경험 인터페이스를 개발했으며, 14B 규모의 주 모델과 단일 GPU 배포가 가능한 경량 1.3B 모델을 함께 제공합니다.

핵심 주장은 **"지속성(durability)"**입니다. LingBot-World-Infinity는 인터랙티브 월드 모델링을 위한 개방형 causal video generation 모델로서, 최첨단 시각적 품질과 드리프트(drift)에 대한 강한 저항성을 결합하며, 이로부터 720p·60fps로 무한하고 드리프트 없는 세계를 유지하는 실시간 모델을 증류합니다.

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능, 한계

### (1) 해결하려는 문제
이 연구팀이 공략하는 두 가지 실패 모드는 장기 지평 드리프트(long-horizon drift)와 상호작용 지연(interactive latency)입니다. 인터랙티브 월드 모델은 사용자 행동 스트림에 조건화되어 프레임 단위로 비디오를 생성하며, 각 상태는 과거 프레임과 현재 입력에만 의존합니다.

기존 비디오 기반 월드 모델들은 자기회귀적(autoregressive) 롤아웃이 길어질수록 오차가 누적되어 화면이 붕괴하는 "드리프트" 문제와, 고품질 프레임 생성을 위한 다단계 디노이징(denoising) 과정으로 인한 느린 응답 속도라는 두 가지 근본적 난제를 안고 있었습니다.

### (2) 제안하는 방법 (수식 포함)

**① 인과적 분해(Causal Factorization)**

연구팀은 이를 인과적 분해로 형식화합니다. 여기서 $x_t$는 시간 $t$에서의 시각적 상태이며, 행동 $a_t$는 카메라 포즈와 텍스트 프롬프트를 결합합니다. 카메라 포즈는 Plücker embedding을 사용하여 adaptive layer normalization(AdaLN)을 통해 주입되고, 텍스트는 chunk-wise 프롬프트로 cross-attention을 통해 입력됩니다.

이를 일반적인 형태로 표현하면 다음과 같습니다:

$$p(x_{1:T} \mid a_{1:T}) = \prod_{t=1}^{T} p(x_t \mid x_{ < t},\, a_{\le t})$$

여기서 각 상태 $x_t$의 생성은 오직 과거 프레임 $x_{ < t}$와 현재까지의 행동 $a_{\le t}$에만 의존하는 엄격한 인과적 구조를 가집니다.

**② 3단계 학습 파이프라인**

연구팀은 기초 비디오 생성기를 인터랙티브 월드 시뮬레이터로 전환하기 위한 다단계 진화 전략을 제안합니다. Pre-training 단계는 고신뢰도 개방형 도메인 생성을 보장하는 견고한 일반 비디오 프라이어를 확립하고, Middle-training 단계는 세계 지식과 행동 제어 가능성을 주입하여 모델이 일관된 상호작용 논리로 장기 동역학을 시뮬레이션할 수 있게 하며, Post-training 단계는 실시간 상호작용을 위해 아키텍처를 적응시켜 causal attention과 few-step distillation을 활용해 낮은 지연시간과 엄격한 인과성을 달성합니다.

**③ Block-Causal Attention**

연구팀은 전체 시간적 attention을 block causal attention으로 대체하여, 청크(chunk) 내부의 지역적 양방향 의존성과 청크 간 전역적 인과성을 결합합니다. 이 모델은 Stage II의 high-noise expert로부터 초기화되며, 전문가 특화를 연결하기 위해 mixed-timestep 프로토콜로 학습됩니다. 이는 시간적 일관성을 유지하면서 KV 캐싱을 통한 효율적인 자기회귀적 생성을 가능하게 합니다.

**④ Few-Step Distillation (Consistency Distillation + DMD)**

Post-training 단계에서는 사전학습된 multi-step causal diffusion world model을 실시간 상호작용에 적합한 few-step generator로 압축하는 동시에, 장기 자기회귀적 롤아웃에서 발생하는 오차 누적을 완화합니다. 이를 위해 디노이징 단계 수를 줄이는 consistency distillation과, 충실도를 개선하고 롤아웃 드리프트를 억제하는 distribution matching distillation(DMD)을 결합합니다.

사전학습된 teacher는 고품질 프레임을 생성하지만 각 프레임 생성에 많은 디노이징 단계가 필요해 인터랙티브 사용에는 비용이 지나치게 크므로, teacher를 consistency model $G_\theta$로 증류하며, 이는 중간 노이즈 latent로부터 궤적 불변(trajectory-invariant) 타겟을 예측합니다.

DMD의 핵심 원리는 다음과 같은 KL 발산 최소화로 표현됩니다:

$$\mathcal{L}_{\text{DMD}} = \mathbb{E}_{x \sim p_{\theta}}\left[ D_{KL}\big(p_{\theta}(x_t) \,\|\, p_{\text{data}}(x_t)\big) \right]$$

여기서 생성기(generator)는 노이즈가 추가된 student 분포와 노이즈가 추가된 data 분포 사이의 KL 기울기를 따르며, 중요한 세부사항은 이를 teacher-forced 상태뿐 아니라 장기 자기회귀 롤아웃 궤적 전체에 걸쳐 적용한다는 점입니다. 즉 student는 자신의 예측이 유도하는 상태 분포 위에서 최적화되는데, 이것이 anti-drift의 핵심 메커니즘으로 제시됩니다.

### (3) 모델 구조

연구팀은 생성기를 Director-Pilot Co-Simulation Framework로 감싸는데, Vision-Language Model이 Director로서 거시적 의미 규칙과 인과적 추론을 관장하고, Diffusion Transformer 비디오 생성기가 Pilot로서 저수준 물리적 동역학을 시뮬레이션하고 전환을 렌더링합니다. Mode A(Direct Semantic Interaction)에서는 VLM이 현재 프레임을 읽고 이벤트 카드를 생성합니다.

전작 LingBot-World의 백본 구조는 Wan2.2라는 14B 파라미터 image-to-video diffusion 백본으로 시작해, 고노이즈·저노이즈 디노이징에 각각 하나씩 배정된 Mixture-of-Experts DiT 블록을 도입하고, 공간-시간 일관성을 위한 self-attention, 창발적 공간 기억, 연속적 카메라 회전을 위한 Plücker encoding, AdaLN을 통한 이산 키보드/행동 주입, 텍스트 조건화 cross-attention을 핵심 메커니즘으로 사용합니다.

### (4) 성능 향상

LingBot-World-Infinity는 일반 도메인 내에서 시간 단위(무한) 생성 지속시간을 달성하는 유일한 모델로 두드러집니다. 이는 720p·60fps에서 1시간 이상의 지속적 생성에도 품질 저하 없이 검증된 드리프트 없는 인터랙티브 세계를 렌더링하는 실시간 증류 모델로 입증되었습니다.

전작 LingBot-World의 정량 평가에서는 VBench에서 720p 기준 동적 정도(dynamic degree) 0.8857로, 경쟁 모델 대비 절대적으로 16% 높은 수치를 기록하며 일반 도메인에서 높은 동적 정도, 미적 품질, 전체 일관성을 실시간 속도로 동시에 달성하는 유일한 오픈소스 모델임을 보였습니다. 두 베이스라인(Yume-1.5, HY-World-1.5) 대비 이미징 품질, 미적 품질, 동적 정도에서 더 높은 점수를 기록했으며, 동적 정도 격차가 커서(0.8857 대 0.7612, 0.7217) 더 풍부한 장면 전환과 사용자 입력에 반응하는 복잡한 움직임을 시사했습니다. 모션 스무스니스와 시간적 플리커링은 최고 베이스라인과 비슷했고, 전체 일관성 지표는 3개 모델 중 최고를 달성했습니다.

### (5) 한계

모델은 상당한 잠재력을 보이지만 몇 가지 기술적 제약이 남아 있습니다. 높은 추론 비용은 현재 엔터프라이즈급 GPU를 필요로 하여 일반 소비자 하드웨어에서는 접근이 어렵고, 메모리가 명시적 저장 모듈이 아닌 컨텍스트 윈도우로부터 창발적으로 발생하기 때문에 시뮬레이션이 장기적 안정성을 결여하며 장시간에 걸쳐 장면이 구조적 무결성을 점진적으로 잃는 환경 드리프트가 자주 발생합니다. 제어 능력 역시 기본 내비게이션에 제한되어 복잡한 상호작용이나 특정 객체 조작에 필요한 정밀함이 부족합니다.(이는 전작 기준이며, Infinity 버전은 이러한 드리프트·행동 다양성 문제를 상당 부분 개선한 것으로 보입니다.)

또한 독립적인 벤치마크 연구(MemoBench)에서는 LingBot-World가 ORS(Object Recall Score)에서 최상위권 모델 중 하나임에도 타겟 객체가 프레임을 벗어난 후 재등장할 때 이를 충실하게 복원하지 못하는 것으로 나타났습니다. 전체적으로 카메라 제어 가능성, 높은 ORS, 경쟁력 있는 시각적 품질을 동시에 달성하는 단일 모델은 아직 없으며, 이를 해소하는 것이 향후 월드 생성 모델의 핵심 과제로 지목됩니다.

---

## 3. 모델의 일반화 성능 향상 가능성

이 논문의 구조적 특징들은 일반화 성능 측면에서 다음과 같은 함의를 갖습니다.

**(1) 도메인 범용성(General-domain generalization)**: 대부분의 월드 모델이 사진처럼 현실적인 콘텐츠에만 작동하는 반면, LingBot-World는 독자적인 다중 도메인 학습 접근법 덕분에 다양한 시각적 스타일에 걸쳐 품질을 유지합니다. 이는 게임, 애니메이션, 실사 등 이질적 데이터 분포에 걸친 일반화 능력을 시사합니다.

**(2) 자기 롤아웃 기반 학습을 통한 분포 이동 대응**: DMD를 teacher-forced 상태뿐 아니라 모델 자신의 롤아웃 궤적에 적용하는 방식은, 훈련-추론 간 분포 불일치(exposure bias) 문제를 정면으로 다룹니다. 이는 실제 배포 환경에서 모델이 스스로 생성한 (훈련 데이터와는 다를 수 있는) 상태 분포에서도 안정적으로 작동하도록 하는 핵심 일반화 메커니즘으로, 관련 연구에서도 캐시된 key-value 쌍이 인터리브된 비디오-행동 궤적을 보존해 시간적 드리프트를 완화하는 데 도움을 주는, 지속적 컨텍스트를 통한 폐루프(closed-loop) 방식이 장기 지평 과업에서 개방루프 방식이 겪는 분포 드리프트를 완화한다는 유사한 원리가 확인됩니다.

**(3) 아키텍처 수준의 전이 가능성**: Mixture-of-Experts와 block-causal attention의 결합은 단일 백본이 다양한 노이즈 레벨·시간 스케일에 특화된 표현을 학습하게 함으로써, 새로운 행동 유형(공격, 궁술, 마법 등)이나 텍스트 기반 이벤트로의 확장에도 기존 표현을 재사용할 수 있는 잠재력을 제공합니다. Agentic harness(Director-Pilot 구조)의 도입은 저수준 물리 시뮬레이션과 고수준 의미적 계획을 분리함으로써, 새로운 상호작용 규칙이나 게임 로직을 VLM 층에서 유연하게 추가할 수 있어 재학습 없이도 일반화 범위를 넓힐 수 있는 구조적 이점을 갖습니다.

**(4) 한계로서의 일반화**: 그러나 시각적 부분 관찰 하에서의 메모리라는 근본적 과제는 여전히 미해결로 남아있으며, 이는 장기 일반화(특히 객체 영속성)가 아직 완전히 해결되지 않았음을 보여줍니다. 즉, 현재의 "무한 생성"은 주로 전역적 시각 품질의 안정성을 의미하며, 세밀한 의미적 일관성(객체 정체성 유지 등)의 일반화는 후속 과제로 남아 있습니다.

---

## 4. 향후 연구에 미치는 영향과 고려할 점

**영향**

1. **오픈소스 기준선 제공**: 이 모델은 리딩 수준의 시각적 품질과 드리프트에 대한 강한 저항성을 결합한, 커뮤니티에 사용 가능한 백본과 강력한 증류 teacher를 동시에 제공하는 개방형 최첨단 causal world model로서, 향후 연구자들이 자체 인터랙티브 월드 모델을 구축할 때의 표준 참조점이 될 수 있습니다.
2. **평가 체계에 대한 자극**: VBench, EvalCrafter, VideoPhy 같은 기존 비-인터랙티브 평가 스위트는 행동 입력을 받지 않고 다중 턴 상호작용을 평가하지 않는다는 한계가 지적되며, WBench·MemoBench 같은 인터랙티브 평가 벤치마크의 등장을 촉발했습니다. 이는 향후 월드 모델 연구가 단순 화질 지표를 넘어 상호작용성·기억력·인과 일관성을 정량화하는 방향으로 나아가야 함을 보여줍니다.
3. **에이전틱 통합의 선례**: Director-Pilot 구조는 세계 시뮬레이션과 자율 에이전트 계획을 결합하는 새로운 연구 방향(예: 폐루프 RL 훈련, 임베디드 에이전트 학습)을 제시하며, 글로벌 에이전틱 시스템을 위한 고신뢰도, 언어·행동 조건화 시뮬레이션의 토대가 될 수 있음을 시사합니다.

**향후 연구 시 고려할 점**

- **메모리 메커니즘의 명시화**: 현재의 창발적(컨텍스트 기반) 메모리는 부분 관찰 후 재등장하는 객체를 충실히 복원하지 못하는 한계가 있으므로, 명시적 장기 메모리 모듈(외부 메모리, 3D 일관성 제약 등)의 통합이 필요합니다.
- **평가 지표의 정교화**: 카메라를 거의 움직이지 않아 지표가 트리비얼하게 최대화되는 경우처럼, 표준 비디오 품질 지표가 실제 시점 변화 속에서 외형을 보존하는 모델과 단순히 움직임을 회피하는 모델을 구분하지 못하는 한계가 있어, 카메라 활동성을 통제한 평가 프로토콜 설계가 중요합니다.
- **계산 효율성과 접근성**: 고성능 GPU에 의존하는 현재 구조는 폭넓은 채택을 저해하므로, 경량화·양자화·온디바이스 추론 연구가 병행되어야 합니다.
- **행동 공간의 세밀한 제어**: 내비게이션 중심에서 세밀한 객체 조작으로 행동 공간을 확장하는 연구가 필요하며, 이는 로보틱스·임베디드 AI 응용에서 특히 중요합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 모델/연구 | 특징 | LingBot-World-Infinity와의 비교 |
|---|---|---|
| **InfiniteWorld** (2024) | Nvidia Isaac Sim 기반의 통합·확장 가능한 vision-language 로봇 상호작용 시뮬레이터로, 물리 에셋 구축 방법과 일반화된 자유 로봇 상호작용 벤치마크를 포괄 | 물리 시뮬레이터 기반(rule-based physics)인 반면, LingBot-World는 순수 생성 모델(diffusion) 기반으로 학습된 암묵적 물리 지식을 사용 |
| **Yume-1.5 / HY-World-1.5** | 각각 자연어 행동을 사용하는 언어 기반 상호작용과, 내비게이션·기하학적 일관성에 중점을 둔 카메라 제어 생성을 대표 | LingBot-World-Infinity는 동적 정도·일관성 지표에서 이들을 상회 |
| **Matrix-Game 2.0/3.0, Hunyuan-GameCraft** | 실시간 키보드·마우스 제어를 추진하며, Matrix-Game 3.0은 명시적 메모리를 통해 장기 지평 일관성을 더욱 개선 | 명시적 메모리 도입 방향은 LingBot-World의 창발적 메모리 한계를 보완할 잠재적 방향으로 시사됨 |
| **Genie 3 (Google, 비공개)** | 폐쇄형 시스템으로서 이 분야의 추진력을 보여주는 대표 사례 | LingBot-World는 이에 대응하는 완전 오픈소스 대안으로 포지셔닝 |
| **Causal World Modeling for Robot Control (2026)** | 비디오·행동 토큰을 통합 시퀀스로 구성해 매 자기회귀 단계마다 최신 실제 관측에 기반해 재조정하는 반응형 AR 루프를 특징으로 함 | 로봇 제어 특화이나, 폐루프를 통한 드리프트 완화 원리는 LingBot-World의 self-rollout DMD와 상통 |
| **MemoBench (2026)** | 카메라 제어 가능성, ORS, 시각적 품질을 동시에 달성하는 모델이 없음을 지적하며 이를 향후 과제로 제시 | LingBot-World를 포함한 현 세대 모델들의 공통 한계를 실증적으로 드러냄 |

전반적으로, 2020년대 초반의 연구(InfiniteWorld 등)가 규칙 기반 물리 시뮬레이터와 벤치마크 구축에 집중했던 것과 달리, 2025~2026년의 최신 흐름(LingBot-World, Yume, Matrix-Game, Genie 3 등)은 **대규모 diffusion/DiT 기반 생성 모델을 "학습된 물리·인과 엔진"으로 활용**하는 방향으로 수렴하고 있으며, LingBot-World-Infinity는 그중에서도 **무한 생성 지평, 실시간 처리, 에이전틱 통합**을 동시에 달성한 최초의 오픈소스 시도로 평가할 수 있습니다.

---

### 참고 자료 (출처)
1. arXiv:2607.07534, *Infinite Worlds with Versatile Interactions* — https://arxiv.org/abs/2607.07534 / https://arxiv.org/pdf/2607.07534 / https://arxiv.org/html/2607.07534v1
2. GitHub — Robbyant/lingbot-world-v2 — https://github.com/Robbyant/lingbot-world-v2
3. Hugging Face — robbyant/lingbot-world-v2-14b-causal-fast — https://huggingface.co/robbyant/lingbot-world-v2-14b-causal-fast
4. MarkTechPost, *"Meet LingBot-World-Infinity: An Open Causal World Model With An Agentic Harness"* (2026.07.09) — https://www.marktechpost.com/2026/07/09/meet-lingbot-world-infinity-an-open-causal-world-model-with-an-agentic-harness/
5. MarkTechPost, *"Robbyant Open Sources LingBot World"* (2026.01.30) — https://www.marktechpost.com/2026/01/30/robbyant-open-sources-lingbot-world-a-real-time-world-model-for-interactive-simulation-and-embodied-ai/
6. Emergent Mind, *"LingBot-World: Open-Source Simulator"* — https://www.emergentmind.com/topics/lingbot-world
7. Emergent Mind, *"LingBot-World: Real-Time Simulation"* — https://www.emergentmind.com/topics/lingbot-world-92824f09-49aa-45d8-9876-f4e1359ca0b1
8. arXiv:2601.20540, *Advancing Open-source World Models* — https://arxiv.org/html/2601.20540v1
9. arXiv:2601.21998, *Causal World Modeling for Robot Control* — https://arxiv.org/pdf/2601.21998
10. arXiv:2412.05789, *InfiniteWorld: A Unified Scalable Simulation Framework for General Visual-Language Robot Interaction* — https://arxiv.org/abs/2412.05789
11. arXiv:2606.27537, *MemoBench: Benchmarking World Modeling in Dynamically Changing Environments* — https://arxiv.org/html/2606.27537
12. arXiv:2605.25874, *WBench: A Comprehensive Multi-turn Benchmark for Interactive Video World Model Evaluation* — https://arxiv.org/html/2605.25874v1
13. Robbyant Technology 공식 페이지 — https://technology.robbyant.com/lingbot-world
14. LingBot-World 공식 사이트 — https://www.lingbot-world.org/
