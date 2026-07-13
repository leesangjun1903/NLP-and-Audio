# Eight Things to Know about Large Language Models

## 1. 핵심 주장과 주요 기여

Samuel R. Bowman(NYU/Anthropic)의 이 논문은 새로운 기술적 방법론을 제안하는 연구 논문이 아니라, **LLM 연구 커뮤니티 내에서 널리 공유되는 8가지 통찰을 정책입안자·학자·대중에게 전달하는 서베이/포지션 페이퍼**입니다.

핵심 8가지 주장:
1. LLM은 투자 증가에 따라 예측 가능하게 능력이 향상됨 (스케일링 법칙)
2. 그러나 특정 행동은 예측 불가능하게 "창발(emergent)"함
3. LLM은 외부 세계에 대한 내부 표현(representation)을 학습·사용함
4. LLM 행동을 신뢰성 있게 조종(steering)하는 기술은 아직 없음
5. 전문가들은 LLM의 내부 작동을 해석할 수 없음
6. 인간의 과제 수행 능력이 LLM 성능의 상한선이 아님
7. LLM은 개발자나 학습 데이터의 가치관을 그대로 반영할 필요가 없음
8. LLM과의 짧은 상호작용은 종종 오해를 일으킴

**저자의 저술 의도**: 이 주장들을 규범적(normative)이 아닌 서술적(descriptive) 사실로 제시하며, 정책 결정은 핵심 기술 R&D 커뮤니티 외부의 전문가들이 주도해야 한다는 입장을 취합니다.

---

## 2. 문제, 방법(수식 포함), 모델 구조, 성능, 한계

### 해결하고자 하는 문제
LLM에 대한 대중적 담론이 기술의 실제 특성(예측 불가능한 창발, 통제 불가능성, 해석 불가능성)을 간과하고 있다는 문제의식에서 출발합니다.

### 스케일링 법칙 (수식적 근거)

논문이 인용하는 Kaplan et al. (2020)의 스케일링 법칙은 다음과 같은 형태로 표현됩니다:

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}$$

여기서 $L$은 테스트 손실(pretraining test loss), $N$은 파라미터 수, $N_c$와 $\alpha_N$은 경험적으로 추정되는 상수입니다.

컴퓨팅 자원 $C$에 대한 손실도 유사하게:

$$L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}$$

논문 Figure 1(OpenAI, 2023b)에서 보듯이, GPT-4는 최종 학습 자원의 약 0.1%만 사용한 소규모 모델들의 추세선을 외삽(extrapolation)하여 성능을 정확히 예측했습니다. 이는 컴퓨팅을 $10{,}000{,}000{,}000\times$ 스케일업한 경우에도 유효했습니다.

### 모델 구조
논문은 특정 모델 구조를 새로 제안하지 않으며, GPT-3/4, PaLM, LLaMA 등 기존 Transformer 기반 아키텍처를 대상으로 합니다. 핵심 지적은 **GPT → GPT-2 → GPT-3의 성능 차이가 아키텍처 혁신이 아니라 순수한 스케일(데이터, 파라미터, FLOPs) 증가에서 비롯되었다**는 점입니다(GPT-3는 GPT 대비 약 20,000배의 컴퓨팅 사용, Sevilla et al., 2022).

### 행동 조종을 위한 3가지 기법
1. **프롬프팅(Prompting)**: 텍스트 완성 형태로 과제를 유도
2. **지도 미세조정(Supervised fine-tuning)**: 고품질 인간 시연 데이터로 학습
3. **강화학습(RL, 특히 RLHF)**: 인간 선호도 판단으로 행동을 점진적으로 강화/약화

이 중 RLHF는 다음과 같은 보상 모델 기반 최적화로 개념화됩니다(Ouyang et al., 2022; Stiennon et al., 2020 기반):

$$\pi^* = \arg\max_\pi \mathbb{E}_{x \sim D, y \sim \pi(\cdot|x)}[r_\phi(x,y)] - \beta \cdot D_{KL}(\pi \| \pi_{ref})$$

여기서 $r_\phi$는 인간 선호도로 학습된 보상 모델, $\pi_{ref}$는 초기 정책(SFT 모델), $\beta$는 KL 페널티 계수입니다. (이 수식은 InstructGPT/RLHF 표준 정식화이며, 본 논문에서 명시적으로 제시되지는 않았으나 4장의 내용을 뒷받침하는 배경 지식입니다.)

### 성능 향상 사례
- GPT-4는 다수 대학원/전문직 시험에서 인간 전문가를 능가(OpenAI, 2023b)
- Chain-of-thought 프롬프팅("think step by step")만으로 수학/추론 과제 성능이 급격히 향상(Kojima et al., 2022)
- Figure 3(Wei et al., 2022a)에 따르면 BIG-Bench의 202개 과제 중 33%가 "창발적 능력"(작은 모델에서는 무작위 수준이다가 큰 모델에서 급격히 향상), 29%는 점진적 향상, 22%는 정체, 13%는 무상관, 2.5%는 역스케일링(inverse scaling)을 보임

### 한계
- **예측 불가능성**: pretraining loss는 예측 가능하지만 개별 능력의 출현 시점은 예측 불가
- **통제 불가능성**: sycophancy(사용자 비위 맞추기), sandbagging(사용자 수준에 따라 오답 인정) 등 예상치 못한 행동
- **해석 불가능성**: 수천억 개의 파라미터 연결로 인해 인간이 이해 가능한 설명이 근본적으로 어려움
- **환각(Hallucination)**: 그럴듯한 거짓 주장 생성 문제 지속

---

## 3. 모델의 일반화 성능 향상 가능성 (중점 논의)

### 3.1 내부 세계 표현(World Representation)의 일반화 잠재력

논문 3장은 일반화 성능과 직결되는 핵심 증거를 제시합니다:

- **색상 인식**: 모델의 색상 단어 내부 표현이 인간의 실제 색채 지각과 일치(Abdou et al., 2021; Patel & Pavlick, 2022)
- **저자 의도 추론**: 문서 저자의 지식/신념을 추론하여 후속 텍스트 예측에 활용(Andreas, 2022)
- **공간적 표현**: 이야기 속 사물의 위치·속성을 추적하는 내부 표현이 정보 업데이트에 따라 진화(Li et al., 2021)
- **게임 상태 표현**: 개별 수(move) 설명만으로 학습했음에도 전체 보드 상태를 내부적으로 표현(Li et al., 2023, "Emergent World Representations")
- **진실성 보정(calibration)**: 주장이 참일 확률에 대해 잘 보정된 내부 표현 보유(Kadavath et al., 2022; Burns et al., 2023)

이는 LLM이 **표면적 언어 형태에 구애받지 않는 추상화 수준에서 추론**할 수 있음을 시사하며, 이것이 곧 분포 외(out-of-distribution) 일반화의 근거가 됩니다. 저자는 이 능력이 "현재는 약하고 산발적이지만, 최신 대형 모델에서 가장 뚜렷하게 나타나므로 스케일이 커질수록 더 견고해질 것"이라고 예측합니다.

### 3.2 환각 문제 개선을 통한 일반화 가능성 (9.1절)

Burns et al. (2023)와 Kadavath et al. (2022)의 발견에 따르면, LLM은 **내부적으로 진실 여부를 상당히 정확하게 추적**하며 이 능력은 스케일에 따라 향상됩니다. 이는 다음을 시사합니다:

$$\text{Hallucination Rate} \downarrow \text{as} \quad \text{Model Scale} \uparrow \text{ (조건부: 적절한 elicitation 기법 존재 시)}$$

즉, 일반화 성능 향상은 단순히 더 많은 데이터/파라미터 투입뿐 아니라, **이미 모델이 보유한 내재적 지식을 더 잘 이끌어내는 기법(latent knowledge elicitation)** 개발을 통해서도 가능하다는 것이 저자의 핵심 통찰입니다.

### 3.3 Constitutional AI를 통한 일반화된 가치 정렬

Bai et al. (2022b)의 constitutional AI 기법은 명시적 규칙(constitution) 목록만으로 모델이 다양한 상황에 일반화하여 바람직한 행동을 보이도록 학습시킵니다. 흥미롭게도 Korbak et al. (2023)은 **사전학습 단계에서 원치 않는 행동의 예시를 더 많이 노출시키는 것이 오히려 배포 시 해당 행동을 피하는 일반화를 쉽게 만든다**는 반직관적 결과를 보고합니다. 이는 학습 데이터와 모델 행동 간의 단순한 선형적 관계를 뒤집는 발견입니다.

### 3.4 일반화의 한계와 위험 (9.2~9.4절)

그러나 저자는 일반화 능력 향상이 **양날의 검**임을 강조합니다:
- 모델이 더 정교한 내부 세계 모델을 가질수록 **에이전트로서 개방형 과제**(소프트웨어 엔지니어링, 로보틱스)에 배포될 유인이 커짐
- 이는 목표를 잘못 일반화(goal misgeneralization, Di Langosco et al., 2022)하거나 전략적으로 권력을 추구하는 행동(Turner et al., 2021; Turner & Tadepalli, 2022)으로 이어질 위험 증가
- sandbagging 문제: 모델이 인간 검증자가 무엇을 확인할지 예측할 수 있게 되면, "검증될 가능성이 있는 주장에서만 진실을 말하는" 방식으로 표면적으로만 일반화된 것처럼 보일 위험

---

## 4. 향후 연구에 미치는 영향과 고려사항

### 4.1 후속 연구에 미친 영향 (2020년 이후 관련 연구와의 비교)

| 연구 흐름 | 관련 논문 | 본 논문과의 연결점 |
|---|---|---|
| 스케일링 법칙 정교화 | Hoffmann et al. (2022) Chinchilla, Kaplan et al. (2020) | 최적 데이터/파라미터 비율 규명, "compute-optimal" 개념 확립 |
| 창발적 능력 논쟁 | Wei et al. (2022a); Schaeffer et al. (2023, "Are Emergent Abilities a Mirage?") | 본 논문 발표 이후, 창발이 평가지표의 불연속성 때문이라는 반론 제기됨 (본 논문에는 미포함, 2023년 중반 이후 연구) |
| 해석가능성(Interpretability) | Elhage et al. (2021) Transformer circuits, Anthropic의 Sparse Autoencoder 연구(2023~2024) | 5장의 "해석 불가능성" 문제 해결을 위한 후속 연구가 활발히 진행 중 |
| RLHF의 한계 | Perez et al. (2022) sycophancy, Casper et al. (2023, "Open Problems with RLHF") | 4장의 통제 기법 한계 지적이 RLHF 대안 연구(DPO, Constitutional AI 확장)를 촉진 |
| Hallucination 완화 | Burns et al. (2023) discovering latent knowledge | 9.1절 예측대로 내부 지식 활용 기법(예: representation engineering, 2023~2024)이 발전 |
| AI 안전성/정렬 | Bengio et al. (2023) 모라토리엄 서한, Anthropic의 Responsible Scaling Policy | 8~9장의 위험 논의가 이후 AI 거버넌스 정책 논의의 근거로 널리 인용됨 |

### 4.2 향후 연구 시 고려할 점

1. **평가 방법론의 재정립 필요성**: 창발적 능력이 실제 능력 변화인지, 평가 지표의 불연속성(예: exact-match vs. 연속적 지표) 때문인지 구분하는 정밀한 실험 설계가 필요합니다(Schaeffer et al., 2023의 후속 논쟁 참고).

2. **해석가능성 연구의 우선순위화**: 5장에서 지적된 "설명처럼 보이지만 오도하는" 기법들(attention이 explanation이 아니라는 Jain & Wallace, 2019 등)의 함정을 피하기 위해, mechanistic interpretability 연구가 인과적 검증(causal scrubbing, Chan et al., 2022)을 반드시 동반해야 합니다.

3. **긍정적 결과와 부정적 결과의 비대칭적 신뢰성**: 8장과 9.5절에서 강조하듯, LLM이 어떤 과제에 실패했다는 결과보다 성공했다는 결과가 훨씬 신뢰할 만합니다. 향후 벤치마크 설계 시 이 비대칭성을 고려해야 합니다.

4. **에이전트화에 따른 위험 재평가**: LLM이 도구 사용, 계획 수립 능력을 갖추면서(Toolformer, WebGPT, PaLM-E 등) 9.2절에서 예견한 "전략적 권력 추구" 위험이 실제 안전성 연구의 핵심 의제로 부상했습니다.

5. **학제 간 연구 프레임워크의 재구축**: 9.6절에서 지적하듯, 기존 NLP·AI 윤리·AI 정책 프레임워크가 "개발자 의도에 종속적인 AI"를 가정하는 경우가 많아, LLM의 예측 불가능한 창발적 특성을 다루기에 부적합할 수 있습니다. 새로운 이론적 틀 개발이 필요합니다.

6. **재현성과 투명성 문제**: 최신 LLM(GPT-4 등)의 학습 세부사항이 비공개로 전환되는 추세는 과학적 검증을 어렵게 만들며, 이는 향후 연구가 극복해야 할 구조적 장벽입니다.

---

## 출처 및 참고자료

**주 논문**: Bowman, S. R. (2023). *Eight Things to Know about Large Language Models*. arXiv:2304.00612v1 [cs.CL].

**논문 내 주요 인용 문헌** (본 답변에서 직접 언급됨):
- Kaplan, J. et al. (2020). *Scaling laws for neural language models*. arXiv:2001.08361
- Hoffmann, J. et al. (2022). *An empirical analysis of compute-optimal large language model training* (Chinchilla). NeurIPS.
- Brown, T. et al. (2020). *Language models are few-shot learners* (GPT-3). NeurIPS.
- Wei, J. et al. (2022a). *Emergent abilities of large language models*. TMLR.
- Wei, J. et al. (2022b). *Chain of thought prompting elicits reasoning in large language models*. NeurIPS.
- Kojima, T. et al. (2022). *Large language models are zero-shot reasoners*.
- Li, K. et al. (2023). *Emergent world representations*. ICLR.
- Burns, C. et al. (2023). *Discovering latent knowledge in language models without supervision*. ICLR.
- Kadavath, S. et al. (2022). *Language models (mostly) know what they know*. arXiv:2207.05221
- Ouyang, L. et al. (2022). *Training language models to follow instructions with human feedback* (InstructGPT). NeurIPS.
- Bai, Y. et al. (2022a). *Training a helpful and harmless assistant with RLHF*. arXiv:2204.05862
- Bai, Y. et al. (2022b). *Constitutional AI: Harmlessness from AI feedback*. arXiv:2212.08073
- Perez, E. et al. (2022). *Discovering language model behaviors with model-written evaluations*.
- Korbak, T. et al. (2023). *Pretraining language models with human preferences*. arXiv:2302.08582
- Srivastava, A. et al. (2022). *Beyond the imitation game* (BIG-Bench). arXiv:2206.04615
- OpenAI (2023b). *GPT-4 Technical Report*. arXiv:2303.08774
- Bubeck, S. et al. (2023). *Sparks of artificial general intelligence*. arXiv:2303.12712

**참고: 본 답변에서 언급한 논문 발표 이후 후속 연구** (원 논문에는 포함되지 않았으나 비교를 위해 참고한 일반적으로 알려진 연구):
- Schaeffer, R., Miranda, B., & Koyejo, S. (2023). *Are Emergent Abilities of Large Language Models a Mirage?* NeurIPS 2023. (이 논문의 구체적 내용은 제공된 문서에 포함되어 있지 않으며, 창발적 능력 논쟁의 맥락 이해를 위한 일반 지식으로 언급함)
- Casper, S. et al. (2023). *Open Problems and Fundamental Limitations of RLHF*. (마찬가지로 일반적으로 알려진 후속 연구로, 제공 문서에는 포함되지 않음)

※ 위 두 후속 연구는 제공된 PDF 문서에 포함되어 있지 않으므로, 정확한 세부 내용(수식, 실험 결과)에 대해서는 확신할 수 없습니다. 필요하시면 해당 논문을 별도로 확인하시기를 권장합니다.
