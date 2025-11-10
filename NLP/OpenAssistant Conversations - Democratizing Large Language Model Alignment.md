# OpenAssistant Conversations - Democratizing Large Language Model Alignment

### 1. 핵심 주장 및 주요 기여 요약

이 논문의 **핵심 주장**은 대규모 언어 모델의 정렬(alignment)을 민주화하기 위해 고품질의 인간 주석 데이터가 필수적이라는 것입니다. 기존의 RLHF(Reinforcement Learning from Human Feedback)와 지도 미세조정(Supervised Fine-Tuning, SFT) 기술들이 고비용의 고품질 데이터에 의존하고 있던 상황에서, 본 논문은 이를 해결하기 위해 **OpenAssistant Conversations(OASST1)** 데이터셋을 공개합니다.[1]

**주요 기여는 다음과 같습니다:**

- **대규모 공개 데이터셋**: 35개 언어로 161,443개의 메시지(프롬프터 91,829개, 어시스턴트 69,614개)로 구성되며, 461,292개의 품질 평가와 함께 10,000개 이상의 완전히 주석 처리된 대화 트리를 포함합니다.[1]

- **글로벌 크라우드소싱**: 13,500명 이상의 자원봉사자의 참여로 625,000개 이상의 작업을 통해 수집되었습니다.[1]

- **완전 공개 라이선스**: 코드와 데이터를 완전히 공개적인 라이선스로 공개하여 연구 커뮤니티 접근성 제고합니다.[1]

### 2. 해결 문제, 제안 방법, 모델 구조 및 성능

#### 해결하고자 하는 문제

논문이 식별한 주요 문제는 다음과 같습니다:[1]

1. **고품질 인간 피드백 데이터의 부족**: RLHF 기술의 효과성이 데이터 품질에 크게 의존하지만, 공개적으로 접근 가능한 대규모 인간 피드백 데이터셋이 극히 부족합니다.

2. **기존 공개 데이터셋의 한계**: 대부분의 공개 데이터셋(Alpaca 등)은 언어 모델으로 자동 생성된 지시사항을 포함하거나 언어 모델이 생성한 응답을 사용하므로 다양성, 창의성, 품질이 제한됩니다.

3. **연구 접근성 문제**: 고품질 정렬 연구가 소수의 대형 연구 기관으로만 제한되어 포함적이고 다양한 연구 수행이 제약됩니다.

#### 제안하는 방법: 데이터 수집 및 구조

논문은 **대화 트리(Conversation Tree, CT)** 구조를 중심으로 설계된 체계적인 수집 방식을 제안합니다:[1]

**단계별 수집 과정:**

각 대화 트리는 5가지 분리된 단계로 점진적으로 구축됩니다:

1. **프롬프트 생성(Create a prompt)**: 사용자가 새로운 대화의 시작점이 되는 초기 프롬프트를 작성합니다. 유입을 조절하기 위해 로터리 시스템을 사용합니다.[1]

2. **레이블 지정(Labelling prompts)**: 생성된 프롬프트를 스팸 탐지, 지침 준수, 품질의 3가지 차원에서 평가합니다.[1]

3. **응답 생성(Adding reply messages)**: 어시스턴트 또는 프롬프터 역할로 응답을 추가합니다. 어시스턴트 응답은 보상점 시스템으로 인센티브를 제공합니다.[1]

4. **응답 레이블 지정(Labelling replies)**: 5점 리커트 척도를 사용하여 품질, 창의성, 유머, 친절함, 해로움 없음을 평가합니다.[1]

5. **응답 순위 지정(Ranking assistant replies)**: 같은 프롬프트에 대한 여러 응답을 비교하여 선호도 순서대로 순위를 매깁니다.[1]

**트리 상태 머신(Message Tree State Machine)**:

각 대화 트리는 다음 상태를 거칩니다:[1]
- 초기 프롬프트 검토 상태(Initial Prompt Review State)
- 성장 상태(Growing State)  
- 종료 상태(End State)
- 저등급 상태(Aborted Low-Grade State) 또는 중재자에 의한 중단 상태

#### 모델 구조

논문의 실험에서는 세 가지 유형의 모델을 훈련합니다:[1]

**1. SFT 모델 (Supervised Fine-Tuning)**

기본 구조: $$L(\theta) = -\sum_{i=1}^{n} \sum_{t=1}^{m_i} \log p_\theta(a_{i,t} | p_i, a_{i,1:t-1})$$

여기서:
- $$\theta$$: 모델 파라미터
- $$p_i$$: i번째 프롬프트
- $$a_{i,t}$$: i번째 대화의 t번째 토큰
- 프롬프트 토큰의 손실은 마스킹되어 어시스턴트 응답 토큰만 훈련에 사용됩니다.[1]

**2. Reward Model (RM)**

구조: 언어 모델링 헤드를 단일 출력 $$r_\theta(x, y)$$을 생성하는 선형 층으로 대체합니다.

손실 함수: $$\text{loss}(\theta) = -\frac{1}{\binom{K}{2}} \mathbb{E}\_{(x, y_w, y_l)}[\log(\sigma(r_\theta(x, y_w) - r_\theta(x, y_l)))]$$

여기서:
- $$\sigma$$: 시그모이드 함수
- $$y_w$$: 선호되는 완성
- $$y_l$$: 비선호되는 완성
- K개의 서로 다른 응답에 대해 모든 $$\binom{K}{2}$$ 비교를 생성합니다.[1]

**3. RLHF 모델**

PPO(Proximal Policy Optimization) 알고리즘을 사용하여 SFT 모델을 최적화합니다. 목적 함수는:

$$\mathcal{L}(\pi) = \mathbb{E}[\min(r_t(\theta), \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon))] - \beta \text{KL}(D || \pi_{\text{ref}})$$

여기서:
- $$r_t(\theta)$$: 확률 비(probability ratio)
- $$\epsilon$$: 클리핑 범위
- $$\beta$$: KL 페널티 계수로 참조 모델으로부터의 편차 방지[1]

#### 성능 향상

논문의 실험 결과(표 1):[1]

| 모델 | LMEH | VEL | OAIE | HE |
|------|------|-----|------|-----|
| gpt-3.5-turbo (ChatGPT) | 1110 | 0.87 | 0.72 | - |
| Pythia-12B (기본) | 60.33 | - | - | - |
| OpenAssistant/Pythia-12B-SFT | 60.28 | 997 | 0.10 | 0.10 |
| Falcon-40B (기본) | 72.29 | - | - | - |
| OpenAssistant/Falcon-40B-SFT-top1 | 74.04 | 1192 | 0.26 | 0.09 |
| OpenAssistant/Falcon-40B-SFT-mix | 74.40 | 1053 | 0.44 | 0.13 |
| LLaMA-65B (기본) | 67.24 | - | - | - |
| OpenAssistant/LLaMA-30B-SFT | 68.03 | 979 | 0.52 | 0.20 |
| OpenAssistant/LLaMA-30B-RLHF | 68.51 | 1068 | 0.51 | 0.15 |

**주요 성과:**

- OpenAssistant 데이터로 훈련된 모델들이 기본 모델보다 일관되게 성능 향상을 보입니다.[1]
- RLHF는 일부 벤치마크에서 SFT를 능가하지만, 다른 벤치마크에서는 그렇지 않습니다.[1]
- 더 큰 기본 모델(LLaMA-65B)을 능가하는 더 작은 미세조정 모델(LLaMA-30B-RLHF)의 성능을 달성했습니다.[1]

### 3. 모델의 일반화 성능 향상 가능성

#### 현 논문에서의 일반화 논의

논문은 직접적인 도메인 외(out-of-domain) 성능 평가는 제시하지 않지만, 여러 가지 일반화 관련 발견을 제시합니다:[1]

**다양성 기반 일반화:**
- 35개 언어의 포함으로 다언어 일반화 가능성을 높입니다.
- 다양한 주제와 작업 유형(LDA 분석으로 33개 주제 확인)을 포함하여 작업 다양성을 확보합니다.[1]
- 사용자 다양성: 약 95% 응답자가 기여한 것에 만족하고, 40% 이상이 오픈소스 프로젝트 첫 기여입니다.[1]

**한계와 편향:**

논문은 다음의 일반화 관련 한계를 명시합니다:[1]

- **주관적, 문화적, 기여 빈도 편향**: 기여자의 89.1%가 남성이고 평균 나이가 26세로 인구통계학적 편중이 있습니다.
- **불균형한 기여 분포**: 소수의 파워 유저가 데이터셋의 상당 부분을 기여하여 이들의 가치와 관심사가 과대표현됩니다(그림 2 참조).[1]
- **보상 모델 수집 방식의 차이**: InstructGPT는 SFT 모델이 생성한 메시지의 순위 데이터로 보상 모델을 훈련했지만, OASST1은 인간이 생성한 메시지로 훈련하여 RLHF의 성능 향상이 제한적입니다.[1]

#### 최신 연구에서의 일반화 성능 개선 방향

2024-2025년의 최신 연구들은 명시적으로 일반화 성능을 다루고 있습니다:[2][3]

**1. Weighted Instruction Tuning (WIT) 접근법**[2]

최근 연구에서는 명령어 튜닝 손실 함수의 최적화를 통해 일반화성을 개선합니다:

$$\text{loss} = \lambda_p \sum_{t \in \text{prompt}} \log p(y_t | x) + \lambda_r \sum_{t \in \text{response}} \log p(y_t | x)$$

여기서 낮음-중간 프롬프트 토큰 가중치($$\lambda_p$$)와 중간-높은 응답 토큰 가중치($$\lambda_r$$)가 도메인 외 성능을 크게 향상시킵니다.[2]

**2. RLA (Reinforcement Learning from Supervised Alignment)**[3]

새로운 접근법은 감독된 정렬을 통해 보상 모델을 구성하여 다음과 같은 결과를 달성합니다:[3]

- 도메인 내 작업에서 기본 LLaMA3 대비 최대 55% 성능 향상
- **도메인 외 작업에서 기본 모델 대비 최대 16% 향상**
- SFT 대비 도메인 외 및 교차 작업 평가에서 **최대 50배 성능 우위**[3]

**3. 일반화 오류 이론적 경계**[4]

최신 연구들은 DPO-COV 알고리즘을 통해 일반화 오류의 이론적 경계를 제시합니다:

$$\text{Gen. Error} = O\left[\frac{\log(N)}{\sqrt{N}}\right]$$

이는 오염된 데이터에도 안정적인 일반화를 제공합니다.[4]

**4. 멀티태스크 학습과 도메인 적응**

최신 지침 조정 조사에 따르면:[5]

- 멀티태스크 지침 조정이 공유 표현 학습을 통해 과적합을 최소화합니다
- 매개변수 효율적 기술(LoRA, QLoRA)이 일반화를 개선합니다
- 곡선 규획 접근법이 저자원 설정에서 전이 학습을 개선합니다[5]

**5. 조성적 일반화(Compositional Generalization)**[6]

최신 발견에 따르면 LLM의 OOD 일반화 능력은 네트워크 계층 간 주요 부분공간 정렬을 통한 조성적 구조에 달려 있습니다.[6]

### 4. 논문의 후속 연구 영향과 고려사항

#### 학계 및 업계에 미친 영향

**OpenAssistant의 기여:**[1]

1. **민주화 효과**: 소수 기관의 독점 상황을 타개하여 전 세계 연구자들에게 고품질 정렬 데이터 접근을 제공합니다.

2. **산업 표준화**: OASST1 데이터셋은 이후 많은 오픈소스 LLM 프로젝트에서 기준 데이터셋으로 채택되었습니다.[7]

3. **글로벌 협력 모델**: 13,500명 이상의 자원봉사자의 참여로 대규모 크라우드소싱의 효과성을 입증했습니다.[1]

#### 향후 연구 시 고려할 점

**1. 보상 모델 데이터 수집의 개선**[1]

논문에서 RLHF 성능이 제한적인 이유로 식별된 보상 모델 학습 데이터 차이를 해결할 필요가 있습니다. 자체 SFT 모델이 생성한 메시지에 대한 순위 데이터 수집이 권장됩니다.[1]

**2. 편향 완화 및 포함성 향상**[2][1]

- 기여자 인구통계학적 다양성 증대
- 불균형한 기여 분포 문제 해결을 위한 메커니즘
- 제약된 대화(예: 무작위 Wikipedia 페이지 기반) 도입으로 편향 감소

**3. 손실 함수 최적화**[2]

최신 연구(WIT)는 프롬프트와 응답 토큰에 대한 차등 가중치가 일반화성을 크게 개선함을 보여줍니다. 향후 OASST1 기반 훈련에서 이를 적용할 필요가 있습니다.[2]

**4. 멀티태스크 및 도메인 적응 전략**[5]

- 여러 도메인에 걸친 지침 조정 데이터 혼합
- 작업 및 도메인 토큰을 활용한 조건부 생성
- 저자원 언어에 대한 과제 난이도 순위 기반 커리큘럼 학습[5]

**5. 안전성 및 윤리적 고려사항**[1]

- 훈련된 모델의 환각(hallucination), 독성, 편향 위험성 지속 모니터링
- 이차 안전 필터링 및 추론 시점 모니터링 메커니즘 강화
- 모델 배포 전 철저한 안전성 및 편향 평가[1]

**6. 최신 정렬 기법과의 결합**

최근 DPO, GRPO 등 더 효율적인 정렬 기법들이 등장하고 있습니다. OASST1은 이들 기법의 벤치마크로서의 역할을 계속할 것으로 예상되며, 이들 기법과 결합하여 더욱 효율적이고 일반화된 모델을 개발할 수 있습니다.[8]

**7. 다국어 및 문화적 공정성**[8]

RLHF-CML(Multilingual RLHF)과 같은 접근법처럼, 23개 이상의 언어에서 선호도 데이터를 생성하고 저자원 언어를 상향 가중치하여 더욱 공정한 다국어 정렬을 추구할 필요가 있습니다.[8]

***

### 결론

OpenAssistant Conversations는 대규모 언어 모델 정렬 연구의 민주화라는 중요한 목표를 달성했으며, 고품질의 다양한 인간 주석 데이터를 공개함으로써 학계와 산업에 중대한 기여를 했습니다. 특히 인간 중심의 데이터 수집과 크라우드소싱 모델의 확장성을 입증했다는 점에서 의미가 있습니다.[1]

그러나 향후 연구에서는 보상 모델 학습 방식의 개선, 편향 완화, 최신 손실 함수 최적화, 그리고 새로운 정렬 기법과의 결합을 통해 일반화 성능을 더욱 개선할 수 있을 것으로 예상됩니다. 특히 도메인 외 성능과 저자원 설정에서의 강화는 향후 연구의 핵심 과 설정에서의 강화는 향후 연구의 핵심 과제가 될 것입니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e2006f11-5fad-459f-a978-166d93fc302d/2304.07327v2.pdf)
[2](https://arxiv.org/abs/2507.07817)
[3](https://aclanthology.org/2025.findings-emnlp.378.pdf)
[4](https://openreview.net/pdf?id=K2OWrXUVby)
[5](https://www.arxiv.org/pdf/2508.17184.pdf)
[6](https://www.pnas.org/doi/10.1073/pnas.2417182122)
[7](http://www.diva-portal.org/smash/get/diva2:1985482/FULLTEXT01.pdf)
[8](https://arxiv.org/html/2511.03939v1)
[9](http://arxiv.org/pdf/2410.03845.pdf)
[10](http://arxiv.org/pdf/2502.13337.pdf)
[11](http://arxiv.org/pdf/2410.05080.pdf)
[12](http://arxiv.org/pdf/2305.00948.pdf)
[13](https://arxiv.org/pdf/2502.06807.pdf)
[14](https://arxiv.org/html/2411.06805v1)
[15](http://arxiv.org/pdf/2411.05877.pdf)
[16](http://arxiv.org/pdf/2410.01792.pdf)
[17](https://arxiv.org/html/2504.01789v1)
[18](https://openreview.net/pdf?id=VSJotgbPHF)
[19](https://github.com/zackschen/CoIN)
[20](https://openreview.net/forum?id=EpnsUQavJA)
[21](https://blog.sionic.ai/finetuning_llama)
[22](https://arxiv.org/pdf/2402.02416.pdf)
[23](http://arxiv.org/pdf/2406.01252.pdf)
[24](http://arxiv.org/pdf/2403.04224.pdf)
[25](https://arxiv.org/pdf/2411.00062.pdf)
[26](https://arxiv.org/pdf/2503.02846.pdf)
[27](http://arxiv.org/pdf/2408.10392.pdf)
[28](http://arxiv.org/pdf/2503.18991.pdf)
[29](https://arxiv.org/html/2502.02659)
[30](https://arxiv.org/html/2410.15595v3)
[31](https://www.emergentmind.com/topics/domain-specific-instruction-tuning)
[32](https://arxiv.org/html/2507.00439v2)
[33](https://datasciencedojo.com/blog/rlhf-and-dpo-for-finetuning-llms/)
[34](https://arxiv.org/html/2503.13868v3)
[35](https://cameronrwolfe.substack.com/p/direct-preference-optimization)
[36](https://www.digitaldividedata.com/blog/fine-tuning-techniques-for-domain-specific-language-models)
[37](https://www.alignmentforum.org/posts/4XdxiqBsLKqiJ9xRM/llm-agi-may-reason-about-its-goals-and-discover)
