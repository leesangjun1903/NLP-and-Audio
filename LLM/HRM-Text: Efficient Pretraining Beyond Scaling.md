
# HRM-Text: Efficient Pretraining Beyond Scaling

> **논문 정보**
> - **제목**: HRM-Text: Efficient Pretraining Beyond Scaling
> - **arXiv ID**: [2605.20613](https://arxiv.org/abs/2605.20613)
> - **저자**: Guan Wang, Changling Liu, Chenyu Wang, Cai Zhou, Yuhao Sun, Yifei Wu, Shuai Zhen, Luca Scimeca, Yasin Abbasi Yadkori (Sapient Intelligence, MIT)
> - **발표**: 2026년 5월

---

## 1. 핵심 주장과 주요 기여 요약

현재 LLM(대규모 언어 모델)의 사전학습 패러다임은 막대한 컴퓨팅 자원과 인터넷 규모의 원시 텍스트에 의존하고 있어, 기초 연구에 대한 진입 장벽이 매우 높다.

이 논문의 핵심 주장은 다음 두 가지입니다:

1. **아키텍처 혁신**: 생물학적 시스템이 전두두정엽 루프(frontoparietal loop)와 같은 다중 시간 스케일 처리를 통해 고효율 샘플 학습을 수행한다는 것에서 영감을 받아, 표준 Transformer를 HRM(Hierarchical Recurrent Model)으로 대체하였으며, 이는 느리게 진화하는 전략적(slow/strategic) 레이어와 빠르게 진화하는 실행(fast/execution) 레이어로 연산을 분리한다.

2. **학습 목표 혁신**: 깊은 재귀 구조를 언어 모델링에 안정화하기 위해 MagicNorm과 warmup deep credit assignment를 도입하였으며, 표준 원시 텍스트 사전학습 대신 instruction-response 쌍만을 사용한 task-completion 목표와 PrefixLM 마스킹으로 학습한다.

**주요 성과**: 1B 파라미터 HRM-Text 모델이 단 400억 개의 고유 토큰과 $1,500의 예산으로 처음부터(from scratch) 학습하여 MMLU 60.7%, ARC-C 81.9%, DROP 82.2%, GSM8K 84.5%, MATH 56.2%를 달성하였다.

HRM-Text는 기존 대비 **130~600배 적은 컴퓨팅**, **150~900배 적은 데이터**로 사전학습을 가능하게 하는 완전한 사전학습 프레임워크를 제공한다.

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

현재 LLM 사전학습 패러다임은 막대한 컴퓨팅 자원과 인터넷 규모의 원시 텍스트에 의존하여 기초 연구에 심각한 장벽을 형성하고 있다.

구체적으로는:
- **데이터 비효율성**: 4T~36T 토큰 수준의 데이터 필요
- **연산 비효율성**: 수백만 달러의 학습 비용
- **연구 접근성 저하**: 소규모 연구소 및 개인 연구자의 진입 불가

단순한 구조적 사전(prior)만으로는 충분하지 않으며, 경쟁력 있는 성능을 달성하기 위해서는 아키텍처와 학습 방법의 종합적인 co-design이 필요하다는 것을 이 논문은 강조한다.

---

### 2-2. 제안하는 방법 (수식 포함)

#### (A) HRM 아키텍처 — 이중 시간 스케일 재귀 구조

HRM은 두 개의 Transformer 모듈인 H(고수준/느림)와 L(저수준/빠름)이 동일한 입력 임베딩에 대해 $H_{\text{cycles}} \times (L_{\text{cycles}} + 1)$ 스텝으로 반복하며, 가산적 상태 주입($z_L + z_H$)으로 결합한다. 이는 파라미터 수를 유지하면서 사실상 무한한 컴퓨팅 깊이를 달성한다.

수식으로 표현하면:

$$z_{\text{combined}} = z_L + z_H$$

$$\text{Total Depth} = H_{\text{cycles}} \times (L_{\text{cycles}} + 1)$$

표준 Transformer가 단일 스택의 단일 순전파(forward pass)를 수행하는 것과 달리, HRM-Text는 어떠한 출력이 생성되기 전에 연속적인 잠재 공간(continuous latent space)에서 두 스택이 중첩 재귀(nested recurrence)로 동작한다.

#### (B) MagicNorm — 안정화 정규화 기법

MagicNorm은 절단된 BPTT(Truncated Backpropagation Through Time) 하에서 순전파와 역전파의 계산 깊이 비대칭성을 활용하는 하이브리드 정규화 전략이다.

MagicNorm은 모듈 내부에 PreNorm을 적용하고, 모듈 출력에 추가적인 정규화를 적용하여 깊은 재귀 학습의 안정성을 향상시킨다.

수식으로 표현하면:

$$\text{MagicNorm}(x) = \text{Norm}_{\text{output}}(\text{Module}(\text{PreNorm}(x)))$$

이는 순전파 활성화 분산(forward activation variance)을 제한하면서 역전파 최적화 안정성(backward optimization stability)을 유지한다.

#### (C) Warmup Deep Credit Assignment

Warmup Deep Credit Assignment는 학습 초기에 마지막 2개의 재귀 스텝에만 그래디언트를 역전파하고, 이후 선형적으로 마지막 5개의 스텝까지 확장한다.

$$K_t = 2 + \lfloor 3 \cdot \frac{t}{T_{\text{warmup}}} \rfloor, \quad K_t \leq 5$$

여기서 $K_t$는 step $t$에서 역전파되는 재귀 단계의 수, $T_{\text{warmup}}$는 warmup 총 스텝 수이다.

#### (D) Task-Completion Objective + PrefixLM Masking

모델이 추론 시 주로 조건부 생성(conditional generation)에 사용되기 때문에, HRM-Text는 instruction-response 쌍에서 직접 사전학습하며, 응답 부분에만 Negative Log-Likelihood 손실을 계산하는 task-completion 목표를 최적화한다:

$$\mathcal{L} = -\log P(x_a \mid x_q)$$

여기서 $x_a$는 응답(answer/response) 토큰, $x_q$는 질의(query/instruction) 토큰이다.

PrefixLM 어텐션 마스크를 결합하여, instruction 토큰에 대해서는 완전한 양방향(bidirectional, encoder-like) 어텐션을 허용하고, 응답 생성에 대해서는 표준 인과적(causal) 생성을 유지한다.

$$\text{Attention Mask}(i, j) = \begin{cases} 1 & \text{if } i \in x_q \text{ (bidirectional)} \\ \mathbb{1}[j \leq i] & \text{if } i \in x_a \text{ (causal)} \end{cases}$$

---

### 2-3. 모델 구조

HRM-Text는 계층적 재귀 아키텍처, PrefixLM 시퀀스 패킹, FlashAttention 3 커널, PyTorch FSDP2 학습, 평가, 체크포인트 변환 도구로 구성된다.

| 구성 요소 | 설명 |
|---|---|
| H-module (slow) | 고수준 전략적 처리, slow-evolving |
| L-module (fast) | 저수준 실행 처리, fast-evolving |
| 상태 결합 | $z_L + z_H$ (가산적 주입) |
| 정규화 | MagicNorm (PreNorm + Output Norm) |
| 어텐션 | PrefixLM mask + FlashAttention 3 |
| 학습 목표 | Task-completion ( $-\log P(x_a \mid x_q)$ ) |

**학습 비용 기준:**
- L 크기 (0.6B 파라미터): H100 8개, 단일 노드, 약 50시간 (~$800)
- XL 크기 (1B 파라미터): H100 16개, 2노드, 약 46시간 (~$1,472)

---

### 2-4. 성능 향상

HRM-Text는 2B~7B 파라미터의 오픈소스 모델과 비견할 만한 성능을 달성하면서, 학습 토큰은 약 100~900배, 추정 컴퓨팅은 약 96~432배 적게 사용한다.

본 논문의 스케일링 실험은 Transformer의 경우 3B, HRM-Text의 경우 1B 파라미터까지 확장되었으며, 이 범위 내에서 제한된 데이터로 학습된 모델이 최대 36T 토큰을 사용한 산업 규모의 사전학습과 경쟁력을 유지할 수 있음을 보여준다.

Looped Transformer 및 RINS는 일반적으로 동일 크기의 Transformer보다 우수한 성능을 보이며, 재귀 또는 루프 연산이 효과적인 아키텍처 방향임을 보여준다. 동일한 학습 FLOPs 예산 대비 더 큰 Transformer와 비교할 경우 그 우위는 일부 줄어들지만, HRM은 이 아키텍처 탐색 공간에서 강력한 사례로 나타났다.

---

### 2-5. 한계

**① 스케일링 미검증:**
더 큰 모델 스케일에서의 유사한 효율성 이득을 입증하는 것은 향후 연구 과제로 남아 있다.

**② PrefixLM 엔지니어링 제약:**
PrefixLM은 실제 배포 시 일부 엔지니어링 구현 한계에 직면하며, vLLM 같은 표준 텍스트 생성 추론 프레임워크에서 실행할 때 prefill 단계에서 커스텀 어텐션 마스크를 지원해야 한다. 멀티턴 대화 시나리오로 확장할 경우, 사용자 세그먼트 내 양방향 가시성(bidirectional visibility)을 보장하면서 assistant 생성 프로세스에 대한 인과적 제약을 유지하기 위한 KV-cache 메커니즘의 추가 설계가 필요하다.

**③ 정렬(Alignment) 미완료:**
이 모델은 사전 정렬(pre-alignment) 체크포인트로, 멀티턴 대화 튜닝, 긴 문맥 적응, instruction 튜닝, RLHF 훈련, 또는 어시스턴트 스타일 활용을 위한 정렬이 적용되지 않은 상태이다.

---

## 3. 일반화 성능 향상 가능성

논문은 HRM의 효과가 재귀(recurrence)에서 기인하며, 이것이 유용한 내부 연산의 양을 증가시킨다고 가설을 제시한다.

일반화 성능 향상의 구체적 근거:

1. **Task-Completion 목표의 일반화 효과**: 표준 원시 텍스트 autoregressive 사전학습의 도그마에 도전하며, 모델이 주로 추론 시 조건부 생성에 사용되므로 instruction-response 쌍으로 직접 사전학습한다. 이는 학습과 추론 간 분포 격차(distribution gap)를 줄여 일반화에 유리하다.

2. **PrefixLM을 통한 맥락 이해 강화**: HRM-Text는 PrefixLM 마스크로 사전학습되어, 프롬프트 토큰들이 서로 양방향으로 어텐션하고 응답 토큰들은 인과적(causally)으로 어텐션한다. 이는 인코더-디코더 구조의 맥락 이해 이점을 취하면서도 생성 능력을 유지한다.

3. **다중 시간 스케일 처리**: 생물학적 시스템이 전두두정엽 루프 같은 다중 시간 스케일 처리를 통해 매우 효율적인 샘플 학습을 보여주는 것을 모방함으로써, 추상적 패턴과 세부 실행을 분리하여 처리하는 능력이 다양한 태스크에 대한 일반화를 돕는다.

4. **Few-shot 성능**: NLP 태스크(분류, 추출, 구조화 출력, 단답형 QA)에서는 2~8개의 few-shot 예시를 활용한 직접 조건(direct condition) 방식이 가장 강력하며, direct + few-shot이 추가 학습 없이 가장 강력한 설정으로 측정되었다.

5. **스케일 내 일반화 근거**: 구조적 사전(structural prior)과 목표 지향적 학습 목표(targeted training objectives)가 사전학습의 장벽을 크게 낮출 수 있으며, 이 학습 방식으로 기초 모델을 처음부터 학습하는 것이 실현 가능하다.

---

## 4. 연구에 미치는 영향 및 향후 고려 사항

### 4-1. 향후 연구에 미치는 영향

| 분야 | 영향 |
|---|---|
| **민주화(Democratization)** | 소규모 연구소나 개인 연구자도 기초 모델 사전학습 접근 가능 |
| **아키텍처 패러다임 전환** | Transformer 중심에서 재귀·계층적 구조로의 전환 촉진 |
| **학습 목표 재설계** | 원시 텍스트 next-token prediction에서 task-completion 목표로의 전환 논의 확산 |
| **생물학적 영감 AI** | Frontoparietal loop 등 뇌과학 기반 아키텍처 설계 활성화 |

Sapient Intelligence는 전통적 AI 프레임워크의 구조적 한계를 극복하는 차세대 뇌 영감 AI 아키텍처를 개발하고 있으며, 강화학습(RL), 진화 알고리즘, 신경역학 원리를 통합하여 고급 논리 추론, 평생 학습, 높은 해석 가능성을 가진 모델을 개발하고 있다.

Sapient Intelligence는 2025년 6월 HRM을 처음 발표했으며, 이는 수학 문제와 스도쿠를 포함한 복잡한 태스크에서 뛰어난 추론 성능을 보여주었고, ARC-AGI 챌린지에서 수만 배 적은 파라미터로 DeepSeek R1 및 OpenAI o3를 능가하였다.

### 4-2. 향후 연구 시 고려할 점

1. **대규모 스케일링 검증**: 더 큰 모델 스케일에서의 유사한 효율성 이득을 입증하는 것이 향후 연구 과제로, 3B, 7B, 70B 파라미터 스케일로의 확장 검증이 필요하다.

2. **멀티턴 대화 및 KV-Cache 설계**: 멀티턴 대화 시나리오 확장 시, 사용자 세그먼트 내 양방향 가시성을 보장하면서 assistant 생성에 대한 인과적 제약을 유지하는 KV-cache 메커니즘의 추가 설계가 필요하다.

3. **정렬(Alignment) 파이프라인 구축**: HRM-Text를 채팅 모델처럼 사용하려면 태스크별 데이터에 대한 SFT 및/또는 RL 같은 추가 정렬을 수행해야 한다.

4. **훈련 데이터 도메인 다양성**: 현재 instruction-response 쌍 중심 학습이 특정 도메인 편향을 초래할 수 있어, 도메인 커버리지 다양화 연구가 필요하다.

5. **재현성 및 하드웨어 의존성**: Hopper 클래스 GPU가 FlashAttention 3 의존성으로 인한 기대 학습 타겟이므로, 다양한 하드웨어 환경에서의 이식성 연구가 필요하다.

6. **RLHF 및 장문맥 적응**: 사전학습 이후의 강화학습 기반 정렬과 긴 문맥 처리 능력 향상이 중요한 후속 과제이다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 핵심 접근법 | 토큰 수 | 특징 |
|---|---|---|---|
| **GPT-3** (Brown et al., 2020) | 대규모 Transformer, few-shot | ~300B | 스케일링 법칙 확립 |
| **Chinchilla** (Hoffmann et al., 2022) | 최적 토큰-파라미터 비율 | ~1.4T | Compute-optimal training |
| **LLaMA 2** (Touvron et al., 2023) | 오픈소스 Transformer | 2T | 오픈소스 접근성 향상 |
| **Mamba** (Gu & Dao, 2023) | SSM 기반 선형 재귀 | ~300B | 선형 복잡도 재귀 |
| **Looped Transformer** (2023~) | 가중치 공유 반복 | 가변 | 파라미터 효율 재귀 |
| **TinyLlama** (Zhang et al., 2024) | 소규모 Transformer | 3T | 소형 효율 모델 |
| **HRM-Text** (Wang et al., 2026) | HRM + Task-completion | **~40B** | **150~900배 데이터 절감** |

Looped Transformer 및 RINS는 일반적으로 동일 크기의 Transformer를 능가하나, 동일한 학습 FLOPs 예산 대비 더 큰 Transformer와 비교할 경우 그 우위는 일관되지 않다.

HRM은 모든 스케일에서 안정적인 학습 역학을 유지하는 반면, 유사한 구조의 TRM(Twin Recurrent Model)은 1B 파라미터 스케일에서 심각한 불안정성을 겪는다.

HRM-Text의 가장 차별화된 점은 아키텍처 개선과 학습 목표 재설계를 **동시에(holistic co-design)** 수행했다는 점이며, 소규모 task-completion 지향 사전학습 실행이, 훨씬 더 많은 토큰과 컴퓨팅 예산으로 학습된 오픈 모델의 성능 범위에 진입할 수 있음을 보여준다.

---

## 📚 참고 자료 및 출처

1. **논문 원문**: Wang, G. et al. (2026). *HRM-Text: Efficient Pretraining Beyond Scaling*. arXiv:2605.20613. https://arxiv.org/abs/2605.20613
2. **논문 HTML 전문**: https://arxiv.org/html/2605.20613v1
3. **공식 GitHub 저장소**: https://github.com/sapientinc/HRM-Text
4. **Hugging Face 모델 카드**: https://huggingface.co/sapientinc/HRM-Text-1B
5. **공식 보도자료**: Sapient Intelligence, PR Newswire (May 18, 2026). https://www.prnewswire.com/news-releases/sapient-intelligence-launches-hrm-text-challenging-the-llm-monopoly-with-a-brain-inspired-foundation-model-trained-on-up-to-1000x-fewer-tokens-302774638.html
6. **기술 분석 기사**: KuCoin News — *Tsinghua alumnus Wang Guan's HRM-Text achieves SOTA with 1/900 tokens and 1/432 compute*. https://www.kucoin.com/news/flash/tsinghua-alumnus-wang-guan-s-hrm-text-achieves-sota-with-1-900-token-and-1-432-compute
7. **재현 가이드**: OpenTrain.ai. https://www.opentrain.ai/papers/hrm-text-efficient-pretraining-beyond-scaling--arxiv-2605.20613/
8. **공식 웹사이트**: Sapient Intelligence. https://sapient.inc/hrm-text/

> ⚠️ **정확도 고지**: 본 답변은 공개된 arXiv 논문 초록, GitHub 공식 저장소, Hugging Face 모델 카드, 공식 보도자료를 기반으로 작성되었습니다. 논문 본문의 일부 상세 수식(특히 MagicNorm의 정확한 수식)은 공개된 출처에서 완전히 확인되지 않아, 논문에서 기술된 원리를 기반으로 재구성하였습니다. 정확한 수식 확인을 위해서는 [arXiv 원문](https://arxiv.org/abs/2605.20613)을 직접 참고하시기 바랍니다.
