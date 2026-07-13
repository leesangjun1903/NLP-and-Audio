# Emergent Abilities of Large Language Models

## 1. 핵심 주장과 주요 기여

이 논문(Wei et al., 2022, TMLR)의 핵심 주장은 다음과 같습니다:

**핵심 정의**: "창발적 능력(emergent ability)은 작은 모델에는 존재하지 않지만 큰 모델에는 존재하는 능력"이며, 이는 작은 모델의 성능을 단순히 외삽(extrapolation)해서는 예측할 수 없다.

**주요 기여**:
- Philip Anderson의 "More Is Different"(1972) 개념을 언어모델 스케일링에 적용
- Few-shot prompting과 augmented prompting(chain-of-thought, instruction tuning 등)에서 수십 개의 창발적 능력 사례를 체계적으로 정리
- GPT-3, LaMDA, Gopher, Chinchilla, PaLM 등 5개 모델 패밀리에 걸친 실증적 증거 제시
- 창발 현상에 대한 설명 시도(evaluation metric의 한계, cross-entropy loss 분석 등)와 향후 연구 방향 제시

이 논문은 새로운 방법론을 제안하지 않고 **기존 문헌을 서베이/종합**하는 성격의 논문입니다.

---

## 2. 문제, 방법론, 모델 구조, 성능, 한계

### 해결하고자 하는 문제
기존 스케일링 법칙(scaling law) 연구는 cross-entropy loss가 계산량에 따라 매끄럽게(smooth) 개선된다고 보고했으나(Kaplan et al., 2020), 특정 downstream task 성능은 예측 불가능하게 특정 임계 규모에서 갑자기 나타나는 현상을 설명하고자 함.

### 정의 (수식적 표현)
논문은 정형화된 수식을 제시하지 않지만, 창발성의 정의를 다음과 같이 형식화할 수 있습니다:

모델 성능 $P(N)$을 파라미터 수 또는 학습 FLOPs $N$의 함수라 할 때, 태스크가 창발적이라 함은:

$$
P(N) \approx P_{random} \quad \text{for } N < N_c
$$
$$
P(N) \gg P_{random} \quad \text{for } N \geq N_c
$$

여기서 $N_c$는 임계 스케일(critical threshold)이며, 이 전이는 **위상 전이(phase transition)** 형태로 나타남 (Huberman & Hogg, 1987 참조).

FLOPs 계산은 다음과 같이 근사됩니다 (Kaplan et al., 2020):

$$
C \approx 6ND
$$

( $C$: 훈련 FLOPs, $N$: 파라미터 수, $D$: 훈련 토큰 수)

### 모델 구조
논문 자체는 새로운 모델을 제안하지 않고, 기존 Transformer 기반 LM들(GPT-3, LaMDA, Gopher, Chinchilla, PaLM)의 스케일링 곡선을 비교·분석함. Table 2에 각 모델의 파라미터 수, 학습 토큰 수, 훈련 FLOPs가 정리되어 있음(예: PaLM 540B는 $2.53\times10^{24}$ FLOPs).

### 성능 향상 사례 (Table 1 요약)
- **Few-shot prompting**: 3자리 덧셈/뺄셈 - GPT-3 13B(2.3×10²² FLOPs)에서 창발
- **MMLU**: GPT-3 175B, Chinchilla 70B에서 창발 (57개 주제 평균)
- **TruthfulQA**: Gopher 280B에서 창발 (20% 이상 향상)
- **Chain-of-thought**: LaMDA 68B(1.3×10²³ FLOPs)에서 표준 프롬프팅을 능가
- **Instruction tuning**: FLAN 68B 이하에서는 오히려 성능 저하, 그 이상에서 개선

### 한계점 (논문이 명시)
1. **평가 지표의 문제**: Exact match와 같은 엄격한 지표가 점진적 개선을 마스킹할 가능성 (Appendix A.1에서 cross-entropy loss 분석 시도했으나 완전한 설명 실패)
2. **스케일이 유일한 요인 아님**: PaLM 62B가 GPT-3 175B/LaMDA 137B보다 파라미터가 적음에도 14개 BIG-Bench 태스크에서 창발 (데이터 품질, 아키텍처 차이 때문일 가능성)
3. **예측 불가능성 자체가 한계**: 왜 창발이 일어나는지에 대한 완전한 이론적 설명 부재
4. **위험(risk)의 창발**: 능력뿐 아니라 편향, 독성, 허위정보 등 위험도 창발 가능

---

## 3. 모델의 일반화 성능 향상 가능성 (중점 논의)

### 3.1 스케일을 넘어선 일반화 (Section 5.2 "Beyond scaling")

논문은 일반화 성능이 **단순 파라미터 수 증가만이 아니라 여러 요인의 조합**에서 비롯됨을 강조합니다:

- **데이터 품질**: PaLM은 LaMDA보다 다국어·코드 데이터 비중이 높아 더 적은 파라미터로도 창발 달성
- **아키텍처 개선**: split digit-encoding 등 세부 아키텍처 차이
- **사전학습 목적함수**: Mixture-of-denoisers 방식의 continued pretraining이 적은 추가 계산량(0.1%)으로 BIG-Bench 창발 성능 유도 (Tay et al., 2022c)
- **파인튜닝 전략**: Instruction tuning은 원래 68B 이상에서만 작동했으나, Sanh et al.(2022)이 encoder-decoder 구조의 11B 모델에서도 유사 효과 유도 → **일반화 능력의 스케일 문턱값을 낮출 수 있음**을 시사
- **RLHF**: InstructGPT의 1.3B 모델이 인간 평가에서 훨씬 큰 모델을 능가 (Ouyang et al., 2022)

### 3.2 다른 관점에서의 창발 (Section 5.3)

논문은 WikiText103 perplexity를 x축으로 사용한 대안적 분석(Figure 4)을 제시하며, 일반화 성능이 다음과 같은 다변수 함수로 봐야 한다고 주장:

$$
\text{Performance} = f(\text{FLOPs}, \text{Params}, \text{Data quality}, \text{Perplexity}, \ldots)
$$

이는 retrieval-augmented 모델처럼 적은 계산량으로도 낮은 perplexity를 달성하는 구조가 일반화 성능 향상의 새로운 경로가 될 수 있음을 시사합니다.

### 3.3 프롬프팅 기법을 통한 일반화 확장

Chain-of-thought, least-to-most prompting, self-consistency 등은 **모델 파라미터를 바꾸지 않고도** 특정 임계 규모 이상에서 일반화 성능(다단계 추론)을 크게 향상시킴. 이는 일반화 성능이 순수 스케일이 아니라 **추론 전략과 스케일의 상호작용**에서 나타남을 보여줍니다.

### 3.4 Frontier tasks와 향후 일반화 가능성 (Section 5.6)

BIG-Bench에는 현재 최대 모델(GPT-3, PaLM)도 무작위 수준 성능밖에 내지 못하는 수십 개 태스크가 존재(체스, 고난도 수학 등, Appendix E.4). 저자들은 이것이 **향후 스케일링이나 새로운 훈련 방법으로 일반화 성능이 추가로 확장될 잠재적 후보**라고 제안합니다.

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 연구에 미친 영향
1. **Chain-of-Thought, Instruction Tuning 등 후속 연구의 이론적 근거 제공**: 이 논문 이후 "창발적 능력이 언제, 왜 나타나는가"를 규명하려는 연구가 폭발적으로 증가 (예: Chung et al. 2022 Scaling Instruction-Finetuned LMs)
2. **비판적 후속 연구 촉발**: Schaeffer et al. (2023, NeurIPS) "Are Emergent Abilities of Large Language Models a Mirage?"는 본 논문에서 관찰된 창발이 **비선형적 평가지표(exact match 등) 선택의 인공물(artifact)**이라고 반박. Linear metric(예: token edit distance)으로 바꾸면 창발이 완만한 곡선으로 나타남을 보임
3. **Chinchilla 스케일링 법칙(Hoffmann et al., 2022)과의 결합**: "compute-optimal" 훈련이 창발 임계값에 미치는 영향에 대한 논의 확대
4. **AI 안전성 연구**: 창발적 위험(emergent risks) 개념이 Anthropic, DeepMind 등의 정렬(alignment) 연구 의제에 반영됨 (Ganguli et al., 2022 Inverse Scaling Prize 등)
5. **Mixture-of-Experts, Retrieval-augmented 모델**: 순수 파라미터 스케일링의 한계를 넘어서려는 아키텍처 연구(Switch Transformer, RETRO 등)에 방향성 제공

### 향후 연구 시 고려할 점
1. **평가지표 선택의 신중함**: Schaeffer et al.(2023)의 지적처럼, 창발 현상이 실제 능력의 불연속적 발생인지, 평가지표(비연속적 accuracy/exact-match)의 인공물인지 구분 필요. Brier score, cross-entropy 등 연속적 지표와 병행 보고 권장
2. **다변수적 스케일 정의**: 파라미터 수, FLOPs 뿐 아니라 데이터 품질, 데이터량, 아키텍처, 훈련 목적함수를 모두 통제한 ablation 연구 필요
3. **작은 모델에서의 재현 가능성 탐구**: 창발이 "고정된 속성"이 아니라 훈련 방법 개선으로 낮은 스케일에서도 재현 가능하다는 점(Sanh et al. 2022 사례)을 고려해, 효율적 소형 모델 연구의 가치를 저평가하지 않아야 함
4. **창발적 위험의 사전 예측 체계 구축**: 능력의 창발과 마찬가지로 편향·독성·기만 등 위험도 예측 불가능하게 나타날 수 있으므로, red-teaming과 forecasting 방법론의 병행 발전 필요
5. **다국어/다중모달 창발**: Persian QA 사례처럼 특정 언어/모달리티에서는 훈련데이터 구성이 스케일보다 더 결정적일 수 있음 - 향후 다국어 공정성 연구에서 중요 고려사항

---

## 2020년 이후 관련 최신 연구 비교분석

| 연구 | 연도 | 핵심 주장 | 본 논문과의 관계 |
|---|---|---|---|
| Kaplan et al., "Scaling Laws for Neural LMs" | 2020 | Loss는 계산량에 대해 power-law로 매끄럽게 감소 | 본 논문의 배경이 되는 대조적 발견 |
| Hoffmann et al. (Chinchilla), "Training Compute-Optimal LLMs" | 2022 | 기존 스케일링법칙은 데이터량을 과소평가, compute-optimal 훈련 필요 | 본 논문에서 창발 임계값 계산에 활용된 모델 |
| Wei et al., "Chain-of-Thought Prompting" | 2022 | CoT는 대규모 모델에서만 효과적 (창발적 기법의 대표 사례) | 본 논문 Section 4의 핵심 근거 |
| **Schaeffer, Miranda, Koyejo, "Are Emergent Abilities of LLMs a Mirage?"** | **2023 (NeurIPS)** | **창발은 실제 현상이 아니라 비연속 metric 선택의 인공물(mirage)이라고 반박** | **본 논문에 대한 가장 강력한 반론** |
| Tay et al., "Transcending Scaling Laws with 0.1% Extra Compute" (U-PaLM) | 2022 | Mixture-of-denoisers 목적함수로 적은 추가 계산으로 창발 유도 가능 | 본 논문 Section 5.2에서 인용, "beyond scaling" 근거 |
| Chung et al., "Scaling Instruction-Finetuned Language Models" (Flan-PaLM) | 2022 | Instruction tuning의 스케일링 효과를 체계적으로 확장 검증 | 본 논문의 instruction following 창발 논의를 확장 |
| Ganguli et al., "Predictability and Surprise in Large Generative Models" | 2022 | 스케일링의 예측가능성과 창발적 위험을 동시에 논의 | 본 논문의 "Emergent risks" 절과 직접 연관 |

**비교분석 요약**: 본 논문(2022)이 창발 현상을 처음 체계적으로 정리한 이후, 가장 중요한 후속 논쟁은 **Schaeffer et al. (2023)의 "Mirage" 논문**입니다. 이들은 동일한 벤치마크에서 metric을 accuracy(비연속)에서 token-level edit distance나 log-likelihood(연속)로 바꾸면 창발적 도약이 사라지고 매끄러운 곡선이 나타남을 보였습니다. 이는 본 논문의 Appendix A.1(cross-entropy loss 분석)에서 저자들 스스로 일부 인지했던 한계("evaluation metric이 창발처럼 보이게 할 수 있다")를 정면으로 검증한 것으로, 현재 학계에서는 **"창발이 실재하는 현상인가, 측정의 산물인가"**가 지속적인 논쟁거리로 남아 있습니다.

---

## 참고자료 출처

본 답변은 다음 문서에 기반하여 작성되었습니다:
- **원문**: Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, William Fedus. **"Emergent Abilities of Large Language Models."** *Transactions on Machine Learning Research (TMLR)*, 08/2022. arXiv:2206.07682v2 [cs.CL], 26 Oct 2022. (제공된 PDF 문서 전체 활용)

**추가 언급 참고문헌** (원 논문 References 섹션에서 인용, 제공 문서에 포함됨):
- Kaplan et al., "Scaling Laws for Neural Language Models," arXiv:2001.08361, 2020
- Hoffmann et al., "Training Compute-Optimal Large Language Models" (Chinchilla), NeurIPS 2022, arXiv:2203.15556
- Wei et al., "Chain of Thought Prompting Elicits Reasoning in Large Language Models," NeurIPS 2022, arXiv:2201.11903
- Chowdhery et al., "PaLM: Scaling Language Modeling with Pathways," arXiv:2204.02311, 2022
- Tay et al., "Transcending Scaling Laws with 0.1% Extra Compute," arXiv:2210.11399, 2022
- Chung et al., "Scaling Instruction-Finetuned Language Models," arXiv:2210.11416, 2022
- BIG-Bench, "Beyond the Imitation Game," arXiv:2206.04615, 2022

**본 답변 작성자가 별도로 인지하고 있는 외부 최신 연구** (제공된 PDF에는 포함되지 않았으나, 학계에서 널리 알려진 후속 비판 연구로서 신뢰도 높게 언급):
- Schaeffer, R., Miranda, B., & Koyejo, S. **"Are Emergent Abilities of Large Language Models a Mirage?"** NeurIPS 2023. (arXiv:2304.15004) — ※ 이 논문의 구체적 수치나 세부 실험 내용까지는 제공된 문서에 없으므로, 정확한 인용이 필요한 경우 원문 확인을 권장합니다.

*참고: 위 "2020년 이후 최신 연구 비교분석" 표에서 Schaeffer et al. 관련 내용은 제가 학습한 일반 지식에 기반한 것으로, 제공된 PDF 문서에는 포함되어 있지 않습니다. 정확한 세부 내용(구체적 실험 결과, 수치 등)은 원 논문을 직접 확인하시기 바랍니다.*
