
# Over-Tokenized Transformer: Vocabulary is Generally Worth Scaling

**📄 논문 정보**
- **저자:** Hongzhi Huang, Defa Zhu, Banggu Wu, Yutao Zeng, Ya Wang, Qiyang Min, Xun Zhou (ByteDance Seed Team)
- **arXiv:** [2501.16975](https://arxiv.org/abs/2501.16975) (2025년 1월 28일)
- **학회:** ICML 2025 accepted

---

## 1. 핵심 주장과 주요 기여 요약

### 🔑 핵심 주장

토크나이제이션(Tokenization)은 대형 언어 모델(LLM)의 핵심 구성 요소임에도 불구하고, 모델 스케일링과 성능에 미치는 영향이 충분히 탐구되지 않았다. 이 논문은 **입력 어휘(input vocabulary)와 출력 어휘(output vocabulary)를 분리(decouple)**하여 언어 모델 성능을 향상시키는 새로운 프레임워크인 **Over-Tokenized Transformer**를 소개한다. 특히, 입력 어휘를 확장하여 멀티-그램(multi-gram) 토큰을 활용한다.

### 📌 주요 기여

광범위한 실험을 통해 **입력 어휘 크기와 학습 손실 사이의 로그-선형(log-linear) 관계**를 발견하였으며, 이는 더 큰 입력 어휘가 모델 크기에 관계없이 지속적으로 모델 성능을 향상시킴을 보여준다. 큰 입력 어휘를 사용함으로써 **추가 비용 없이 두 배 크기의 베이스라인 모델과 동등한 성능**을 달성하였고, 이는 스케일링 법칙에서 토크나이제이션의 중요성을 강조하며 토크나이저 설계에 실질적인 통찰을 제공한다.

---

## 2. 상세 설명

### 2-1. 해결하고자 하는 문제

Context-Free Grammar(CFG) 모델링에 대한 합성 실험에서 시작하여 다양한 규모의 모델에 대한 토큰 세분성과 어휘 크기의 영향을 체계적으로 분석하였다. 그 결과 더 큰 토크나이저는 **대형 모델 성능을 향상**시키지만 **소형 모델에는 어려움**을 야기하는 것으로 나타났다. 나아가, 입력과 출력 어휘를 분리하면 **입력 어휘만 확장했을 때는 모델이 꾸준히 향상**되는 반면, 더 큰 출력 어휘는 소형 모델에 오히려 해가 될 수 있다는 사실을 발견하였다.

즉, 기존 접근법은 입력/출력 어휘를 항상 동일하게 유지했기 때문에 이 두 요소가 모델에 미치는 영향의 **비대칭성**을 활용하지 못했다는 것이 문제의식이다.

---

### 2-2. 제안하는 방법 (수식 포함)

#### ① Over-Encoding (OE) — 입력 어휘 확장

Over-Encoding은 **계층적 인코딩 패러다임(hierarchical encoding paradigm)**을 사용하며, 구체적으로 GPT 모델에 대한 입력 임베딩을 1-gram, 2-gram, … 의 합산으로 계산한다.

수식으로 표현하면:

$$
\mathbf{e}_{x_t} = \sum_{n=1}^{N} \mathbf{e}^{(n)}_{x_{t-n+1:t}}
$$

여기서:
- $\mathbf{e}_{x_t}$: 위치 $t$에서의 최종 입력 임베딩
- $\mathbf{e}^{(n)}\_{x_{t-n+1:t}}$: $n$-gram 토큰에 해당하는 임베딩
- $N$: 최대 n-gram 차수

이는 스케일링 법칙의 새로운 차원을 대표하며, 임베딩 파라미터를 새로운 확장 가능한 희소 차원(scalable sparse dimension)으로 나타낸다.

##### Over-Encoding의 효율성

Over-Encoding의 가장 큰 장점은 **어휘 입력이 모델 아키텍처로부터 분리(decouple)**되어 있다는 점이다. 이 분리 덕분에 다음 마이크로 배치에 대한 임베딩 룩업을 미리 수행할 수 있어, 파이프라인-병렬 훈련 프레임워크에서 임베딩 룩업에 필요한 통신을 현재 마이크로 배치의 트랜스포머 순전파와 겹치도록 설계할 수 있다. 또한 Over-Encoding 파라미터를 CPU로 오프로드하여 GPU 메모리 압박을 완전히 해소할 수 있다.

---

#### ② Over-Decoding (OD) — 출력 어휘 확장

Over-Decoding은 더 큰 출력 어휘를 활용하여 더 세밀한 감독(fine-grained supervision)을 제공하는 개념이다.

CFG 실험에서 얻은 결론에 따르면 추가 토큰 디코딩은 **충분히 큰 모델에서만 효과적**이다. 실제로 Multi-Token Prediction(MTP, Gloeckle et al., 2024)에 관한 이전 연구들도 Over-Decoding의 근사치이며, 대형 모델만이 미래 토큰 예측으로부터 이익을 얻는다는 동일한 결론을 공유한다. 이 논문에서는 MTP-류 방법들을 Over-Decoding으로 간주한다.

Over-Decoding의 훈련 손실은 다음과 같이 정의된다:

$$
\mathcal{L}_{\text{OD}} = \sum_{k=1}^{K} \lambda_k \cdot \text{CE}\left(\hat{y}_{t+k},\, y_{t+k}\right)
$$

여기서:
- $K$: 예측하는 미래 토큰 수
- $\lambda_k$: 각 예측 헤드에 대한 재가중치 하이퍼파라미터
- $\text{CE}(\cdot, \cdot)$: 크로스 엔트로피 손실
- $\hat{y}_{t+k}$: 위치 $t+k$에 대한 예측값

주목할 만한 도전 과제는 디코딩 임베딩이 밀집 활성화(densely activated)되어 특히 소형 모델에서 비임베딩(unembedding) 레이어의 계산 비용이 매우 높다는 점이다. 이를 해결하기 위해 마지막 은닉 상태를 더 작은 차원으로 투영하는 **저순위 분해(low-rank decomposition)**를 선택적으로 적용한다.

---

#### ③ Over-Tokenized Transformer (OT) — 통합 프레임워크

Over-Tokenized Transformer(OT)는 Over-Encoding과 Multi-Token Prediction을 결합한 모델이다.

전체 훈련 목표:

$$
\mathcal{L}_{\text{OT}} = \mathcal{L}_{\text{OE}} + \alpha \cdot \mathcal{L}_{\text{OD}}
$$

여기서:
- $\mathcal{L}_{\text{OE}}$: 표준 다음 토큰 예측 손실 (Over-Encoding 기반 입력 임베딩 사용)
- $\mathcal{L}_{\text{OD}}$: Over-Decoding 손실 (미래 토큰 예측 손실)
- $\alpha$: 균형 하이퍼파라미터

---

### 2-3. 모델 구조

실험에 사용된 아키텍처:

OLMo2(OLMo et al., 2024)의 실험 설정을 따르되, 아키텍처를 수정하여 **151M, 400M, 1B**의 밀집 파라미터를 가진 모델인 OLMo2-151M, OLMo2-400M, OLMo2-1B를 얻었다.

OLMo2-151M 및 OLMo2-400M 모델은 400B 토큰으로, OLMo2-1B는 1T 토큰으로 학습되었다. 또한 OE-12.8M 모델 외에도 어휘 크기에 대한 대략적인 스케일링 경향을 파악하기 위해 OE-1.2M 실험도 수행하였다.

| 구성 요소 | 세부 내용 |
|---|---|
| 기반 모델 | OLMo2, OLMoE (MoE 모델) |
| Over-Encoding 어휘 크기 | 1.2M ~ 12.8M (기본 100K 대비 최대 128배) |
| n-gram 범위 | 1-gram ~ N-gram 계층적 합산 |
| Over-Decoding | MTP 방식, 2-gram 예측 + product decomposition |

---

### 2-4. 성능 향상

연구자들은 OLMo 및 OLMoE 아키텍처에서 실험을 수행하여 세 가지 핵심 발견을 도출하였다. **로그-선형 스케일링**: 입력 어휘 크기가 지수적으로 증가함에 따라 학습 손실이 선형적으로 감소하였다. **400M 파라미터 모델**에 1280만 개의 입력 어휘를 적용하면 1B 파라미터 베이스라인과 동등한 성능을 달성하여, 동일한 계산 비용으로 **2.5배의 효과적 스케일링**을 실현하였다.

**수렴 가속화**: Over-Encoding은 MMLU 및 PIQA와 같은 작업에서 수렴에 필요한 학습 단계를 **3~5배 감소**시켰다. **희소 파라미터 효율성**: 128배 더 큰 입력 어휘를 사용했음에도 불구하고, 희소 임베딩 접근 및 최적화된 샤딩 전략 덕분에 메모리 및 계산 오버헤드가 **5% 미만** 증가하였다.

다양한 모델 유형에 걸쳐 일관된 성능 향상이 나타났다. 밀집 모델의 경우, 151M Over-Encoded(OE) 모델은 베이스라인 대비 **14% 퍼플렉서티 감소**를 달성하였다.

다운스트림 평가에서도 OE는 MMLU-Var에서 **3.2배**, Hellaswag에서 **3.0배**, ARC-Challenge에서 **2.6배**, ARC-Easy에서 **3.1배**, PIQA에서 **3.9배**의 속도 향상을 보였다.

---

### 2-5. 한계점

Over-Decoding 접근법은 **충분히 큰 모델에서만 실용적**이다. 대형 모델은 비임베딩에 소비되는 계산 비율이 더 작고, 이 설정을 활용할 수 있는 더 큰 용량을 갖추고 있기 때문이다.

또한 극단적인 어휘 스케일링 시 메모리 또는 룩업 비용이 소폭 증가하고, 소형 모델에서는 출력 측 설계에 세심한 주의가 필요하며, 과도하게 큰 출력 어휘는 성능 저하를 야기할 수 있다.

더 큰 입력 어휘는 모든 이득을 얻기 위해 더 긴 학습이 필요하다. OLMoE-7B의 경우 500B 토큰 학습 후에도 OE의 이점이 완전히 수렴되지 않았으며, 더 큰 모델은 Over-Encoding의 파워를 완전히 활용하기 위해 더 많은 학습을 필요로 할 가능성이 있다.

---

## 3. 일반화 성능 향상 가능성

Over-Decoding 관점에서, 학습 과정이 더 **세밀한 감독 신호(finer-grained supervision signals)**를 제공하여 모델이 더 나은 표현(representation)을 학습할 수 있게 한다. 그러나 이 접근법은 충분히 큰 모델에서만 실용적이며, 대형 모델은 비임베딩에 소비되는 계산 비율이 더 작고, 이 설정을 활용할 더 큰 용량을 갖추고 있다.

연구에 따르면 더 큰 어휘로 학습된 모델은 더 빠르게 학습하고, 성능 향상이 모델 크기에 따라 확장되며, **다양한 언어**에서도 이점이 유지되고, **일반 작업과 특수 작업 모두**에서 Over-Tokenization이 효과적임이 증명되었다.

어휘 크기를 확장하면 모델 성능이 일관되게 향상되며, 모델이 커질수록 그 혜택은 더욱 두드러진다.

사전 토크나이제이션(pre-tokenization) 규칙은 형태론 및 문법에 기반한 귀납적 편향(inductive bias)으로 작용하며, 토크나이저가 자연어의 보편적인 통계 패턴을 포착하도록 도와줌으로써 트랜스포머 언어 모델이 **미학습 데이터(unseen data)에 일반화**하는 데 중요한 역할을 한다.

즉, Over-Encoding이 제공하는 계층적 n-gram 표현은 문맥에 대한 풍부한 선험적 정보를 모델에 전달하여:
- 특정 도메인에 과적합(overfitting)될 위험을 줄이고
- 더 적은 학습 데이터로도 강건한 언어 이해 능력을 습득할 수 있게 하며
- 다국어, 도메인 특화 태스크, 제로샷(zero-shot) 설정 등 다양한 환경에서의 일반화 가능성을 높인다.

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 4-1. 향후 연구에 미치는 영향

이 연구는 토크나이제이션을 언어 모델 설계의 확장 가능한 차원으로 재정의한다. 입력과 출력 어휘를 분리함으로써 Over-Tokenized Transformer는 전통적인 트레이드오프를 극복하고, 소형 모델도 지나치게 복잡한 예측 작업 없이 압축된 입력 시퀀스의 이점을 누릴 수 있다. 입력 어휘 크기와 성능 사이의 로그-선형 관계는 임베딩 파라미터가 **기존의 모델 깊이·너비와 함께 스케일링 법칙의 새로운 축**임을 시사한다.

이 연구 결과는 미래 언어 모델이 어휘 크기를 **모델 깊이와 너비와 함께 핵심 스케일링 파라미터**로 고려해야 함을 시사한다.

어휘 스케일링이 왜 그토록 잘 작동하는지를 설명하고, 다양한 사용 사례에 대한 최적의 어휘 스케일링 비율을 파악하기 위한 **토크나이제이션 이론에 대한 추가 연구**가 필요하다.

### 4-2. 앞으로 연구 시 고려할 점

| 고려 사항 | 내용 |
|---|---|
| **스케일링 법칙 재정립** | 어휘 크기를 파라미터 수, 연산량(FLOPs), 데이터 크기와 함께 스케일링 공식에 통합할 필요 있음 |
| **소형 모델 적용 전략** | Over-Decoding은 대형 모델에서만 유효하므로, 소형 모델을 위한 대안적 출력 어휘 확장 전략 필요 |
| **멀티모달 확장** | 고급 디코딩 감독, 초대형 임베딩 분해, **크로스-모달(cross-modal) Over-Tokenization** 등의 방향 탐색 필요 |
| **수렴 속도 vs. 계산 비용** | 더 큰 입력 어휘는 더 긴 학습이 필요하므로, 최적 어휘 크기와 학습 스케줄의 공동 최적화 필요 |
| **이론적 근거 마련** | 왜 n-gram 계층 합산이 효과적인지에 대한 수학적 분석 부족 |
| **기존 모델과의 호환성** | Over-Encoding 통합이 최소한의 코드 변경만 필요하며 즉각적인 효율성 향상을 제공하지만, 기존에 학습된 대규모 모델에 어떻게 적용할지에 대한 연구 필요 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 접근법 | 어휘 처리 방식 | 특징 |
|---|---|---|---|
| **ByT5** (Xue et al., 2022) | 바이트 레벨 | 어휘 없음 (byte-level) | 토크나이저 불필요, 시퀀스 길이 증가 |
| **BPE Scaling** (Guo et al., 2023) | BPE 어휘 확장 | 단일 어휘 확장 | 입출력 어휘 동시 확장, 소형 모델 성능 저하 |
| **Multi-Token Prediction (MTP)** (Gloeckle et al., 2024) | 미래 토큰 예측 | 출력 어휘 확장 | 대형 모델에서만 효과적 |
| **Byte Latent Transformer** (Meta, 2024) | 패치 기반 | 동적 바이트 그룹화 | "패치가 토큰보다 더 잘 스케일링됨"을 주장 |
| **Scone / f-gram Model** (2025) | f-gram 임베딩 | 입력 임베딩 확장 | 입출력 임베딩을 분리하여 기본 어휘의 각 토큰을 $n$-gram으로 보강, 컨텍스트화된 토큰은 입력 임베딩 계산에만 사용되어 출력 레이어의 계산 비용에 영향 없음 |
| **Over-Tokenized Transformer** (본 논문, 2025) | OE + OD 분리 | 입력만 대폭 확장 | 추가 비용 없이 2.5× 효과적 스케일링 |

최근 관련 연구(Vocabulary Frequency Imbalance 연구, 2025)에서는 더 큰 어휘가 언어 모델 성능을 향상시키는 메커니즘을 설명하였는데, **어휘를 확장하면 토크나이즈된 텍스트의 복잡도가 감소**하여 non-i.i.d. 패턴을 더 쉽게 학습할 수 있고, 자연 데이터의 낮은 고유 복잡성에 더 잘 근접하여 언어 모델링 난이도를 낮춘다는 것을 보였다.

---

## 📚 참고 자료 및 출처

1. **arXiv 원문**: Hongzhi Huang et al., "Over-Tokenized Transformer: Vocabulary is Generally Worth Scaling," arXiv:2501.16975, 2025. https://arxiv.org/abs/2501.16975
2. **HTML 전문**: https://arxiv.org/html/2501.16975v1
3. **ar5iv 렌더링**: https://ar5iv.labs.arxiv.org/html/2501.16975
4. **ICML 2025 포스터**: https://icml.cc/virtual/2025/poster/44467
5. **PMLR 게재**: https://proceedings.mlr.press/v267/huang25bb.html
6. **OpenReview**: https://openreview.net/forum?id=gbeZKej40m
7. **ByteDance Seed 공식 페이지**: https://seed.bytedance.com/en/public_papers/over-tokenized-transformer-vocabulary-is-generally-worth-scaling
8. **Hugging Face Papers**: https://huggingface.co/papers/2501.16975
9. **MarkTechPost 해설**: https://www.marktechpost.com/2025/01/30/decoupling-tokenization-how-over-tokenized-transformers-redefine-vocabulary-scaling-in-language-models/
10. **ResearchGate PDF**: https://www.researchgate.net/publication/388460106
11. **관련 비교 연구**: "Scaling Embedding Layers in Language Models" (arXiv:2502.01637), "Exploiting Vocabulary Frequency Imbalance in Language Model Pre-training" (arXiv:2508.15390)

> ⚠️ **정확도 주의 사항**: 논문 내 세부 수식(특히 OD 손실의 정확한 가중치 정의, product decomposition 수식)은 공개된 HTML/PDF 기반으로 재구성하였으며, 원문 전체에 직접 접근하지 못한 일부 수식은 논문의 맥락과 설명에 기반하여 표준적 형태로 표현하였습니다. 완전한 수식 확인을 위해서는 원문 PDF를 직접 참조하시기 바랍니다.
