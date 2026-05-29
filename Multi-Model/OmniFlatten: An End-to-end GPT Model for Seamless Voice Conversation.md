
# OmniFlatten: An End-to-end GPT Model for Seamless Voice Conversation

> **논문 정보**
> - **저자**: Qinglin Zhang, Luyao Cheng, Chong Deng, Qian Chen, Wen Wang, Siqi Zheng, Jiaqing Liu, Hai Yu, Chaohong Tan, Zhihao Du, Shiliang Zhang (Tongyi Lab, Alibaba)
> - **arXiv**: [2410.17799](https://arxiv.org/abs/2410.17799) (2024년 10월)
> - **게재**: ACL 2025 (Proceedings of the 63rd Annual Meeting of the ACL, Volume 1: Long Papers, pp. 14570–14580)
> - **데모**: https://omniflatten.github.io/

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장

풀 듀플렉스(Full-duplex) 음성 대화 시스템은 전통적인 턴 기반(turn-based) 대화 시스템을 크게 능가하는데, 이는 동시 양방향 통신을 가능하게 하여 인간-인간 상호작용을 밀접하게 모방하기 때문입니다. 그러나 풀 듀플렉스 시스템에서 낮은 레이턴시와 자연스러운 상호작용을 달성하는 것은, 특히 끼어들기(interruption), 백채널(backchannel), 중첩 발화(overlapping speech)와 같은 인간 대화의 역동성을 고려할 때 여전히 중요한 과제입니다.

이 논문은 풀 듀플렉스 대화를 위한 새로운 End-to-End GPT 기반 모델 **OmniFlatten**을 소개하며, 낮은 레이턴시로 자연스러운 대화에 내재된 복잡한 행동을 효과적으로 모델링할 수 있습니다.

### 1.2 주요 기여 (Contributions)

논문의 주요 기여는 다음과 같습니다:
- **복잡한 자연 대화 행동 모델링**: 낮은 레이턴시로 자연스러운 인간과 유사한 대화에 내재된 복잡한 행동을 효과적으로 모델링하는 새로운 E2E GPT 기반 모델 OmniFlatten 제안
- **멀티 스테이지 포스트 트레이닝 스킴**: 텍스트 기반 기반 LLM을 강건한 음성-텍스트 대화 모델로 적응시키는 멀티 스테이지 포스트 트레이닝 방식 제안 (ASR·TTS 기반 지도 멀티태스크 파인튜닝으로 음성-텍스트 모달리티 정렬 수행, 이후 음성·텍스트 스트림의 세밀한 청킹 및 단일 시퀀스로의 평탄화(flattening)를 통해 하프 듀플렉스 및 풀 듀플렉스 대화 능력 점진적 학습)

특히, 이 접근 방식은 백본 텍스트 기반 LLM의 아키텍처를 변경하지 않으며, 계산 집약적인 사전 훈련에 의존하지 않습니다.

---

## 2. 해결 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

전통적인 음성 인터페이스는 음성 인식, 언어 처리, 음성 생성을 위한 별도의 컴포넌트에 의존하지만, OmniFlatten은 이 모든 기능을 단일 E2E 모델로 통합합니다.

구체적으로 해결하고자 하는 문제는 다음 세 가지입니다:

1. **하프 듀플렉스의 비자연성**: 기존 하프 듀플렉스 시스템의 비효율성 해결 — 자연스러운 인간 상호작용에 가까운 동시 통신 지원
2. **고레이턴시**: 파이프라인 방식의 ASR → LLM → TTS 구조에서 발생하는 응답 지연
3. **복잡한 대화 역동성 모델링 부재**: 끼어들기, 백채널, 중첩 발화 등의 인간 대화 역동성

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 오디오 토크나이제이션

모델은 CosyVoice와 같은 기존 모델에서 아이디어를 얻은 새로운 오디오 토크나이저 방식을 채택하여 오디오 입력을 이산 음성 토큰(discrete speech tokens)으로 변환합니다. 이 토크나이저는 벡터 양자화(vector quantization)를 활용하여 오디오 신호를 음향 파형을 나타내는 관리 가능한 토큰 시퀀스로 변환합니다.

오디오 토크나이제이션을 수식으로 표현하면:

$$
\mathbf{s} = \text{Tokenizer}(\mathbf{a}) = \text{VQ}(\text{Encoder}(\mathbf{a}))
$$

여기서 $\mathbf{a}$는 원시 오디오 신호, $\mathbf{s}$는 이산 음성 토큰 시퀀스입니다.

#### 2.2.2 Flattening Operation (평탄화 연산)

훈련 과정은 모달리티 정렬, 하프 듀플렉스 대화 학습, 풀 듀플렉스 대화 학습의 세 단계로 구성됩니다. 모든 훈련 단계에서 플래트닝 연산을 사용하여 데이터를 표준화하며, 이를 통해 서로 다른 모달리티와 작업에 걸쳐 훈련 방법과 GPT 백본을 통합합니다.

플래트닝 연산의 핵심 아이디어는 사용자(User)와 어시스턴트(Assistant)의 음성·텍스트 스트림을 시간 청크(chunk) 단위로 분할하고 단일 시퀀스로 인터리빙(interleaving)하는 것입니다:

$$
\mathcal{F} = \text{Flatten}\left(\{c_t^{U}\}_{t=1}^{T},\ \{c_t^{A}\}_{t=1}^{T}\right) = [c_1^{U}, c_1^{A}, c_2^{U}, c_2^{A}, \ldots, c_T^{U}, c_T^{A}]
$$

여기서:
- $c_t^{U}$: 시각 $t$에서의 사용자 청크 (음성 토큰 + 텍스트 토큰)
- $c_t^{A}$: 시각 $t$에서의 어시스턴트 청크 (음성 토큰 + 텍스트 토큰)
- $T$: 전체 청크 수

#### 2.2.3 멀티 스테이지 포스트 트레이닝 스킴

멀티 스테이지 포스트 트레이닝 방식은 텍스트 LLM 백본을 음성-텍스트 대화 LLM으로 점진적으로 적응시키며, 백본 LLM의 아키텍처를 수정하지 않고 실시간으로 텍스트와 음성을 생성할 수 있게 합니다.

**Stage 1 — Modality Alignment (모달리티 정렬)**:

멀티 스테이지 포스트 트레이닝 프로세스는 ASR 및 TTS 작업을 이용한 텍스트 LLM 백본의 지도 멀티태스크 파인튜닝으로 시작하여 음성-텍스트 모달리티 정렬을 달성하고, 음성과 텍스트 모두를 정확하게 해석하고 생성할 수 있는 멀티모달 LLM을 얻습니다.

$$
\mathcal{L}_{\text{Stage1}} = \mathcal{L}_{\text{ASR}} + \mathcal{L}_{\text{TTS}}
$$

**Stage 2 — Half-Duplex Dialogue Learning**:

음성-텍스트 LLM을 획득한 후, 인터리빙되고 평탄화된 대화를 사용하여 하프 듀플렉스 대화 학습과 풀 듀플렉스 대화 학습의 점진적 단계를 통해 파인튜닝합니다.

$$
\mathcal{L}_{\text{Stage2}} = -\sum_{t} \log P\left(x_t \mid x_{<t};\, \theta\right)
$$

여기서 $x_t$는 인터리빙된 음성·텍스트 시퀀스의 $t$번째 토큰입니다.

**Stage 3 — Full-Duplex Dialogue Learning**:

풀 듀플렉스 시나리오에서는 사용자 발화와 어시스턴트 응답이 동시에 진행됩니다. 모델은 현재 시각 청크에서 조건부 확률을 최대화합니다:

$$
P\left(c_t^{A} \mid c_{ < t}^{A}, c_{\leq t}^{U}\right) = \prod_{k} P\left(a_{t,k} \mid a_{t, < k},\, c_{ < t}^{A},\, c_{\leq t}^{U};\, \theta\right)
$$

여기서 $a_{t,k}$는 시각 $t$의 어시스턴트 청크 내 $k$번째 토큰입니다.

#### 2.2.4 텍스트 토큰 예측을 통한 의미 능력 향상

SyncLM의 단순 중복 제거 전략과 달리, OmniFlatten은 모델의 의미 능력 향상을 위해 **명시적 텍스트 토큰 예측(explicit text token prediction)** 을 탐구합니다.

이로 인해 OmniFlatten은 음성 토큰만 생성하는 SyncLM과 달리 텍스트와 오디오 토큰을 동시에 생성합니다.

### 2.3 모델 구조

OmniFlatten의 핵심에는 GPT와 유사한 대규모 언어 모델이 있으며, 이는 방대한 양의 텍스트 데이터로 훈련되어 언어에 대한 광범위한 이해와 일관성 있는 응답 생성 능력을 제공합니다. 이 모델은 음성 인식과 텍스트-음성 변환을 위한 추가 컴포넌트로 확장되어 음성을 직접 처리하고 생성합니다.

OmniFlatten은 음성 토큰과 텍스트 토큰을 함께 평탄화(flatten)하여 처리합니다.

구조를 정리하면:

| 컴포넌트 | 역할 |
|---|---|
| **텍스트 LLM 백본** | GPT 기반 언어 모델 (아키텍처 수정 없음) |
| **오디오 토크나이저** | 벡터 양자화(VQ) 기반 오디오 → 이산 토큰 변환 |
| **오디오 디토크나이저** | 이산 토큰 → 음성 파형 복원 |
| **플래트닝 모듈** | 다중 스트림 → 단일 시퀀스 변환 |

### 2.4 성능 향상

SyncLM의 중복 제거 전략은 모델링 복잡성을 줄이지만 재구성 시 오류를 야기합니다. 반면 OmniFlatten은 이산화된 오디오 토큰에 추가 연산을 적용하지 않아 오디오 재구성 품질 저하를 방지합니다.

Moshi는 사용자의 음성 입력과 시스템의 텍스트·음성 출력을 병렬로 모델링하여 풀 듀플렉스 대화 처리를 단순화하지만, 이 병렬 프레임워크는 GPT 기반 모델에서 네이티브로 지원되지 않아 음향 지연(acoustic delay)과 내부 독백(inner monologue) 같은 정교한 설계가 필요합니다.

SyncLLM과 OmniFlatten은 시간 청킹 방법을 사용하여 동기화를 위한 시간 정보를 LLM에 임베딩함으로써 풀 듀플렉스 대화를 달성합니다.

### 2.5 한계

논문은 이 접근 방식의 잠재적 도전과 한계에 대해 깊이 다루지 않습니다.

OmniFlatten 접근 방식의 장단점을 완전히 평가하기 위해서는 추가적인 연구와 실제 환경에서의 테스트가 필요합니다.

주요 한계를 정리하면:
- **학습 데이터 의존성**: 풀 듀플렉스 시뮬레이션 대화 데이터 구축의 어려움
- **다국어 일반화**: 영어·중국어 중심 학습 데이터로 인한 타 언어 성능 불확실성
- **실제 환경 노이즈**: 배경 잡음, 반향 등에 대한 강건성 검증 부족
- **청크 단위 처리 지연**: 청킹 방식 자체가 유발하는 최소 레이턴시 존재

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 아키텍처 수정 없는 LLM 적응

OmniFlatten은 텍스트 기반 LLM 백본에 먼저 모달리티 정렬을 수행한 다음, 대화의 다중 음성·텍스트 스트림을 인터리빙하고 단일 시퀀스로 평탄화하여 대화 학습을 수행합니다. 특히 이 접근 방식은 백본 텍스트 기반 LLM의 아키텍처를 변경하지 않으며 계산 집약적인 사전 훈련에도 의존하지 않습니다.

이는 **다양한 사전 훈련 LLM(GPT-4, LLaMA, Qwen 등)에 OmniFlatten의 멀티 스테이지 포스트 트레이닝을 적용**할 수 있음을 의미하며, 더 강력한 백본 LLM을 사용할수록 일반화 성능이 향상될 수 있습니다.

### 3.2 통합된 훈련 방법론의 일반화 기여

모든 훈련 단계에서 플래트닝 연산을 이용한 데이터 표준화를 통해 서로 다른 모달리티와 작업에 걸쳐 훈련 방법과 GPT 백본을 통합합니다. 특히 이 플래트닝 연산을 통한 대화 데이터 표준화는 서로 다른 모달리티와 작업에 걸쳐 훈련 방법과 GPT 백본을 통합하는 것을 가능케 합니다.

이 통합된 훈련 방법론은:
- **멀티모달 확장 가능성**: 미래에 비디오, 이미지 등 다른 모달리티도 동일한 플래트닝 방식으로 처리 가능
- **도메인 전이(Transfer) 용이성**: ASR·TTS 멀티태스크 파인튜닝이 다양한 도메인 음성에 대한 일반화에 기여

### 3.3 명시적 텍스트 토큰 예측의 일반화 효과

SyncLM의 단순 중복 제거 전략과 달리, OmniFlatten은 명시적 텍스트 토큰 예측을 통해 모델의 의미 능력을 향상시킵니다.

이는 다음과 같은 일반화 향상으로 이어집니다:
- 음성 인식 오류에 대한 강건성 (텍스트를 내부 앵커로 활용)
- 다양한 발화 스타일·억양·방언에 대한 의미 수준 일반화
- 텍스트 정보를 통한 음성 생성의 일관성 유지

$$
P(\mathbf{y}_{\text{speech}}, \mathbf{y}_{\text{text}} \mid \mathbf{x}_{\text{speech}}, \mathbf{x}_{\text{text}}) = P(\mathbf{y}_{\text{text}} \mid \mathbf{x}) \cdot P(\mathbf{y}_{\text{speech}} \mid \mathbf{y}_{\text{text}}, \mathbf{x})
$$

텍스트 예측이 음성 생성의 중간 앵커(anchor) 역할을 하여 일반화 성능을 높입니다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

| 모델 | 연도 | 방식 | 풀 듀플렉스 | 특징 |
|---|---|---|---|---|
| **SpeechGPT** | 2023 | E2E S2S | ✗ | 최초 음성-텍스트 통합 LLM, 턴 기반 |
| **dGSLM** | 2023 | E2E | △ | 샴 네트워크 + 크로스 어텐션으로 2채널 대화 |
| **Mini-Omni** | 2024 | E2E S2S | ✗ | 실시간 음성 입출력, 턴 기반 |
| **LLaMA-Omni** | 2024 | E2E S2S | ✗ | LLaMA 기반, 턴 기반 |
| **LSLM** | 2024 | E2E | △ | TTS 통합으로 턴 테이킹 수행 |
| **Moshi** | 2024 | E2E | ✓ | 병렬 다중 스트림 처리, inner monologue |
| **SyncLLM** | 2024 | E2E | ✓ | 시간 청킹, 음성 토큰만 생성 |
| **OmniFlatten** | 2024 | E2E GPT | ✓ | 플래트닝 + 텍스트·음성 동시 생성 |
| **SALMONN-omni** | 2024 | E2E | ✓ | 상태 토큰으로 턴 테이킹 향상 |
| **MinMo** | 2025 | E2E | ✓ | 상태 토큰 기반 풀 듀플렉스 |

SpeechGPT, LauraGPT, Mini-Omni, LLaMA-Omni, GLM-4-Voice는 음성과 텍스트 입력을 이해하고 두 가지 모두로 출력을 생성할 수 있지만, 주로 턴 기반 대화 모델이며 풀 듀플렉스 대화를 지원하지 않습니다.

LSLM은 TTS 모델을 통합하여 턴 테이킹 작업의 E2E 모델링을 수행함으로써 풀 듀플렉스 시나리오를 탐색하며, 말하는 동안 지속적으로 듣고 어느 순간이든 멈출 수 있게 합니다.

오디오 데이터에 직접 작동하는 E2E 방식은 초언어적(paralinguistic) 및 비언어적 신호와 백채널링 같은 대화 행동을 포함하여 더 광범위한 음성 특징을 포착할 잠재력이 있습니다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5.1 앞으로의 연구에 미치는 영향

1. **아키텍처 패러다임 전환 촉진**
OmniFlatten의 접근 방식은 효율적이고 자연스러운 E2E 풀 듀플렉스 음성 대화 시스템 개발을 위한 유망한 연구 방향을 제시합니다. 이는 향후 연구가 파이프라인 기반 시스템보다 E2E 방식으로 이동하는 것을 가속화할 것입니다.

2. **포스트 트레이닝 패러다임의 재정의**
백본 LLM의 아키텍처를 수정하지 않고 텍스트 LLM을 음성-텍스트 대화 LLM으로 점진적으로 적응시키는 멀티 스테이지 포스트 트레이닝 스킴은 기존 LLM의 음성 인터페이스 확장에 대한 새로운 기준(baseline)을 제시합니다.

3. **멀티모달 대화 시스템 연구 자극**
OmniFlatten은 통합 AI 아키텍처로 구동되는 보다 자연스럽고 대화적인 음성 인터페이스에 대한 설득력 있는 비전을 제시하며, 음성, 언어, 음성 생성 기능을 단일 E2E 모델로 결합합니다. 이는 비디오, 이미지 등을 포함한 멀티모달 풀 듀플렉스 연구로의 확장 경로를 열어줍니다.

4. **벤치마크 표준화에 기여**
Moshi는 실시간 사용자 음성을 처리하고 중첩과 끼어들기를 지원하며, SyncLLM은 2채널 음성을 위한 시간 동기 오디오 청크 모델링을 도입하고, OmniFlatten은 음성과 텍스트 토큰을 함께 처리하며, SALMONN-omni와 MinMo는 턴 테이킹 모델링 향상을 위한 상태 토큰을 주입합니다. 이처럼 다양한 방식의 E2E 모델들이 나오면서 풀 듀플렉스 대화 시스템 평가를 위한 공정한 벤치마크 수립 연구가 활성화되고 있습니다.

### 5.2 앞으로 연구 시 고려할 점

#### (1) 데이터 품질 및 다양성

풀 듀플렉스 대화 시뮬레이션 데이터 구축의 어려움이 있습니다. 실제 인간-인간 대화 데이터와 시뮬레이션 데이터 간의 도메인 갭(domain gap)을 줄이는 연구가 필요합니다.

$$
\mathcal{D}_{\text{full-duplex}} = \{(\mathbf{x}^U_{1:T}, \mathbf{x}^A_{1:T}) \mid \exists t: \mathbf{x}^U_t \neq \emptyset \land \mathbf{x}^A_t \neq \emptyset\}
$$

즉, 사용자와 어시스턴트가 동시에 발화하는 구간이 포함된 데이터셋 확보가 핵심입니다.

#### (2) 평가 메트릭 표준화

풀 듀플렉스 기능의 공정한 평가를 위해 끼어들기 처리에 대한 철저한 평가를 포함하는 벤치마크 파이프라인이 필요하며, 이 파이프라인은 끼어들기 처리와 응답 품질의 두 가지 핵심 차원에서 풀 듀플렉스 성능을 정의해야 합니다.

#### (3) 다국어 및 교차 문화 일반화

현재 모델은 주로 중국어·영어 데이터로 훈련되었습니다. 향후에는:
- 저자원(low-resource) 언어에 대한 적용 가능성 검토
- 억양, 방언, 코드스위칭(code-switching) 대화 처리 능력 강화

#### (4) 실제 환경 강건성

실험 결과는 끼어들기에 대한 강건성, 레이턴시 관리, 자연스러운 응답 지연 등의 주요 측면을 부각시키며, 이러한 시스템들의 현재 한계에 대한 귀중한 통찰을 제공합니다.

실제 배포 환경에서의 배경 잡음, 반향 제거(echo cancellation), 다중 화자 처리 등이 중요한 연구 과제입니다.

#### (5) 윤리적·안전성 고려

E2E 방식은 초언어적 및 비언어적 신호와 백채널링 같은 대화 행동을 포착할 잠재력이 있습니다. 이런 기능이 강해질수록 딥페이크 음성, 개인정보 침해 등의 윤리적 이슈에 대한 사전 연구가 병행되어야 합니다.

#### (6) 경량화 및 엣지 배포

현재 OmniFlatten은 대규모 LLM 백본을 기반으로 하므로, 실제 모바일·엣지 디바이스 배포를 위한 모델 경량화(knowledge distillation, quantization) 연구가 필요합니다.

---

## 📚 참고 자료 (출처)

| # | 제목 / 링크 |
|---|---|
| 1 | **[arXiv:2410.17799]** Qinglin Zhang et al., "OmniFlatten: An End-to-end GPT Model for Seamless Voice Conversation", 2024. https://arxiv.org/abs/2410.17799 |
| 2 | **[ACL 2025]** OmniFlatten, ACL Anthology, Proceedings of ACL 2025 (Volume 1: Long Papers), pp.14570–14580. https://aclanthology.org/2025.acl-long.709/ |
| 3 | **[arXiv PDF]** OmniFlatten Full Paper PDF: https://arxiv.org/pdf/2410.17799 |
| 4 | **[arXiv HTML v1]** OmniFlatten HTML Version: https://arxiv.org/html/2410.17799v1 |
| 5 | **[ResearchGate]** OmniFlatten 논문 페이지: https://www.researchgate.net/publication/385177063 |
| 6 | **[Semantic Scholar]** OmniFlatten: https://www.semanticscholar.org/paper/ff0e3b8902121832e8b9518745c394b2c1d1efad |
| 7 | **[HuggingFace Papers]** https://huggingface.co/papers/2410.17799 |
| 8 | **[Moonlight Literature Review]** https://www.themoonlight.io/en/review/omniflatten-an-end-to-end-gpt-model-for-seamless-voice-conversation |
| 9 | **[Full-Duplex-Bench arXiv:2503.04721]** "A Benchmark to Evaluate Full-Duplex Spoken Dialogue Models on Turn-taking Capabilities", 2025. https://arxiv.org/html/2503.04721v2 |
| 10 | **[ISCA Interspeech 2025]** Hu et al., "Efficient and Direct Duplex Modeling for Speech-to-Speech Language Model": https://www.isca-archive.org/interspeech_2025/hu25f_interspeech.pdf |
| 11 | **[FD-Bench arXiv:2507.19040]** "A Full-Duplex Benchmarking Pipeline for Full Duplex Spoken Dialogue Systems", 2025. https://arxiv.org/html/2507.19040v1 |
| 12 | **[데모 사이트]** OmniFlatten Audio Samples: https://omniflatten.github.io/ |

# OmniFlatten: An End-to-end GPT Model for Seamless Voice Conversation

### **1. 핵심 주장 및 주요 기여 요약 (Executive Summary)**

**OmniFlatten**은 기존의 턴제(Turn-based) 대화 시스템의 한계를 넘어, 인간과 유사한 실시간 **전이중(Full-duplex)** 음성 대화를 구현하기 위해 제안된 **End-to-End GPT 기반 모델**입니다.

*   **핵심 주장:** 복잡한 아키텍처 수정 없이, 음성과 텍스트 토큰을 하나의 시퀀스로 **'평탄화(Flattening)'** 하여 학습함으로써, LLM(Large Language Model)이 본래 가진 능력을 유지하면서도 낮은 지연 시간(Low Latency)으로 끼어들기(Interruption)와 동시 발화(Overlapping speech)를 처리할 수 있다는 것입니다.
*   **주요 기여:**
    1.  **Flattening 기법 도입:** 텍스트와 음성 스트림을 미세한 청크(Chunk) 단위로 쪼개어 하나의 선형 시퀀스로 변환, 표준 GPT 아키텍처에서 텍스트와 음성을 통합 처리하는 단순하고 효율적인 방법론 제시.
    2.  **다단계 사후 학습(Multi-stage Post-training):** 모달리티 정렬(Alignment) → 반이중(Half-duplex) 학습 → 전이중(Full-duplex) 학습으로 이어지는 단계적 학습 파이프라인 구축.
    3.  **데이터 합성 파이프라인:** 고비용의 전이중 대화 데이터를 수집하는 대신, 오픈소스 텍스트 데이터를 활용해 정교한 전이중 음성 대화 데이터를 생성하는 시뮬레이션 방법론 제안.

***

### **2. 상세 분석: 문제 정의, 제안 방법, 모델 구조 및 성능**

#### **2.1 해결하고자 하는 문제 (Problem Statement)**
기존의 음성 대화 시스템은 주로 **반이중(Half-duplex)** 방식으로, 사용자가 말을 끝낼 때까지 기다려야만 시스템이 응답할 수 있었습니다. 이는 인간의 자연스러운 대화 특징인 **끼어들기(Barge-in)**, **맞장구(Backchannel)**, **동시 발화**를 구현하지 못하며, 응답 지연 시간이 길어지는 한계가 있습니다. 반면, 기존의 전이중 시스템(예: Moshi)은 복잡한 병렬 스트림 처리를 위해 아키텍처를 수정해야 하거나 구현 난이도가 높았습니다.

#### **2.2 제안하는 방법 및 수식 (Proposed Method)**

OmniFlatten은 **'Flattening(평탄화)'** 연산을 통해 다중 모달리티(텍스트, 음성)와 다중 화자(사용자, 시스템)의 데이터를 단일 시퀀스로 변환합니다.

**[수식적 표현]**
시간 $t$에서의 대화 상태를 나타내기 위해, 사용자의 음성 토큰 $S^u$, 시스템의 텍스트 토큰 $T^a$, 시스템의 음성 토큰 $S^a$를 정의합니다. 기존 방식이 이들을 별도의 채널로 처리했다면, OmniFlatten은 이를 시간 순서에 따라 인터리빙(interleaving)하여 하나의 시퀀스 $X$로 구성합니다.

특정 시간 청크 $k$에서의 입력 시퀀스 $X_k$는 다음과 같이 표현될 수 있습니다:

$$
X_k = [S^u_{k,1}, ..., S^u_{k,N}, T^a_{k,1}, ..., T^a_{k,M}, S^a_{k,1}, ..., S^a_{k,N}]
$$

여기서 $S^u_{k}$는 $k$번째 청크의 사용자 음성 입력, $T^a_{k}$와 $S^a_{k}$는 해당 입력에 대해 모델이 예측해야 할 시스템의 텍스트 생각(Thought)과 음성 반응입니다.
모델은 자기회귀(Autoregressive) 방식으로 다음 토큰을 예측합니다:

$$
P(X) = \prod_{i=1}^{L} P(x_i | x_{ < i})
$$

이때 손실 함수 $\mathcal{L}$은 텍스트와 음성 토큰 예측에 대한 Cross-Entropy Loss로 정의됩니다.

**학습 파이프라인 (3단계):**
1.  **Modality Alignment:** ASR(음성인식) 및 TTS(음성합성) 태스크를 통해 텍스트 LLM이 음성 토큰을 이해하고 생성하도록 튜닝.
2.  **Half-duplex Learning:** 턴제 대화 데이터로 기본적인 대화 흐름 학습.
3.  **Full-duplex Learning:**
    *   **3-Stream Training:** User 음성 $\rightarrow$ System 텍스트 $\rightarrow$ System 음성 순으로 학습 (System의 텍스트적 사고 과정 포함).
    *   **2-Stream Training:** 지연 시간을 극도로 줄이기 위해 중간 Text 생성을 생략하고 음성-to-음성(User 음성 $\rightarrow$ System 음성)으로 직접 매핑하는 단계 추가.

#### **2.3 모델 구조 (Model Architecture)**
*   **Backbone:** **Qwen2-0.5B** (경량화된 LLM 사용). 구조적 수정 없이 표준 Transformer Decoder 아키텍처를 그대로 사용합니다.
*   **Tokenizer:** **CosyVoice**의 음성 토크나이저와 Flow Matching 기반 De-tokenizer를 사용하여 고품질 음성 생성 및 의미적(Semantic) 정보 보존을 동시에 달성했습니다.

#### **2.4 성능 향상 및 한계 (Performance & Limitations)**
*   **성능 향상:**
    *   **지연 시간(Latency):** Moshi 등 경쟁 모델 대비 빠른 응답 속도(Assistant Turn-taking)와 사용자 끼어들기 반응 속도를 기록했습니다.
    *   **대화 품질:** 텍스트 LLM의 지식을 효과적으로 전이하여, 단순 음성 모델보다 문맥 이해도가 높습니다.
*   **한계:**
    *   **모델 크기:** 0.5B의 작은 모델을 사용하여 복잡한 지식 추론에는 한계가 있을 수 있습니다.
    *   **User Turn-taking:** 사용자가 끼어들 때 발화를 멈추는 기능은 향상되었으나, 여전히 100% 완벽하지 않으며(약 50~70% 정확도), 응답 속도와 정확도 간의 트레이드오프가 존재합니다.
    *   **Backchannel 부재:** "음", "아하" 같은 자연스러운 맞장구 반응 생성은 아직 미흡합니다.

***

### **3. 모델의 일반화 성능 향상 가능성 (Generalization Capabilities)**

이 논문에서 가장 주목해야 할 점은 **'단순화(Simplification)'를 통한 일반화 가능성**입니다.

1.  **Backbone Agnostic (백본 독립성):** Flattening 기법은 특정 LLM 구조에 의존하지 않습니다. 즉, Qwen뿐만 아니라 Llama 3, GPT-4 등 더 크고 강력한 모델에도 동일한 방법론을 즉시 적용할 수 있습니다. 이는 모델의 일반화 성능(Generalization Performance)을 비약적으로 확장할 수 있는 잠재력을 가집니다.
2.  **Modality Unification (모달리티 통합):** 데이터를 단일 시퀀스로 평탄화함으로써, 텍스트, 음성뿐만 아니라 향후 **이미지, 비디오** 등 다른 모달리티가 추가되더라도 동일한 학습 프레임워크를 유지할 수 있습니다. ("Omni" 모델로의 확장성)
3.  **Data Synthesis Pipeline:** 논문에서 제안한 데이터 합성 파이프라인은 특정 도메인에 국한되지 않고 다양한 시나리오(감정 표현, 다화자 대화 등)로 데이터를 무한히 확장할 수 있게 해 주어, 데이터 부족으로 인한 일반화 저하 문제를 해결할 핵심 열쇠가 됩니다.

***

### **4. 향후 연구에 미치는 영향 및 고려할 점 (Impact & Future Directions)**

최신 연구(2024년 말~2025년 초) 흐름을 바탕으로 본 논문의 영향과 향후 연구 방향을 제안합니다.

#### **영향 (Impact)**
*   **Omni-model의 표준화:** 복잡한 병렬 아키텍처 대신, **Flattening** 방식이 전이중 음성 모델(Duplex Model)의 표준 베이스라인(Baseline)으로 자리 잡을 가능성이 큽니다. 최근 발표된 **FlexDuo (2025)**, **SOVA-Bench** 등의 연구에서도 OmniFlatten은 시간 청킹(Time Chunking) 기반 모델의 대표적인 사례로 인용되며 비교되고 있습니다.
*   **실시간 대화의 민주화:** 거대한 컴퓨팅 자원 없이 0.5B 수준의 경량 모델로도 전이중 대화가 가능함을 입증하여, 온디바이스(On-device) AI 에이전트 연구를 가속화할 것입니다.

#### **향후 연구 시 고려할 점 (Based on Recent Research)**
1.  **제어 모듈의 분리 (Decoupling Control):** 최신 연구인 **FlexDuo** (Liao et al., 2025)는 OmniFlatten과 같은 통합 모델이 '상태 제어(언제 듣고 언제 말할지)'와 '대화 생성'을 동시에 수행할 때 발생하는 혼선을 지적합니다. 향후 연구에서는 OmniFlatten의 구조에 **명시적인 턴 제어(Turn-taking Control) 토큰**이나 별도의 경량 제어 모듈을 결합하여 안정성을 높이는 방향이 고려되어야 합니다.
2.  **Non-verbal Communication 강화:** 단순히 말하고 듣는 것을 넘어, **비언어적 신호(웃음, 숨소리, 억양 변화)** 를 명시적으로 모델링하는 연구가 필요합니다. 이는 모델이 인간의 감정 상태를 더 잘 파악하고 공감하는 데 필수적입니다.
3.  **Streaming 구조 최적화:** 현재의 Chunk 기반 처리는 미세한 지연을 발생시킵니다. 이를 **토큰 단위 스트리밍(Token-level Streaming)** 으로 더욱 세분화하거나, 문맥 정보를 잃지 않으면서 청크 사이즈를 동적으로 조절하는 적응형(Adaptive) 알고리즘 연구가 유망합니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9689dee5-59f7-4a5b-b1b2-6c69e0e86a85/2410.17799v2.pdf)
[2](https://link.springer.com/10.1007/s12351-023-00790-w)
[3](https://link.springer.com/10.1007/s41324-023-00551-z)
[4](https://ijebmr.com/uploads/pdf/archivepdf/2024/IJEBMR_1391.pdf)
[5](https://jrenewables.springeropen.com/articles/10.1186/s40807-024-00119-x)
[6](http://www.emerald.com/ijebr/article/29/4/816-837/117413)
[7](https://link.springer.com/10.1007/978-3-319-97385-2_14)
[8](https://link.springer.com/10.1007/s10479-020-03912-1)
[9](https://journals.sagepub.com/doi/10.1177/15330338221132927)
[10](https://www.mdpi.com/0718-1876/18/4/109)
[11](https://journals.sagepub.com/doi/10.1177/13621688241229534)
[12](http://arxiv.org/pdf/2410.17799.pdf)
[13](http://arxiv.org/pdf/2105.04448.pdf)
[14](http://link.aps.org/pdf/10.1103/PhysRevD.104.076027)
[15](http://arxiv.org/pdf/1911.09107.pdf)
[16](https://arxiv.org/pdf/2105.09923.pdf)
[17](https://arxiv.org/pdf/2402.16014.pdf)
[18](https://arxiv.org/html/2504.06857v1)
[19](https://pmc.ncbi.nlm.nih.gov/articles/PMC9489871/)
[20](https://aclanthology.org/2025.acl-long.709.pdf)
[21](https://arxiv.org/html/2502.13472v1)
[22](https://www.isca-archive.org/interspeech_2025/hou25b_interspeech.pdf)
[23](https://chatpaper.com/paper/70632)
[24](https://www.emergentmind.com/topics/full-duplex-spoken-dialogue-model)
[25](https://chatpaper.com/paper/176105)
[26](https://arxiv.org/html/2410.17799v2)
[27](https://www.isca-archive.org/interspeech_2025/hu25f_interspeech.pdf)
[28](https://www.isca-archive.org/interspeech_2025/peng25b_interspeech.pdf)
[29](https://arxiv.org/html/2410.17799v1)
[30](https://www.themoonlight.io/en/review/omniflatten-an-end-to-end-gpt-model-for-seamless-voice-conversation)
[31](https://powerdrill.ai/discover/discover-OmniFlatten-An-End-to-end-cm2nsfxirwip801dfnjb23vf1)
