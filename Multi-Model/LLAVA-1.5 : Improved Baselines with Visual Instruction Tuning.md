# Improved Baselines with Visual Instruction Tuning

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

LLaVA 프레임워크에서 **완전 연결(fully-connected) 비전-언어 커넥터**가 놀라울 정도로 강력하고 데이터 효율적임을 실증적으로 밝힌다. 복잡한 visual resampler 없이도 두 가지 단순한 수정만으로 11개 벤치마크에서 SOTA를 달성할 수 있다.

### 주요 기여 (5가지)

| 기여 | 설명 |
|------|------|
| ① MLP 커넥터 | 선형 projection → 2층 MLP로 교체 |
| ② 응답 형식 프롬프트 | VQA 단답 vs. 장문 응답 균형 해결 |
| ③ 학술 VQA 데이터 통합 | VQAv2, GQA, OKVQA 등 추가 |
| ④ 고해상도 확장(LLaVA-1.5-HD) | 이미지 그리드 분할 인코딩 |
| ⑤ 개방 문제 탐색 | 할루시네이션, 조합 능력, 데이터 효율성 분석 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

논문이 지적하는 핵심 문제는 세 가지다:

**문제 ①: LMM 설계 선택의 불명확성**
- LLaVA는 conversational VQA에서 강하지만 단답형 학술 벤치마크에서 약함
- InstructBLIP은 반대로 단답형에서 강하지만 자연 대화에서 약함
- 두 모델 간 차이의 근본 원인이 불명확

**문제 ②: 멀티태스크 균형 실패**
- InstructBLIP의 경우, "Is this unusual? Please explain in detail." 같은 질문에 단순히 "yes"로만 응답하는 과적합 문제 발생
- 원인: 모호한 응답 형식 프롬프트 + LLM 미세조정 부재

**문제 ③: 고해상도 처리의 어려움**
- 기존 CLIP 비전 인코더는 최대 $336^2$ 해상도로 제한됨
- positional embedding interpolation은 대규모 재학습 필요

---

### 2.2 제안하는 방법 (수식 포함)

#### (A) MLP 비전-언어 커넥터

기존 LLaVA의 선형 projection:

$$f_{\text{linear}}(\mathbf{Z}_v) = \mathbf{W} \cdot \mathbf{Z}_v$$

LLaVA-1.5에서 제안하는 2-layer MLP:

$$f_{\text{MLP}}(\mathbf{Z}_v) = \mathbf{W}_2 \cdot \sigma(\mathbf{W}_1 \cdot \mathbf{Z}_v + \mathbf{b}_1) + \mathbf{b}_2$$

여기서:
- $\mathbf{Z}_v$: CLIP-ViT-L-336px에서 추출된 시각적 특징
- $\mathbf{W}_1, \mathbf{W}_2$: 학습 가능한 가중치 행렬
- $\sigma$: 활성화 함수 (GELU)
- $\mathbf{b}_1, \mathbf{b}_2$: 편향 벡터

#### (B) 학습 목적 함수

LLaVA 프레임워크의 자기회귀 언어 모델링 손실:

$$\mathcal{L} = -\sum_{t=1}^{T} \log P_\theta(x_t \mid \mathbf{X}_v, x_1, x_2, \ldots, x_{t-1})$$

여기서:
- $\mathbf{X}_v$: MLP로 투영된 시각 토큰
- $x_t$: $t$번째 언어 토큰
- $\theta$: 모델 파라미터 (LLM 전체 + MLP 커넥터)

#### (C) 응답 형식 프롬프트

단답형 VQA 질문에 다음 프롬프트를 추가:

> *"Answer the question using a single word or phrase."*

다중선택형에는:

> *"Answer with the option's letter from the given choices directly."*

이를 통해 동일 LLM이 장/단문 응답을 모두 정확히 생성 가능.

#### (D) LLaVA-1.5-HD 고해상도 확장

이미지 $I$를 $N$개의 그리드 패치로 분할:

$$I \rightarrow \{P_1, P_2, \ldots, P_N\}, \quad P_i \in \mathbb{R}^{224 \times 224 \times 3}$$

각 패치를 독립적으로 인코딩 후 병합:

$$\mathbf{F}_{\text{high}} = \text{Merge}(\{\text{ViT}(P_i)\}_{i=1}^{N})$$

전역 컨텍스트 추가:

$$\mathbf{F}_{\text{final}} = \text{Concat}(\mathbf{F}_{\text{high}},\; \text{ViT}(\text{Resize}(I, 224)))$$

최종 LLM 입력:

$$\mathbf{X}_{\text{input}} = [f_{\text{MLP}}(\mathbf{F}_{\text{final}});\; \mathbf{X}_{\text{text}}]$$

---

### 2.3 모델 구조

```
입력 이미지
    │
    ▼
[CLIP-ViT-L-336px]  ← 비전 인코더 (frozen)
    │
    ▼
[2-layer MLP Connector]  ← 학습 가능 (비전-언어 정렬)
    │
    ▼
[Vicuna v1.5 (7B or 13B)]  ← LLM (전체 파인튜닝)
    │
    ▼
텍스트 응답
```

**2단계 학습 파이프라인:**

| 단계 | 목적 | 학습 대상 | 데이터 | LR |
|------|------|-----------|--------|-----|
| Stage 1: Pretraining | 시각-언어 정렬 | MLP만 학습 | 558K 이미지-텍스트 쌍 | 1e-3 |
| Stage 2: Instruction Tuning | 지시 따르기 학습 | MLP + LLM 전체 | 665K 멀티태스크 데이터 | 2e-5 |

**데이터 구성 (Table 7 기반):**

| 데이터셋 | 크기 | 역할 |
|----------|------|------|
| LLaVA-Instruct | 158K | 시각 대화 |
| ShareGPT | 40K | 언어 대화 (다국어) |
| VQAv2 | 83K | 시각 이해 |
| GQA | 72K | 구성적 VQA |
| OKVQA | 9K | 외부 지식 VQA |
| OCRVQA | 80K | OCR |
| A-OKVQA | 66K | 다중선택 VQA |
| TextCaps | 22K | 텍스트 캡셔닝 |
| RefCOCO | 48K | 영역 수준 이해 |
| Visual Genome | 86K | 세밀한 지각 |
| **합계** | **665K** | |

---

### 2.4 성능 향상

**점진적 스케일링 결과 (Table 2):**

| 구성 | GQA | MME | MM-Vet |
|------|-----|-----|--------|
| 기본 LLaVA (7B) | — | 809.6 | 25.5 |
| +VQAv2 | 47.0 | 1197.0 | 27.7 |
| +Format Prompt | 46.8 | 1323.8 | 26.3 |
| +MLP Connector | 47.3 | 1355.2 | 27.8 |
| +OKVQA/OCR | 50.0 | 1377.6 | 29.6 |
| +Region VQA | 50.3 | 1426.5 | 30.8 |
| +336px 해상도 | 51.4 | 1450.0 | 30.3 |
| **LLaVA-1.5 13B** | **63.3** | **1531.3** | **36.1** |

**타 모델 대비 (Table 3, 4):**
- InstructBLIP-13B 대비: MME +318.5점 (1212.8 → 1531.3)
- Qwen-VL-Chat 대비: MMBench-CN +7.3% (56.7% → 63.6%)
- 훈련 데이터: LLaVA-1.5는 1.2M vs InstructBLIP 130M (약 **100배 효율적**)

---

### 2.5 한계

1. **고해상도 학습 시간**: $336^2$ → $448^2$ 전환 시 학습 시간 약 2배 증가
2. **다중 이미지 처리 불가**: 단일 이미지만 처리 가능, 멀티이미지 instruction data 부재
3. **특정 도메인 추론 한계**: 의료·수학 등 전문 분야 문제 해결 능력 제한
4. **할루시네이션 잔존**: 감소되었지만 완전 제거 불가, 의료 등 critical application에서 주의 필요
5. **일부 언어 성능 저하**: 한국어 등 일부 언어에서 오류 발생

---

## 3. 모델의 일반화 성능 향상 가능성 (중점 분석)

### 3.1 응답 형식 일반화 (Format Instruction Generalization)

LLaVA-1.5는 훈련 시 제한된 형식 지시만 학습했음에도, **미학습 형식 지시에 일반화**:

- VizWiz 벤치마크: "Unanswerable" 응답 정확도 $11.1\% \rightarrow 67.8\%$ (6배 향상)
- JSON 구조화 출력, Stable Diffusion 프롬프트 생성 등 미학습 형식에도 적용 가능

이는 LLM의 instruction-following 능력이 시각 도메인으로 **전이(transfer)** 됨을 시사.

### 3.2 다국어 멀티모달 일반화

**핵심 발견**: 영어 시각 지시만으로 학습했음에도 다국어 시각 대화 가능.

메커니즘:
$$\text{시각 지시 (영어)} \oplus \text{ShareGPT (다국어)} \rightarrow \text{다국어 시각 대화}$$

이는 **조합적 일반화(compositional generalization)** 의 대표 사례:
- ShareGPT의 다국어 텍스트만 학습 → 시각 대화에서 다국어 응답 행동 학습
- MMBench-CN에서 Qwen-VL-Chat(중국어 VQA로 파인튜닝) 대비 +7.3% 달성

### 3.3 조합적 능력 (Compositional Capabilities)

> 개별 태스크를 독립적으로 학습했지만, 해당 능력의 조합이 필요한 새로운 태스크에 일반화

구체적 사례:
- 장문 언어 추론(ShareGPT) + 단문 시각 추론(VQA) → 시각 글쓰기 능력 향상
- OCR 학습 → TextVQA 일반화
- Region-level VQA → 세밀한 시각 묘사 향상

### 3.4 데이터 효율성과 일반화

**핵심 실험**: 훈련 데이터를 무작위로 서브샘플링했을 때의 성능 변화

$$\text{Perf}(50\% \text{ data}) \geq 98\% \times \text{Perf}(100\% \text{ data})$$

- 50% 데이터에서 MMBench, ScienceQA, POPE 성능이 **오히려 소폭 향상**
- LIMA ("Less Is More for Alignment") 원리의 멀티모달 적용 가능성 시사

이는 **데이터 품질이 양보다 중요**하며, 더 정교한 데이터 압축 전략으로 일반화 성능을 유지하면서 효율을 높일 수 있음을 의미.

### 3.5 할루시네이션 감소와 일반화

**발견**: 할루시네이션은 데이터 오류뿐 아니라 **모델의 처리 해상도 한계**에서 발생:

$$\text{할루시네이션} \propto \frac{\text{데이터 세부 정보 요구 수준}}{\text{모델 처리 해상도 수준}}$$

$448^2$ 해상도로 확장 시 할루시네이션 유의미하게 감소 → 고해상도가 일반화 성능 향상의 핵심 요소.

### 3.6 제로샷 일반화

VizWiz 데이터셋은 시각 장애인이 촬영한 이미지로, 배포 외(out-of-distribution) 데이터:
- LLaVA-1.5-7B: 50.0% (InstructBLIP-7B 34.5% 대비 대폭 향상)
- 이는 MLP 커넥터 + 다양한 VQA 데이터가 OOD 제로샷 일반화에 기여함을 의미

---

## 4. 최신 관련 연구 비교 분석 (2020년 이후)

### 4.1 비전-언어 커넥터 설계 비교

| 모델 | 커넥터 | 사전학습 데이터 | 특징 |
|------|--------|----------------|------|
| BLIP-2 (2023) | Q-Former | 129M | 32개 쿼리 토큰으로 압축 |
| InstructBLIP (2023) | Instruction-aware Q-Former | 129M | 지시 기반 특징 추출 |
| Qwen-VL (2023) | Cross-attention Resampler | 1.4B | 256 쿼리 토큰 |
| Flamingo (2022) | Perceiver Resampler | 수십억 | 64개 시각 토큰 |
| **LLaVA-1.5 (2023)** | **2-layer MLP** | **558K** | **단순하지만 강력** |

**분석**: LLaVA-1.5는 가장 단순한 구조로 가장 높은 성능을 달성함으로써, 복잡한 resampler의 필요성에 근본적인 의문을 제기.

### 4.2 이후 연구 발전 트렌드

**LLaVA-1.5 이후 직접적 영향을 받은 연구:**

| 연구 | 기여 | LLaVA-1.5와의 관계 |
|------|------|-------------------|
| LLaVA-NeXT (2024) | dynamic resolution, 더 많은 그리드 분할 | LLaVA-1.5-HD 아이디어 확장 |
| InternVL (2024) | 대규모 시각 인코더 + MLP 커넥터 | MLP 커넥터 아이디어 계승 |
| Cambrian-1 (2024) | spatial vision aggregator 도입 | 커넥터 설계 재탐색 |
| LLaMA-3-LLaVA (2024) | 더 강력한 LLM 기반 | base LLM 중요성 강조 |

> **주의**: 위 표의 LLaVA-NeXT 등 2024년 이후 연구들은 본 논문(2310.03744v2) 내에 직접 인용되지 않으며, 일반적으로 알려진 후속 연구 맥락입니다. 구체적 수치 비교는 해당 논문을 직접 확인하시기 바랍니다.

### 4.3 데이터 전략 비교

$$\underbrace{\text{Flamingo}}_{\text{수십억 쌍}} > \underbrace{\text{InstructBLIP}}_{\text{129M}} > \underbrace{\text{Qwen-VL}}_{\text{1.4B}} \gg \underbrace{\text{LLaVA-1.5}}_{\text{1.2M 공개 데이터}}$$

LLaVA-1.5는 **100~1000배 적은 데이터**로 경쟁하거나 능가하는 성능을 보임.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려 사항

### 5.1 앞으로의 연구에 미치는 영향

**① 아키텍처 설계 패러다임의 전환**

복잡한 visual resampler → 단순 MLP의 우위 입증으로, **"단순성의 가치"** 재조명. 이는 이후 연구에서 architecture complexity보다 데이터 품질과 학습 레시피를 우선 탐색하는 방향으로의 전환을 유도함.

**② 재현 가능한 공개 베이스라인 제공**

모든 데이터가 공개되고 단일 8×A100 노드에서 1일 학습 가능한 베이스라인 제공 → 오픈소스 LMM 연구의 민주화에 기여. 이후 수많은 연구가 LLaVA-1.5를 베이스라인으로 활용.

**③ 조합적 일반화 연구 촉진**

명시적 멀티태스크 학습 없이도 조합적 능력이 창발(emerge)됨을 보임으로써, instruction tuning의 창발적 특성에 대한 연구 자극.

**④ 해상도-할루시네이션 관계 발견**

해상도 향상이 할루시네이션 감소에 기여한다는 발견은, 이후 고해상도 멀티모달 연구(LLaVA-NeXT, InternVL 등)의 이론적 근거 제공.

**⑤ 데이터 효율성 연구 방향 제시**

50% 데이터로 98% 성능 유지 → **LIMA 원리의 멀티모달 적용** 가능성 시사. 이는 데이터 선택, 커리큘럼 학습, 데이터 압축 연구의 새로운 방향 제공.

### 5.2 앞으로 연구 시 고려할 점

**① 데이터 다양성 vs. 양의 트레이드오프**

$$\text{성능} = f(\text{데이터 품질} \times \text{다양성}, \text{데이터 양})$$

단순 양적 확대보다 태스크 다양성과 품질을 고려한 데이터 큐레이션 전략 필요. 특히 domain-specific hallucination 방지를 위한 세밀한 데이터 균형이 중요.

**② 시각 인코더의 한계 극복**

- CLIP-ViT의 $336^2$ 해상도 제한을 넘어선 native high-resolution 인코더 연구 필요
- EVA-CLIP, SigLIP 등 더 강력한 시각 인코더와의 결합 효과 탐구

**③ 멀티이미지 및 비디오 처리**

LLaVA-1.5의 명시적 한계인 단일 이미지 처리 제약을 극복하기 위한:
- 효율적인 multi-image context management
- 시간적 정보를 포함하는 video instruction tuning

**④ 할루시네이션의 근본적 해결**

단순 해상도 향상만으로는 불충분하며:
- RLHF/DPO 기반 hallucination-aware 학습
- 시각 근거(visual grounding) 강화 훈련
- 팩트 체킹 메커니즘 통합

**⑤ 전문 도메인 적응**

일반 목적 모델의 한계를 인식하고, 의료·과학·법률 등 전문 도메인에서의:
- 도메인 특화 시각 instruction 데이터 구축
- 안전성과 신뢰성 평가 프레임워크 개발

**⑥ LLM 선택 전략의 정교화**

논문에서 밝혔듯이 base LLM 품질이 최종 성능에 결정적:

$$\text{Vicuna-v1.5} > \text{Vicuna-v1.3} \approx \text{LLaMA-2-Chat}\ (\text{다국어 일반화에서})$$

SFT vs. RLHF 학습 전략이 멀티모달 성능에 미치는 차별적 영향에 대한 체계적 연구 필요.

**⑦ 경량화 및 추론 효율성**

Full image patches 사용으로 인한 긴 시퀀스 → 효율적 attention 메커니즘, 시각 토큰 압축, quantization과의 결합 연구 필요.

---

## 참고 자료

- **원본 논문**: Haotian Liu, Chunyuan Li, Yuheng Li, Yong Jae Lee, "Improved Baselines with Visual Instruction Tuning," arXiv:2310.03744v2, 2024.
- **LLaVA (원본)**: Haotian Liu et al., "Visual Instruction Tuning," NeurIPS 2023. (논문 내 [36])
- **InstructBLIP**: Wenliang Dai et al., "InstructBLIP: Towards General-Purpose Vision-Language Models with Instruction Tuning," arXiv:2305.06500. (논문 내 [14])
- **BLIP-2**: Junnan Li et al., "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models," arXiv:2301.12597. (논문 내 [32])
- **Qwen-VL**: Jinze Bai et al., "Qwen-VL: A Frontier Large Vision-Language Model with Versatile Abilities," arXiv:2308.12966. (논문 내 [3])
- **LIMA**: Chunting Zhou et al., "LIMA: Less Is More for Alignment," arXiv:2305.11206. (논문 내 [61])
- **Flamingo**: Jean-Baptiste Alayrac et al., "Flamingo: a Visual Language Model for Few-Shot Learning," arXiv:2204.14198. (논문 내 [2])
- **Scaling instruction-finetuned LMs (FLAN)**: Hyung Won Chung et al., arXiv:2210.11416. (논문 내 [13])
- **LLaVA 프로젝트 홈페이지**: https://llava-vl.github.io
