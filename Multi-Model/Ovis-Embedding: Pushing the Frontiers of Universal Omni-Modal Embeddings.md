# Ovis-Embedding: Pushing the Frontiers of Universal Omni-Modal Embeddings

---

## 1. Executive Summary (10문장 이내)

1. Ovis-Embedding은 텍스트·이미지·비디오·오디오를 **하나의 공유 백본**으로 임베딩하는 omni-modal 임베딩 모델군입니다 (Omni-3B, VL-2B, VL-9B). (p.1, 3)
2. 사전학습된 Qwen2.5-Omni(또는 Qwen3.5)에서 출력 헤드를 제거하고, **마지막 비패딩 토큰의 최종 은닉상태**를 투영 헤드 없이 그대로 임베딩으로 씁니다 (Eq. 1, p.4).
3. 약 50M 쌍(저자는 "placeholder 추정치"라고 명시, p.5)의 텍스트·이미지·비디오·오디오·에이전트 데이터를 구축했습니다.
4. 학습은 4단계입니다. ① LoRA 저랭크 대조학습(focal 가중 InfoNCE + 증류), ② 전체 파라미터 미세조정 + **동종 소스 샘플링**, ③ 오답 중심 **Embedding Distillation**, ④ PCA + 잔차 어댑터 기반 **탄력적 임베딩 차원**(128~2048). (p.9–14)
5. Omni-3B는 MMEB-v3 전체 58.46점으로 차순위(53.27)를 5.19점 앞서고 6개 모달리티 그룹 모두 1위입니다 (Table 1, p.17).
6. Omni-3B는 MAEB·MVEB(beta)에서도 최고 평균을 보고했습니다 (Table 2–3, p.18–19). 단, 순위는 공식 리더보드가 아니라 저자가 로컬 결과를 삽입해 추정한 값입니다.
7. VL-9B는 MMEB-v2에서 81.13, VL-2B는 77.46으로 비교군 내 최고입니다 (Table 5–6, p.20–21).
8. 128차원(16배 축소)에서도 평균 93.2%의 성능을 유지합니다 (Table 7, p.26).
9. 다만 **핵심 설계(native 초기화, focal, 동종 샘플링, ED, LoRA)의 개별 기여를 보여주는 ablation 수치가 본문에 없고**, 통계적 유의성 검정이 없으며, 일부 수치에 내부 불일치가 있습니다 (5절).
10. 모델·코드·데이터 레시피는 "공개 예정"으로, 현재 PDF만으로는 재현할 수 없습니다 (p.2).

### 1-1. 연구 목적과 필요성

**[저자]**
- **목적:** 질의와 후보가 각각 임의의 모달리티 조합(예: 오디오 + 텍스트 → 비디오·PDF 도식·서비스 기록)이어도 매칭되는 **any-to-any 검색**용 단일 임베딩 공간을 만드는 것입니다 (p.2).
- **필요성 1:** 기존 VL 전문 모델은 오디오를 지원하지 않습니다.
- **필요성 2:** 기존 omni 모델은 텍스트–비전 임베더에 오디오 경로를 덧붙이거나 별도 오디오 인코더를 정렬하는 방식입니다. 이 경우 오디오가 "오디오 없이 학습된 기하구조"에 맞춰져 세밀한 정렬이 제한될 수 있다고 봅니다 (p.2, p.22).
- **필요성 3:** 서로 다른 모달리티의 유사도 점수가 **서로 비교 가능하게 보정**되어야 합니다 (p.2).

**[해석]**
- 필요성은 타당하고 실용적입니다. 다만 "retrofit보다 native가 낫다"는 가설은 **동일 데이터·동일 조건의 대조 실험으로 검증되지 않았습니다.**
- 본문의 비교는 서로 다른 학습 데이터·크기의 모델들 사이의 비교입니다.

> 💡 **용어: 임베딩(embedding)** — 입력을 고정 길이 실수 벡터로 바꾼 표현입니다. 의미가 비슷하면 벡터도 가깝습니다.
> 💡 **any-to-any retrieval** — 질의·후보의 모달리티가 무엇이든(단일 또는 혼합) 같은 인덱스에서 검색하는 설정입니다.
> 💡 **Dense retrieval / RAG** — 벡터 유사도로 문서를 찾는 방식과, 찾은 문서를 LLM 생성에 활용하는 방식(Retrieval-Augmented Generation)입니다.
> 💡 **omni-modal** — 텍스트·이미지·비디오·오디오를 모두 다루는 모델입니다.

---

## 2. 핵심 주장과 근거 (위치 표기 포함 · 항목 3 반영)

근거 강도: ◎ = 직접 수치 제시 / △ = 수치는 있으나 교란 요인 존재 / ✕ = 서술만 있고 수치 없음

| # | 핵심 주장 | 근거 | 위치 | 강도 |
|---|---|---|---|---|
| C1 | Native omni 백본 + 투영 헤드 없음으로 오디오까지 하나의 공간에서 정렬 | 구조 서술, 오디오 성과(AUD 50.08) | Fig. 2 (p.3), Eq. 1 (p.4), Table 1 (p.17) | △ (native vs retrofit 통제 실험 없음) |
| C2 | Omni-3B가 MMEB-v3 6개 그룹 모두 1위, 전체 58.46 (차순위 53.27) | 표 수치 | Table 1 (p.17), Fig. 1 (p.1) | ◎ (유의성 검정 없음) |
| C3 | MAEB: Mean(Task) 57.29, Mean(Type) 61.22로 1위 | 표 수치 | Table 2 (p.18) | △ (beta, 로컬 삽입 순위) |
| C4 | MVEB: 61.77 / 59.72로 1위, 6개 중 5개 유형 최고 | 표 수치 | Table 3 (p.19) | △ (동일) |
| C5 | 텍스트 검색도 손해 없음 (RTEB 67.35, MMEB-Text 47.15) | 표 수치 | Table 4 (p.19) | △ (RTEB +0.08p, 도메인 정렬 학습데이터 포함) |
| C6 | VL-9B MMEB-v2 81.13, VL-2B 77.46 (각 비교군 1위) | 표 수치 | Table 5–6 (p.20–21) | ◎/△ (비교군 4개, 규모 불명) |
| C7 | 2B→9B로 +3.67, 특히 비디오 시간 이해에 크게 기여 | 표 수치 | p.21 | △ (백본 세대가 달라 규모 효과만의 결과가 아님) |
| C8 | 동종 소스 샘플링이 성능↑, 배치가 클수록 이득↑, 학습 가속 | 서술만 | p.12 | ✕ |
| C9 | LoRA가 전체 파라미터 학습보다 초기 임베딩 보호에 유리 | 서술만 | p.11 | ✕ |
| C10 | Focal loss가 어려운 질의에 집중, ED가 전반 성능을 추가 향상 | 서술만 | p.10–13 | ✕ |
| C11 | PCA + 어댑터로 128차원에서 93.2% 유지, 단순 절단 대비 +4.30 | 표·그림 | Table 7 (p.26), Fig. 6 (p.15), Fig. 7 (p.27) | ◎ |
| C12 | VisDoc-OOD +23.45p → 일반화 증거 | 표 수치 | p.17, Table 1 | △ (OOD 여부 불확실) |
| C13 | 평가 데이터와 엄격히 중복 제거 | 텍스트 정규화 매칭 + perceptual hash | p.5 | △ (오디오·의미적 중복은 미기술) |

> 💡 **MMEB-v2/v3, MAEB, MVEB, RTEB** — 각각 멀티모달(78개 과제, p.20), 전모달(190개 과제, p.15), 오디오(30개), 비디오(23개), 텍스트 검색(법률·금융·코드·헬스케어) 임베딩 벤치마크입니다 (p.15).
> 💡 **Hit@1 / nDCG@k** — Hit@1은 1순위가 정답인 비율, nDCG@k는 상위 k개 결과의 순위 가중 정확도입니다.
> 💡 **OOD(Out-Of-Distribution)** — 학습 분포 밖의 데이터입니다.

---

## 2-1. 상세 설명: 문제 · 방법(수식) · 구조 · 성능과 한계

### (가) 해결하려는 문제 [저자, p.2]

1. 모달리티별 타워 조합은 파편화(modality fragmentation)를 낳습니다.
2. 서로 다른 모달리티 간 점수가 비교 가능해야 합니다 (calibrated space).
3. 모달리티 내부의 세밀한 구분력도 유지해야 합니다.
4. 배포 환경별 저장·지연·정확도 예산이 달라 차원 유연성이 필요합니다 (p.13).

### (나) 모델 구조 (Fig. 2, p.3–4)

| 변형 | 초기화 | 지원 모달리티 | 특징 | 임베딩 차원 |
|---|---|---|---|---|
| Omni-3B | Qwen2.5-Omni-3B | 텍스트·이미지·비디오·오디오 | Thinker 유지, **Talker 제거**, TMRoPE로 오디오-비디오 시간 정렬 | 2,048 |
| VL-2B | Qwen3.5-2B | 텍스트·이미지·비디오 | 24층, hidden 2,048 | 2,048 |
| VL-9B | Qwen3.5-9B | 텍스트·이미지·비디오 | 32층, hidden 4,096 | 4,096 |

(차원: p.16)

**[저자]** 임베딩 추출(Eq. 1, p.4):

$$\mathbf{e}(x)=\mathbf{h}^{(L)}_{\ell(x)}\in\mathbb{R}^{d}\tag{1}$$

- $x$: 입력(단일 또는 혼합 모달리티, 작업 지시문 포함)
- $L$: 백본 층 수
- $\ell(x)$: 마지막 유효(비패딩) 토큰 위치
- $\mathbf{h}^{(L)}_{\ell(x)}$: 그 위치의 최종층 은닉상태
- $d$: 백본의 은닉 차원 (투영층이 없으므로 그대로 상속)

유사도와 검색에는 코사인 유사도를 씁니다.

> 💡 **Bi-encoder** — 질의와 후보를 따로 인코딩해 벡터 유사도로 비교하는 구조입니다. (cross-encoder는 둘을 함께 넣어 정확하지만 느립니다.)
> 💡 **Last-token pooling** — 디코더 전용 모델에서 마지막 토큰의 상태를 문장 전체 요약으로 쓰는 방식입니다. 인과적 어텐션이라 마지막 토큰만 앞의 모든 토큰을 볼 수 있기 때문입니다.
> 💡 **Thinker–Talker** — Qwen2.5-Omni에서 Thinker는 이해·추론을 맡는 트랜스포머, Talker는 음성 생성기입니다. 이 논문은 Talker를 버립니다 (p.4).
> 💡 **TMRoPE(Time-aligned Multimodal Rotary Position Embedding)** — 오디오와 비디오 토큰의 시간 위치를 맞춰 주는 위치 인코딩입니다 (p.4).
> 💡 **Gated DeltaNet** — 선형 어텐션 계열 층입니다. Qwen3.5는 이 층 3개당 완전 어텐션 1개를 번갈아 쌓아 긴 문맥을 효율적으로 처리합니다 (p.4).

### (다) 학습 데이터 (p.4–9, Fig. 3)

- **구성:** 이미지(분류·QA·검색·그라운딩 + 오픈월드 상품검색), 비디오, 오디오, 텍스트(5가지 패러다임 + RTEB 유사 도메인 + MTEB 유형), 에이전트(도구·GUI·지식).
- **형식:** (질의, 정답, 부정예 K개) 튜플. 중복 제거, 거짓 음성 필터링, 모달리티 균형을 맞춥니다 (p.5).
- **비디오:** 웹 검색 → VLM이 균등 샘플 프레임으로 관련도 0/1/2 판정 → 최고 등급만 유지 → 정답 비디오를 고정한 채 VLM이 질의를 보수적으로 수정 (p.6, Fig. 10–11).
- **다중 조건 텍스트 검색 부정예 (Eq. 3, p.8):**

$$d^{+}=d_{\text{src}},\qquad \mathcal{N}(q_k)=\{\mathrm{HN}_j\}_{j=1}^{k},\qquad \mathrm{HN}_j:=d_{\text{src}}[f_j\to h_j]\tag{3}$$

  - $q_k=\langle f_1,\dots,f_k\rangle$: $k$개 조건의 합집합 질의
  - $d_{\text{src}}$: 조건을 추출한 원문서 (정답)
  - $h_j$: 조건 $f_j$를 무효화하되 핵심 단어는 유지한 대체 문구
  - $\mathrm{HN}_j$: $f_j$만 바꾼 hard negative (조건 하나만 다름)

- 명령어 추종 검색의 관련성 정의 (Eq. 2, p.7):

$$\mathrm{rel}(q,d)\iff (d\models x)\wedge(d\models c),\quad q=x\oplus c\tag{2}$$

  $x$는 기본 요청, $c$는 제약 조건, $\models$는 "만족한다", $\oplus$는 결합입니다. 이상적인 hard negative는 $d^-\models x$이지만 $d^-\not\models c$입니다.

> 💡 **Hard negative** — 정답과 비슷해 모델이 헷갈리는 오답 후보입니다.
> 💡 **False negative** — 실제로는 정답인데 오답으로 취급된 후보입니다. 학습을 방해합니다.
> 💡 **BM25** — 단어 빈도 기반의 전통적 검색 점수입니다 (에이전트 부정예 채굴에 사용, p.9).
> 💡 **Visual grounding** — 텍스트 설명에 해당하는 이미지 영역을 찾는 과제입니다.
> 💡 **Perceptual hash** — 이미지가 시각적으로 유사하면 비슷한 해시가 나오도록 하는 지문 기법입니다 (중복 검출, p.5).

### (라) 제안 방법: 4단계 학습

**Stage-1: 저랭크(LoRA) 대조 사전학습 (p.9–11)**

배치 $N$개의 튜플 $(x_i,y_i^+,\{y^-\_{i,k}\}\_{k=1}^K)$에서 후보 풀 $\mathcal{C}=\{y_j^+\}\_{j=1}^N\cup\{y^-_{j,k}\}$ (크기 $N(1+K)$ )을 모든 질의가 공유합니다. 데이터 병렬 워커 간에도 후보를 모읍니다.

```math
\mathrm{sim}(x,y)=\frac{\mathbf{e}(x)^{\top}\mathbf{e}(y)}{\|\mathbf{e}(x)\|_2\,\|\mathbf{e}(y)\|_2}
```

$$\pi_i=\frac{e^{\mathrm{sim}(x_i,y_i^+)/\tau}}{Z_i},\qquad \ell_i=-\log\pi_i\tag{5}$$

- $\tau$: 온도(분포의 뾰족함 조절)
- $Z_i=\sum_{y\in\mathcal{C}}e^{\mathrm{sim}(x_i,y)/\tau}$: 분배 함수
- $\pi_i$: 정답에 대한 softmax 확률
- $\ell_i$: 질의별 InfoNCE 손실

**Focal 가중 대조 손실 (Eq. 6–7):**

$$a_i=\mathrm{sg}\!\left[(1-\pi_i)^{\gamma}\right],\qquad \tilde w_i=\frac{a_i}{\frac1N\sum_{k=1}^{N}a_k}\tag{6}$$

$$\mathcal{L}_{\text{focal}}=\frac1N\sum_{i=1}^{N}\tilde w_i\,\ell_i=-\frac{\sum_{i=1}^{N}a_i\log\pi_i}{\sum_{i=1}^{N}a_i}\tag{7}$$

- $\mathrm{sg}[\cdot]$: stop-gradient (가중치 자체로는 역전파하지 않음)
- $\gamma\ge0$: 난이도 강조 강도 ($\gamma=0$이면 균등 평균)
- $\tilde w_i$: 평균 1로 정규화된 가중치. 총 손실 규모는 유지하고 예산만 어려운 질의로 재분배합니다.
- $\pi_i$가 높은(이미 잘 푼) 질의는 가중치가 낮아집니다.

**임베딩 증류 손실 (Eq. 8–10):**

$$t_i(c)=\frac{e^{\mathrm{sim}_T(x_i,c)/\tau}}{\sum_{c'\in\mathcal{C}}e^{\mathrm{sim}_T(x_i,c')/\tau}}\tag{8}$$

$$q_i(c)=\frac{e^{\mathrm{sim}(x_i,c)/\tau}}{\sum_{c'\in\mathcal{C}}e^{\mathrm{sim}(x_i,c')/\tau}}\tag{9}$$

$$\ell_i^{\mathrm{KD}}=\mathrm{KL}(t_i\,\|\,q_i)=\sum_{c\in\mathcal{C}}t_i(c)\log\frac{t_i(c)}{q_i(c)},\qquad \mathcal{L}_{\text{dist}}=\frac1N\sum_{i=1}^{N}\ell_i^{\mathrm{KD}}\tag{10}$$

- $\mathrm{sim}_T$: 교사 임베딩 모델의 유사도
- $t_i$, $q_i$: 교사와 학생의 후보 분포
- 교사의 $\log t_i(c)$는 오프라인으로 계산해 각 학습 샘플에 저장합니다 (교사 추론 불필요, p.11).

$$\mathcal{L}_{\text{stage-1}}=\lambda_{\text{focal}}\mathcal{L}_{\text{focal}}+\lambda_{\text{dist}}\mathcal{L}_{\text{dist}}\tag{11}$$

$\lambda_{\text{focal}},\lambda_{\text{dist}}\ge 0$이며, 저자는 두 값을 같게 두고 튜닝하지 않았다고 밝힙니다 (p.11).

LoRA를 쓴 이유는 다음과 같습니다 [저자, p.11].
- 초기 임베딩 공간이 "혼돈 상태"여서, 이때 전체 파라미터를 학습하면 불안정한 그래디언트가 사전학습 의미 구조를 훼손한다는 설명입니다.
- 이 주장에 대한 수치 비교는 없습니다.

> 💡 **대조학습(Contrastive learning)** — 정답 쌍은 가깝게, 오답 쌍은 멀게 학습하는 방식입니다.
> 💡 **In-batch negative** — 같은 배치의 다른 샘플들의 정답을 공짜 오답으로 재활용하는 방식입니다.
> 💡 **InfoNCE** — 정답을 후보들 중에서 골라내는 softmax 교차엔트로피 형태의 대조 손실입니다.
> 💡 **Focal loss** — 쉬운 샘플의 손실 비중을 줄이고 어려운 샘플에 집중하는 손실입니다 (원래 객체 탐지용, Lin et al. 2017).
> 💡 **KL divergence / 지식 증류** — 두 확률분포의 차이입니다. 증류는 교사의 "부드러운 확률분포"를 학생이 모방하게 해, one-hot 정답에서는 얻을 수 없는 순위 정보를 전달합니다 (Hinton et al. 2015).
> 💡 **LoRA** — 가중치 변화를 저랭크 행렬로 제한해 적은 파라미터만 학습하는 기법입니다. (LoRA 원 논문은 이 PDF 참고문헌에 없어, 일반 지식으로 설명했습니다.)

**Stage-2: 전체 파라미터 + 동종 소스 샘플링 (p.12, Eq. 12, Fig. 4)**

```math
\begin{aligned}
\mathcal{B}_d&=\Big\{(x_i,y_i^+,\{y^-_{i,k}\}_{k=1}^K)\Big\}_{i=1}^{B_\mu},\quad s_i=d\ \ \forall i,\\
\mathcal{C}(\mathcal{B}_d)&=\mathrm{Dedup}_{\mathrm{hash}}\Big(\{y_j^+\}\cup\{y^-_{j,k}\}\Big),\\
\mathcal{L}(\mathcal{B}_d)&=-\frac1{B_\mu}\sum_{i=1}^{B_\mu}\log\frac{e^{\mathrm{sim}(x_i,y_i^+)/\tau}}{\sum_{y\in\mathcal{C}(\mathcal{B}_d)}e^{\mathrm{sim}(x_i,y)/\tau}}
\end{aligned}
```

- $B_\mu$: 마이크로배치 크기
- $s_i$: $i$번째 샘플의 출처 데이터셋
- $d$: 데이터셋 인덱스. 이후 차원 $d$와 기호가 겹칩니다.
- $\mathrm{Dedup}_{\mathrm{hash}}$: 해시 기반 중복 제거 (정답과 동일한 후보가 오답으로 섞이는 것을 방지)

핵심은 **한 마이크로배치는 한 데이터셋에서만 구성**하되, 옵티마이저 한 스텝에서는 여러 데이터셋의 그래디언트를 평균한다는 점입니다. 저자는 이것이 모달리티·문장 길이 같은 지름길(shortcut) 학습을 줄이고 망각을 완화한다고 설명합니다 (p.12).

**Stage-3: Annealing Embedding Distillation (p.12–13, Eq. 13–15, Fig. 5a)**

절차는 다음과 같습니다.
1. 교사가 맞힌 샘플만 남깁니다.
2. 그 중 학생이 틀린 샘플을 업샘플링합니다.
3. 학생이 자신 없는 질의일수록 교사 신호를 강하게 줍니다.

$$d_i=\left(1-e^{-\ell_i}\right)^{\gamma}=(1-\pi_i)^{\gamma}\tag{13}$$

$$\lambda_i=\lambda_{\min}+(\lambda_{\max}-\lambda_{\min})\,\mathrm{sg}[d_i]\tag{14}$$

$$\mathcal{L}_{\mathrm{ED}}=\underbrace{\frac1N\sum_{i=1}^N\ell_i}_{\mathcal{L}_{\mathrm{NCE}}^{s}}+\frac1N\sum_{i=1}^N\lambda_i\,\ell_i^{\mathrm{KD}}\tag{15}$$

- $d_i\in[0,1]$: 질의 난이도 ( $\pi_i=q_i(y_i^+)$ )
- $\gamma > 0$: 학생이 자신감을 얻을 때 교사 영향이 얼마나 빨리 줄어드는지 조절
- $\lambda_{\min},\lambda_{\max}$: 증류 가중치의 하한과 상한
- 순방향 KL을 쓰는 이유: InfoNCE가 이미 정답 모드에 확률을 모으므로, mode-seeking인 역방향 KL은 중복이라는 설명입니다.

> 💡 **Annealing** — 학습 후반에 난이도·데이터 구성을 조정하며 마무리하는 단계입니다.
> 💡 **Forward vs Reverse KL** — 순방향 $\mathrm{KL}(t\|q)$은 교사 분포 전체를 덮으려 하고(mass-covering), 역방향은 한 모드에 몰립니다(mode-seeking).
> 💡 **Catastrophic forgetting** — 새 데이터를 학습하며 이전 능력이 사라지는 현상입니다.

**Stage-4: 탄력적 임베딩 (p.13–14, Eq. 16–21, Fig. 5b)**

인코더를 동결한 뒤 $D=2048$차원 단위 벡터를 PCA 기저로 회전하고, 폭 $d\in\{128,256,512,1024\}$마다 선형 잔차 어댑터를 붙입니다.

$$z^{(d)}(v)=\frac{u_{1:d}}{\|u_{1:d}\|_2},\quad u=f_d(V^{\top}v),\quad \mathrm{sim}_d(\mathbf{x},\mathbf{y})=\langle z^{(d)}(\mathbf{x}),z^{(d)}(\mathbf{y})\rangle\tag{16}$$

$$f_d(\tilde v)=\tilde v+W_d\tilde v,\quad W_d\big|_{\text{init}}=0,\quad \tilde v=V^{\top}v\tag{17}$$

$$\Sigma=\sum_m p_m\,\mathbb{E}_{\mathbf{y}\sim\mathcal{C}_m}[\mathbf{y}\mathbf{y}^{\top}],\qquad \Sigma=V\Lambda V^{\top},\quad \mu_1\ge\dots\ge\mu_D\tag{18}$$

- $V$: 공유 직교 PCA 기저 (회전만 하므로 전체 폭 기하는 보존됨)
- $\Lambda=\mathrm{diag}(\mu_1,\dots,\mu_D)$: 고유값 대각행렬
- $\mathcal{C}_m$: 모달리티 $m$의 후보 풀 (균형 다운샘플링)
- $p_m$: 모달리티 혼합 가중치 ($\sum_m p_m=1$). 균등 가중치가 안정적 기본값이라고 보고합니다.
- $u_{1:d}$: 처음 $d$개 좌표. 절단 후 재정규화가 필요한 이유는 절단된 단위벡터의 노름이 줄어 코사인·내적· $\ell_2$ 거리 간 일관성이 깨지기 때문입니다.

어댑터는 후보 코퍼스만으로 **비지도** 학습합니다. Matryoshka-Adaptor(Yoon et al. 2024)의 목적함수를 사용합니다.

$$\mathcal{L}^{\mathrm{pair}}_d=\frac{1}{|B|(|B|-1)}\sum_{i\in B}\sum_{j\in B,\,j\ne i}\Big|\mathrm{sim}_d(\mathbf{y}_i,\mathbf{y}_j)-\mathrm{sim}_D(\mathbf{y}_i,\mathbf{y}_j)\Big|\tag{19}$$

$$\mathcal{L}^{\mathrm{topk}}_d=\frac{1}{|B|S}\sum_{i\in B}\sum_{k=1}^{S}\Big|\mathrm{sim}_d(\mathbf{y}_i,\mathbf{y}_{n_k(i)})-\mathrm{sim}_D(\mathbf{y}_i,\mathbf{y}_{n_k(i)})\Big|\tag{20}$$

$$\mathcal{L}_d=\mathcal{L}^{\mathrm{topk}}_d+\alpha\,\mathcal{L}^{\mathrm{pair}}_d\tag{21}$$

- $B$: 배치, $S$: 앵커당 이웃 수
- $n_k(i)$: 앵커 $i$의 사전 계산된 $k$번째 이웃
- $\alpha$: 두 항의 균형 계수
- 직관: 짧은 폭에서도 전체 폭의 쌍별 유사도(전역)와 근접 이웃 유사도(국소)를 보존하도록 합니다.
- 선형 구조라서 서빙 시 $P_d=[(I+W_d)V^\top]_{1:d,:}$ 하나의 행렬곱으로 합쳐집니다.

> 💡 **MRL(Matryoshka Representation Learning)** — 벡터의 앞쪽 $d$차원만 잘라 써도 성능이 유지되도록 학습하는 방법입니다. 저자는 이를 다목적 학습에 섞으면 전체 차원 성능이 떨어진다고 주장합니다 (수치 없음, p.13).
> 💡 **PCA(주성분 분석)와 고유값 분해** — 분산이 큰 방향 순으로 좌표축을 재배열합니다. 앞 좌표에 중요한 정보가 몰려 절단해도 손실이 적습니다.
> 💡 **잔차(residual) 어댑터** — 입력을 그대로 통과(skip)시키고 작은 보정 $W_d\tilde v$만 더하는 구조입니다. 0으로 초기화하면 학습 시작 시 항등 변환이 됩니다.
> 💡 **Retention(%)** — 축소 차원 평균 점수 ÷ 2048차원 평균 점수입니다 (Table 7).

### (마) 성능 향상 [저자]

- **MMEB-v3 (Table 1):** 전체 58.46 (e5-omni-7B 47.14, Omni-Embed-Nemotron-3B 43.60). 31개 항목 중 1위 22개, 2위 8개, 3위 이하는 MultiConIR 하나입니다 (p.18). 제가 표를 세어 본 결과도 1위 22개, 2위 8개로 일치했습니다.
- **MAEB / MVEB:** Table 2–3 참조. MVEB에서 비디오 클러스터링은 LCO-Omni-7B(27.35)에 2.00점 뒤집니다.
- **MMEB-v2:** VL-9B 전체 81.13. 이미지 4개 하위과제 모두 1위, 비디오 중 V-CLS·V-MRET 1위입니다.
- **탄력 임베딩 (Table 7):** 평균 58.00(2048) → 54.08(128). 단순 절단은 49.78입니다. VisDoc과 Agent는 128차원에서 각 −6.97, −6.18로 손실이 큽니다.

### (바) 한계

**[저자 명시]**
- 다중 조건 텍스트 검색(MultiConIR −7.94p), 메모리 검색(−2.79p), 비디오 클러스터링 (p.17–19).
- VL 모델은 비디오 일부 과제에서 열세입니다. 특히 Octen-VL이 비디오 전체에서 앞섭니다 (Table 5–6).
- 다국어·도메인 다양 문서 검색은 향후 과제입니다 (p.21).
- 저차원에서 VisDoc·Agent 같은 세밀 과제는 "투영이 아니라 용량" 문제입니다 (p.26).

**[해석]**
- 핵심 설계의 효과가 **ablation 없이 서술만** 있어 어떤 요소가 성능을 만들었는지 알 수 없습니다.
- 학습 데이터 규모가 큽니다(약 50M, 일부 비공개·LLM/VLM 합성). 따라서 성능이 방법의 결과인지 데이터의 결과인지 분리되지 않습니다.
- 평가가 주로 쌍(pair) 검색 벤치마크에 한정되어, "임의 조합 any-to-any" 주장 전체를 검증하지는 못합니다.

---

## 4. 저자 보고 vs 내 해석 (연구 주제 · 방법 · 결과)

| 구분 | 저자가 직접 보고한 내용 [저자] | 내 해석 [해석] |
|---|---|---|
| **연구 주제** | 모달리티 파편화를 극복하는 범용 omni 임베딩 및 any-to-any 검색 (p.1–2) | 문제 설정은 시의적절하고, 특히 오디오 포함 공개 자원이 적다는 지적(p.2)은 기여 가치가 큽니다. 그러나 "native가 우월하다"는 핵심 가설은 가정에 가깝고 검증되지 않았습니다. |
| **방법: 구조** | Qwen-Omni/Qwen3.5 백본 + 마지막 토큰 풀링, 투영 헤드 없음 (Eq. 1) | 매우 단순하고 재현이 쉬운 설계입니다. 이는 백본 사전학습의 질에 크게 의존한다는 뜻이기도 합니다. |
| **방법: 손실** | Focal 가중 InfoNCE + 증류(Eq. 4–11), ED(Eq. 13–15) | Eq. 7은 정규화 가중 CE와 동치라서 구현이 간단합니다. 다만 Stage-1에서 증류 분포를 "오프라인 저장"한다는 설명(p.11)과 "후보 풀이 배치 내 무작위 구성"이라는 설명(p.9, p.12)이 어떻게 양립하는지 불명확합니다. |
| **방법: 샘플링** | 동종 소스 배치로 지름길 방지, 해시 dedup (Eq. 12) | 합리적인 아이디어입니다. 해시 dedup은 **동일 항목만** 제거하므로 의미상 같은 정답(거짓 음성)은 남을 수 있습니다. 동일 데이터셋 배치에서는 이 위험이 오히려 커질 수 있습니다 (추정). |
| **방법: 탄력 임베딩** | PCA + 선형 잔차 어댑터, 비지도(Eq. 16–21) | 실용적이고 정직한 분석입니다 ("용량 문제"를 인정). 다만 Stage-2 동결 인코더 기준이라 최종 모델과 다릅니다(아래). |
| **결과: 성능** | MMEB-v3 58.46 등 다수 SOTA (Table 1–6) | 방향성은 일관되게 우수합니다. 그러나 **단일 실행, 신뢰구간 없음**, 비교군·규모·학습데이터가 통제되지 않았습니다. |
| **결과: 일반화** | VisDoc-OOD +23.45p, RTEB에서 4B 텍스트 전용 모델과 동급 (p.17, 19) | 인상적이지만 증거로는 부분적입니다. 학습에 DocVQA/InfoVQA와 RTEB 유사 도메인 데이터가 포함되어, "OOD"와 "간접적 in-domain"을 구분할 수 없습니다. |
| **결과: 압축** | 128차원에서 93.2% 유지 (Table 7) | 설득력 있는 결과이고 한계 분석도 솔직합니다. 다만 VL 모델 및 MMEB-v2·MAEB·MVEB에서는 검증되지 않았습니다. |

---

## 5. 통계적으로 취약한 부분 / 비교 불가능한 수치

### 5-A. 통계적 취약점 ⚠️

| # | 문제 | 위치 |
|---|---|---|
| S1 | **신뢰구간·표준편차·시드 반복·유의성 검정이 전혀 없음.** 모든 수치가 단일 점추정입니다. | 전체 (Table 1–7) |
| S2 | **작은 격차를 "최고"로 표기.** RTEB 67.35 vs 67.27(+0.08), MMEB-v2 VL-9B +1.04, 이미지 QA −0.30 등입니다. 변동성 추정 없이는 우열을 판단할 수 없습니다. | Table 4–5 |
| S3 | **"31개 중 22개 1위"는 이중 계산.** Overall·그룹 평균과 그 구성 하위과제를 함께 세어 독립 증거가 아닙니다. | p.18 |
| S4 | **하위 그룹의 과제 수 미공개.** A-RET, Memory 등 작은 그룹은 분산이 클 가능성이 있습니다. | Table 1 |
| S5 | **MAEB/MVEB 순위는 추정치.** 리더보드 스냅샷에 로컬 결과를 삽입한 Borda 순위이고, 공식 제출은 아직입니다. beta 버전이며 스냅샷 날짜도 불명입니다. | p.18, Table 2–3 |
| S6 | **ablation 부재.** 동종 샘플링, focal, ED, LoRA 우위에 대한 수치가 없습니다. (5절 서두에서 "구성요소 기여"와 "OOD 질의 정성 행동"을 평가하겠다고 했으나, 해당 결과가 제공된 본문에 없습니다.) | p.11–13, p.15 |
| S7 | **OOD·일반화 해석 위험.** 학습에 DocVQA/InfoVQA(p.5), RTEB 유사 도메인, FollowIR/MultiConIR 계열 학습 데이터(p.7–8)가 포함됩니다. 중복 제거는 텍스트 정규화와 perceptual hash로만 기술되어 의미적 누출과 오디오·비디오는 불명확합니다 (p.5). | p.5, 7, 8 |
| S8 | **데이터 규모 수치는 "placeholder 추정"**이라고 각주에 명시되어 있습니다. | p.5 각주 1 |

### 5-B. 비교 불가능하거나 내부 불일치한 수치 ❗

| # | 항목 | 설명 | 위치 |
|---|---|---|---|
| N1 | **오디오 격차 불일치** | 본문은 "차순위보다 7.04p"라 하지만, 표에서 차순위는 LCO-Omni-7B(43.17)이므로 $50.08-43.17=6.91$입니다. 7.04는 e5-omni-7B(43.04) 기준 값입니다. | p.18 vs Table 1 |
| N2 | **InfoSearch 격차 불일치** | 본문은 "런너업 대비 +23.88"이라 하지만, 표의 차순위는 LCO 59.25이므로 $75.08-59.25=15.83$입니다. 23.88은 Tianmu(51.20) 대비 값입니다. | p.17 vs Table 1 |
| N3 | **Fig. 1 캡션 vs Table 5** | 캡션은 VL-9B가 MMEB-v2 5개 그룹 중 4개에서 1위라고 합니다. 그러나 Fig. 1에 표시된 5개 그룹(IMG, IMG-CLS, IMG-QA, V-MRET, VISDOC)은 Table 5 수치상 모두 Ovis가 최고입니다. 캡션 오류이거나 다른 기준일 수 있어 확인이 필요합니다. | p.1 vs p.20 |
| N4 | **Table 7의 2048차원 ≠ Table 1 헤드라인** | Table 7은 평균 58.00, Table 1은 58.46입니다 (예: Audio 48.02 vs 50.08). 어댑터가 "stage-2 학습 후" 동결 인코더에 붙는다고 서술되어(p.13–14), Stage-3(ED)이 반영되지 않은 모델로 보입니다. 이는 **제 추정이며** 논문이 명시하지는 않았습니다. Table 7 캡션의 "full pipeline"은 오해 소지가 있습니다. | p.14, 26 |
| N5 | **Fig. 7 캡션 평균값 불명** | "512에서 59.28 vs 58.46, 128에서 55.49 vs 56.12"는 Table 7(512: 57.55, 128: 54.08)과 맞지 않습니다. 평균 산식도 설명이 없습니다. Fig. 6b 캡션이 언급한 "Table 7의 task weights"도 Table 7에 없습니다. | p.15, 27 |
| N6 | **모델 크기 불일치** | Table 1은 Omni-3B vs 7B/8B 모델이고, Table 5–6 비교군(seed1.6, Octen, DME)은 파라미터 수가 미기재입니다. Talker 제거 후 Omni-3B의 실제 파라미터 수도 미보고입니다. | Table 1, 5, 6 |
| N7 | **학습데이터·계산량 비통제** | 비교 모델의 학습 데이터·규모가 달라 성능 차이를 방법의 효과로 귀속할 수 없습니다. | 전체 |
| N8 | **지표 혼합** | MMEB 전체 평균은 Hit@1과 nDCG@5가 섞인 평균이고, RTEB는 nDCG@10입니다. 벤치마크 간 점수 직접 비교는 불가능합니다. | p.16 |
| N9 | **RTEB 부분 평가** | 15개 영어 공개 분할만 사용했고 비공개 세트는 제외되어 공식 RTEB 리더보드 점수와 동일하지 않습니다. 비교군도 4개뿐입니다. Legal(49.61)은 Qwen3-Embedding-4B(62.67)보다 크게 낮습니다. | Table 4 |
| N10 | **Fig. 1 막대 길이** | 벤치마크 내에서 정규화된 길이라 축 간·벤치마크 간 크기 비교가 불가능합니다 (캡션). Fig. 7도 패널별 스케일과 지표가 다릅니다 (캡션). | p.1, 27 |
| N11 | **비교 대상 비일관** | Fig. 1(MMEB-v3)의 비교군에는 Qwen3-VL-Embedding-2B가 있고, Table 1에는 Tianmu-Emb-Uni(8B)가 있습니다. | p.1 vs 17 |
| N12 | **중복 행** | Table 3의 ebind-audio-vision과 ebind-full은 모든 수치가 동일합니다. | p.19 |
| N13 | **API·문서 기반 baseline** | Seed1.6-Embedding과 Octen은 상용 API·웹 문서이며 접근일이 있습니다. 버전 고정과 재현이 어렵습니다 (p.23–24 참고문헌). | p.23–24 |
| N14 | **모델 간 교차 평가 부재** | VL-9B는 MMEB-v3/MAEB/MVEB에서, Omni-3B는 MMEB-v2에서 평가되지 않았습니다. 초록의 "family achieves SOTA on all five"는 모델별로 다른 벤치마크 결과의 합입니다. 결론(p.22)은 RTEB를 빼고 4개만 나열합니다. | p.1, 22 |

---

## 6. 문서가 답하지 않는 질문

1. **각 구성요소의 기여는?** (native 초기화 vs retrofit, focal 유무, 동종 vs 혼합 샘플링, LoRA vs full, Stage별 향상, ED 유무) 수치가 없습니다.
2. **하이퍼파라미터는?** $\tau$, $\gamma$, $\lambda_{\min/\max}$, $\alpha$, $S$, $K$, LoRA rank, 배치·마이크로배치 크기, 학습률, 스텝 수, GPU 수, 학습 시간이 없습니다 (H100 사용만 언급, p.16).
3. **교사(teacher) 모델은 무엇인가?** "complementary experts"의 정체와 모달리티별 선택 기준이 없습니다 (p.10–13).
4. **오프라인 교사 분포 계산이 배치 내 무작위 후보 풀과 어떻게 정합되나?** (p.11 vs p.9)
5. **모달리티·태스크별 데이터 비율, 최종 샘플 수, 라이선스, 비공개 비중**은? 오디오·비디오의 중복 제거 방법은?
6. **실제 효율 측정치**(지연, 처리량, 인덱스 저장량, Talker 제거 후 파라미터 수)는? 압축은 차원 비율로만 언급됩니다.
7. **MRL을 쓰면 전체 차원 성능이 하락한다**는 주장(p.13)의 정량 근거는?
8. **Stage-4 어댑터가 정확히 어느 체크포인트(Stage-2/3)에 붙는가?** 본문 순서(p.9)와 서술(p.13–14)이 어긋납니다.
9. **진짜 혼합(interleaved)·any-to-any 질의**(예: 오디오 + 텍스트 → 비디오)에 대한 정량·정성 평가는? 5절 서두의 "OOD 질의 정성 행동"에 해당하는 결과가 본문에 없습니다.
10. **다국어·긴 오디오/비디오·노이즈·프롬프트/지시문 변경에 대한 강건성**은?
11. **공정성·편향·안전성**과 웹 크롤링 데이터(Quark 등)·VLM 합성 라벨 오류의 영향은?
12. **규모 효과의 분리:** 2B→9B는 백본 세대(Qwen3.5)와 크기가 함께 바뀌므로 순수한 스케일 효과는 알 수 없습니다.
13. **baseline 평가 프롬프트의 공정성:** "공식 프롬프트 사용"(p.16)이라 했지만 세부 설정은 불명입니다.
14. **공개 시점과 라이선스는?** ("will open-source"라고만 서술, p.2)

---

## 7. 가장 중요한 그림 5개 해석

### ① Figure 1 (p.1): 성능 레이더 요약

- **[저자]** Omni-3B는 MMEB-v3 6개 그룹에서 모두 1위이고, VL-9B는 MMEB-v2에서 선두라고 합니다.
- **[해석]** 한눈에 "균형 잡힌 우위"를 보여 주는 마케팅성 요약입니다. 막대는 벤치마크 내 정규화이므로 길이로 격차 크기를 읽으면 안 되고 숫자 라벨을 봐야 합니다. 캡션의 "4/5"와 Table 5의 불일치는 5절 N3에 정리했습니다.

### ② Figure 2 (p.3): 아키텍처

- **[저자]**
  - (a) Omni-3B는 텍스트 토크나이저, 비전 인코더, 오디오 인코더가 만든 토큰을 인터리빙해 Qwen2.5-Omni Thinker(TMRoPE)에 넣습니다.
  - (b) VL은 Qwen3.5(3×Gated DeltaNet + 1×full attention 반복)를 씁니다.
  - 둘 다 마지막 토큰의 최종층 상태가 e(x)입니다. 입력에 "Represent this input for retrieval:" 형태의 지시문을 붙이는 예가 그려져 있습니다.
- **[해석]** 모달리티별 헤드·프로젝터가 없다는 점이 핵심입니다. 모든 정렬 책임이 백본과 학습 데이터에 있고, 단순해서 이식성은 좋습니다.

### ③ Figure 4 (p.10): 데이터 중심 학습과 동종 소스 샘플링

- **[저자]**
  - (a) 다양한 모달리티 데이터를 섞고 비텍스트 타깃을 업웨이트합니다.
  - (b) 한 마이크로배치를 하나의 데이터셋에서만 뽑고 해시 중복 제거로 정답 충돌을 막습니다.
- **[해석]** "배치 구성이 곧 부정예의 질을 결정한다"는 메시지입니다. 모달리티가 섞이면 모델이 "이미지냐 텍스트냐"만으로 쉽게 구분해 학습 신호가 약해지는데, 이를 막는 것입니다. 다만 효과의 정량 증거는 본문에 없습니다.

### ④ Figure 5 (p.13): ED와 추론 시 저랭크 분해

- **[저자]**
  - (a) 교사가 맞힌 샘플만 남기고 학생 실패를 업샘플링해, 순방향 KL로 교사의 후보 순위분포를 전달합니다.
  - (b) 후보 코퍼스의 비중심 2차 모멘트를 혼합해 한 번 고유분해하고, 0 초기화 잔차 어댑터와 결합해 단일 $d\times D$ 행렬로 서빙합니다.
- **[해석]**
  - 학습 단계(ED)와 배포 단계(탄력 임베딩)를 한 그림에 담은 것은, 이 논문이 "정확도뿐 아니라 배포까지 포함한 레시피"라는 점을 드러냅니다.
  - PCA 회전이 등거리 변환이라 2048차원 성능이 변하지 않는다는 점이 MRL 대비 실용적 장점입니다.

### ⑤ Figure 6 / Figure 7 (p.15, 27): 임베딩 폭별 성능

- **[저자]** 단순 절단, PCA만, 어댑터만, PCA + 어댑터를 비교했습니다.
  - 512 이상에서는 PCA가 이득의 대부분을, 128에서는 어댑터가 더 큰 비중을 차지하며, 어느 하나만으로는 조합을 못 따라갑니다.
  - 어댑터 단독은 1024에서 비디오·VisDoc·Agent가 단순 절단보다 낮아질 수 있습니다 (Fig. 7 캡션).
- **[해석]**
  - 가장 엄밀한 ablation 성격의 그림입니다 (본문 중 구성요소 비교가 있는 거의 유일한 곳).
  - Table 7 기준 128차원 평균은 54.08(조합) vs 49.78(단순 절단)입니다.
  - 모달리티별 편차가 큽니다 (Audio −0.33, VisDoc −6.97). 저차원에서는 "투영 개선"보다 "용량"이 병목이라는 저자 분석(p.26)이 설득력 있습니다.
  - 단, Fig. 7 캡션의 평균값은 5절 N5의 이유로 검증이 어렵습니다.

---

## 8. 결론

### 8-0. 저자가 제시한 시사점과 후속 계획

**[저자]**
- **시사점 (p.22):** native 멀티모달 백본 + 데이터 중심 학습 + 임베딩 특화 최적화로 MMEB-v3/v2, MAEB, MVEB에서 SOTA를 달성했고, 유연한 차원으로 배포 효율을 확보했습니다. 이는 native omni 모델이 범용 검색의 기반이 될 수 있음을 시사합니다.
- **후속 계획:**
  - 체크포인트, 학습·데이터 구축 레시피, 추론 코드, 통합 평가 툴킷을 공개할 예정입니다 (p.2).
  - 카메라레디 버전에서 데이터 수치를 확정합니다 (p.5 각주).
- **개선 방향으로 직접 언급한 영역:**
  - 다중 조건 텍스트 검색, 메모리 검색, 비디오 클러스터링
  - 시간-의미 매칭 강화, 다국어·도메인 다양 문서 검색 (p.17–21)
- 공식 리더보드 제출은 "아직 제출 전"이라고만 있고, 제출 계획은 명시되지 않았습니다 (p.18).

### 8-1. 모델의 일반화 성능 향상 가능성 (중점)

**(1) 저자가 제시한 일반화 단서**

| 단서 | 위치 |
|---|---|
| VisDoc-OOD 67.94 (차순위 대비 +23.45) | p.17 |
| 모달리티 전반 균형 성능 (6개 그룹 1위, 하나의 우세 모달리티 때문이 아니라는 주장) | p.16 |
| 텍스트 검색 데이터(RTEB 유사)가 이미지·오디오·비디오로의 일반화를 돕는다는 서술 | p.8 |
| 3B Omni가 4B 텍스트 전용 모델과 RTEB·MMEB-Text에서 동급 | Table 4 |
| 동종 샘플링이 지름길을 줄이고 분류·클러스터링에도 이득 | p.12 |
| LoRA 초기화가 사전학습 의미를 보호 | p.11 |
| 증류가 약한 영역을 보완하고 강한 영역을 유지 | p.13 |

**(2) 내 해석: 일반화 개선 가능성과 주의점**

- **긍정 요인**
  - 이미 정렬된 omni 백본을 쓰므로 모달리티 간 전이가 쉬울 수 있습니다.
  - 동종 샘플링은 학습 신호의 질을 높이고, 증류는 순위 정보를 전달합니다.
  - 탄력 임베딩은 배포 환경 변화에 대한 적응성입니다 (단, 이는 분포 일반화가 아니라 **자원 일반화**입니다).
- **주의점**
  - OOD 근거는 학습 데이터와 벤치마크의 도메인 겹침 가능성 때문에 단정하기 어렵습니다 (S7).
  - 모달리티 간 전이 주장(p.8)은 ablation이 없습니다.
  - 긴 입력, 다국어, 잡음 오디오, 새로운 모달리티 조합에서의 일반화는 미검증입니다.
  - 규모 확대(2B→9B)의 이득은 주로 비디오 시간 이해에서 나타났습니다 (p.21).
  - 단, 백본 세대가 달라 교란됩니다.
- **일반화 검증을 위해 제안하는 실험** (제 제안)
  1. 학습 코퍼스에서 특정 도메인·모달리티 조합을 통째로 제외(leave-one-domain-out)한 뒤 평가합니다.
  2. 의미 기반(임베딩 유사도) 중복 제거와 오염 감사를 수행합니다.
  3. 동일 데이터·동일 설정에서 native vs retrofit 초기화를 대조합니다.
  4. 프롬프트·지시문 교란에 대한 강건성과 다국어를 평가합니다.
  5. 교사·학생 불일치 사례의 오류 분석을 하고, 시드별 분산을 보고합니다.
  6. 비동기 혼합 입력(예: 오디오 + 텍스트 → 비디오)에 대한 전용 벤치마크로 평가합니다.

### 8-2. 2020년 이후 관련 연구 비교 분석

(비고: 2025~2026년 항목은 **이 논문의 서술에 근거**하며 원문을 직접 확인하지 못했습니다.)

| 연도 | 연구 | 핵심 아이디어 | Ovis-Embedding과의 관계 |
|---|---|---|---|
| 2020 | DPR (Karpukhin et al.) | 듀얼 인코더 대조학습으로 QA 검색 | 대조 학습 기본 틀 계승 (p.21) |
| 2021 | CLIP (Radford et al.) | 웹 규모 이미지–텍스트 듀얼 인코더 | 이미지–텍스트 중심의 한계. 본 논문은 단일 백본 통합 |
| 2022 | E5 (Wang et al.) | 약지도 대조 사전학습으로 범용 텍스트 임베딩 | 다단계 레시피 계승 (p.21) |
| 2022 | Matryoshka Representation Learning (Kusupati et al.) | 앞쪽 $d$차원만으로도 쓸 수 있는 중첩 임베딩 | Stage-4가 MRL 대신 사후 PCA + 어댑터 선택 (p.13) |
| 2023 | GTE, MTEB | 다단계 대조학습, 표준 텍스트 벤치마크 | 평가·학습 유형 구성에 활용 (p.8–9) |
| 2023 | SigLIP, CLAP | 시그모이드 손실 이미지-텍스트, 오디오-텍스트 대조 | 모달리티 쌍에 한정 (p.22) |
| 2024 | BGE-M3 | 다국어·다기능·다입도 텍스트 임베딩 | hard negative 채굴에 사용 (p.9) |
| 2024 | E5-Mistral, NV-Embed | LLM 디코더를 임베딩 백본으로 + 지시문 + last-token | 본 논문의 설계 계보 (p.21) |
| 2024 | VLM2Vec / MMEB, GME, MM-Embed | VLM을 임베딩으로 전환, 통합 멀티모달 검색 | 본 논문의 직접적 선행 (p.22) |
| 2024 | Matryoshka-Adaptor (Yoon et al.) | 비지도 유사도 보존 어댑터 | Eq. 19–21의 원형 (p.14) |
| 2025 | Qwen3-Embedding | 파운데이션 모델 기반 텍스트 임베딩·리랭킹 | Table 4의 텍스트 baseline (Qwen3-Embedding-4B) |
| 2025–26 | Omni-Embed-Nemotron, e5-omni, LCO-Embedding-Omni, jina-embeddings-v5, Qwen3-VL-Embedding | 오디오 추가형·정렬형 omni 임베딩 또는 VL 전문 임베딩 | 직접 경쟁군. "retrofit 또는 모달리티 희생"이라고 서술 (p.2, 22) |

**추가 주의 (확신 불가):** 논문은 LCO-Embedding-Omni를 "learned compression tokens"로 설명하지만(p.22), 제 기억으로는 해당 계열의 핵심이 언어 중심(language-centric) 정렬인 듯합니다. 확실하지 않으므로 원 논문 확인이 필요합니다. 일부 참고문헌(e5-omni, jina-v5, GME 등)의 서지 정보는 검증하지 못했습니다.

**이 논문이 향후 연구에 미칠 영향**
1. **"native omni 백본 + 최소 적응"이라는 단순한 설계가 강한 baseline이 될 수 있음을 시사합니다.** 오디오·비디오 임베딩 연구 진입 장벽이 낮아집니다.
2. **배치 구성(동종 샘플링)과 난이도 인지 손실 및 선택적 증류**는 모달리티를 넘어 재사용 가능한 학습 패턴입니다.
3. **압축을 모델 학습과 분리**(사후 PCA + 어댑터)하는 접근은 배포 중심 연구에 영향을 줄 수 있습니다. 한편 저차원에서 세밀 과제가 손실이 큰 점은 "용량 한계"라는 후속 연구 주제를 제시합니다.
4. 평가 측면에서는 MMEB-v3, MAEB, MVEB 같은 다중 벤치마크를 통합한 "omni 평가 툴킷" 공개 시 비교 표준화에 기여할 것입니다 (공개가 실제로 이루어진다는 조건에서).

**앞으로 연구 시 고려할 점**
- **통제 비교:** 동일 데이터·크기·예산에서 초기화 방식과 손실 구성요소를 분리한 ablation이 필요합니다.
- **통계적 엄밀성:** 시드 반복, 신뢰구간, 부트스트랩 기반 유의성 검정을 도입해야 합니다.
- **오염 방지:** 의미 기반 dedup과 보고서 수준의 오염 감사(오디오·비디오 포함)가 필요합니다.
- **교사 의존성과 합성 데이터:** 교사·VLM 라벨러의 편향이 증류와 데이터 구축을 통해 학생에게 전이되는지 점검해야 합니다.
- **효율 지표 보고:** 지연, 메모리, 인덱스 크기를 실측해야 합니다.
- **현실적 any-to-any 벤치마크:** 혼합 질의·다국어·긴 입력·실제 에이전트 시나리오가 필요합니다.
- **윤리·안전:** 웹 수집 데이터의 라이선스와 프라이버시, 오디오(음성) 데이터의 개인정보 문제를 다뤄야 합니다.
- **재현성:** 상용 API baseline은 버전 고정과 평가 스크립트 공개로 투명하게 비교해야 합니다.

---

## 출처 및 참고자료 (전체)

**1차 자료 (직접 분석)**
- Ovis-Embedding Team, *"Ovis-Embedding: Pushing the Frontiers of Universal Omni-Modal Embeddings"*, arXiv:2609.25165v1 [cs.AI], 2026 (제공된 PDF, 본문·표·그림 캡션).

**위 PDF의 참고문헌 중 본 답변에서 언급·활용한 항목 (제목만 인용, 원문은 직접 열람하지 않음)**
- Karpukhin et al., *Dense Passage Retrieval for Open-Domain Question Answering* (EMNLP 2020)
- Radford et al., *Learning Transferable Visual Models from Natural Language Supervision* (ICML 2021)
- Wang et al., *Text Embeddings by Weakly-Supervised Contrastive Pre-training* (2022)
- Kusupati et al., *Matryoshka Representation Learning* (NeurIPS 2022)
- Li et al., *Towards General Text Embeddings with Multi-Stage Contrastive Learning* (2023)
- Muennighoff et al., *MTEB: Massive Text Embedding Benchmark* (EACL 2023)
- Zhai et al., *Sigmoid Loss for Language Image Pre-training* (2023)
- Wu et al., *Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation* (2023)
- Chen et al., *BGE M3-Embedding* (2024)
- Wang et al., *E5-Mistral: Improving Text Embeddings with Large Language Models* (2024)
- Lee et al., *NV-Embed* (2024)
- Jiang et al., *VLM2Vec* (2024) 및 *VLM2Vec-V2* (2025)
- Zhang et al., *GME: Improving Universal Multimodal Retrieval by Multimodal LLMs* (2024) 및 *MM-Embed* (2024)
- Yoon et al., *Matryoshka-Adaptor: Unsupervised and Supervised Tuning for Smaller Embedding Dimensions* (EMNLP 2024)
- Lin et al., *Focal Loss for Dense Object Detection* (ICCV 2017)
- van den Oord et al., *Representation Learning with Contrastive Predictive Coding* (2018)
- Hinton et al., *Distilling the Knowledge in a Neural Network* (2015)
- Xu et al., *Qwen2.5-Omni Technical Report* (2025); Qwen Team, *Qwen3.5: Towards Native Multimodal Agents* (2026)
- Zhang et al., *Qwen3 Embedding* (2025); Li et al., *Qwen3-VL-Embedding and Qwen3-VL-Reranker* (2026)
- Xu et al., *Omni-Embed-Nemotron* (2025); Xiao et al., *Scaling Language-Centric Omnimodal Representation Learning*; Wang et al., *E5-Omni*; Günther et al., *jina-embeddings-v5*; Li et al., *Conan-embedding*
- Huang et al., *MMEB-v3* (2026); Assadi et al., *MAEB* 및 *MVEB* (2026); Liu et al., *Introducing RTEB* (Hugging Face Blog, 2025)
- Lu et al., *MultiConIR* (Findings of EMNLP 2025); Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond* (2009)

**일반 지식 (외부 자료를 조회하지 않고 설명에 사용)**
- LoRA (Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021)
- PCA와 nDCG 등 기본 개념

> 위 외부 문헌들은 **이 PDF의 인용 서술에만 근거**해 요약했습니다. 웹 사이트는 열람하지 않았고, 서지 정보의 정확성은 별도로 검증하지 못했습니다.
