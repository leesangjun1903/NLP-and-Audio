# LIGE-GR: A Smooth Leap from Ranking to Generative Recommendation in the LLM Era

---

## 1. Executive Summary (10문장)

1. LIGE-GR은 Meta가 제안한 프레임워크로, 기존 itemwise(아이템별 독립 점수화) 랭킹 시스템을 폐기하지 않고 listwise(리스트 전체를 함께 최적화하는) 생성형 추천으로 "일반화"합니다 (p.1).
2. 업그레이드는 세 부분입니다: ① 문맥인식(CA) 예측 모듈, ② itemwise 가치모델에서 listwise 가치모델(ListVM)로의 확장, ③ greedy 디코더에서 빔서치 기반 "RL 디코더(Palette)"로의 교체 (p.5, Fig. 2).
3. CA 모듈은 기존 랭킹 모델(CF)의 중간 표현 $v'_t$ 위에 얹은 4-layer·4-head causal Transformer이며, 기존 모델은 그대로 보존됩니다 (p.6–7, Fig. 3).
4. 설정만 되돌리면 기존 시스템이 정확히 복원되고, 요청별 지연 초과 시 itemwise로 폴백합니다 (p.5, p.10, Alg. 2).
5. 오프라인에서 CA는 CF보다 NE가 낮았습니다. Instagram Reels는 Continue 1.57%, Skip 0.74% 등이고, Facebook Video 17개 과제 중 15개가 개선되었습니다 (Table 1, p.11).
6. 온라인 A/B(b=1 기본 구성)에서 time spent가 Instagram Reels +1.14%, Facebook Video +0.72% 증가했습니다 (Table 2, p.12).
7. Instagram Reels에서 b=6 + $ListVM_{golden}$ + $\widehat F_\text{dur}$ 구성은 b=1 기준 대비 time spent +0.69%, 조회수 +1.82%를 추가로 얻었습니다 (Table 3, p.12).
8. 비용은 CF 연산의 약 10% 추가 자원이고(b=6은 약 20% 추정), Instagram Reels 지연은 약 7%, Facebook Video는 약 2.2% 증가했습니다 (p.3, p.10–11).
9. 한계로 저자들은 후보 풀이 수백 개 수준에 머문다는 점을 들며, Semantic ID와의 결합을 후속 과제로 제시합니다 (p.16).
10. **[내 해석]** 산업 규모 실증과 "점진적 업그레이드 경로"라는 관점이 강점입니다. 다만 외부 listwise/생성형 베이스라인과의 직접 비교, 구성요소별 ablation, 신뢰구간 보고는 부족합니다.

### 1-1. 연구의 목적과 필요성

**목적 (p.1–3).** 성숙한 산업 추천 시스템에 LLM 패러다임(순차 생성과 시퀀스 단위 최적화)을 위험·비용을 낮춰 이식하는 것입니다.

**필요성 (저자 주장).**
- 기존 시스템은 각 아이템을 독립적으로 점수화한 뒤 정렬합니다. 다양성과 무결성은 규칙 기반 휴리스틱으로 후처리하므로 리스트 전체를 학습된 방식으로 평가하지 못합니다 (p.1–2, p.4).
- LLM은 이전 토큰에 조건화해 시퀀스를 생성하고 품질을 시퀀스 전체로 판단합니다. 추천도 구조적으로 같은 문제입니다 (p.2).
- 완전 재구축(예: OneRec)은 두 가지 장벽에 부딪힙니다 (p.2–3).
  - **시스템**: 누적된 기준 성능을 먼저 회복해야 하고, 롤백이 어렵습니다.
  - **조직**: 검색·랭킹·가치모델·서빙 팀 경계를 흔듭니다.

> 📘 **용어 풀이**
> - **itemwise(=pointwise)**: 후보 하나하나를 서로 무관하게 점수화합니다.
> - **listwise**: 아이템 묶음(리스트)을 하나의 단위로 놓고 최적화합니다.
> - **LLM 패러다임**: 다음 토큰을 이전 문맥에 조건화해 한 개씩 생성합니다.
> - **Value Model(VM)**: 예측된 좋아요·공유·시청시간 등 여러 참여 신호를 하나의 스칼라 점수로 합치는 함수입니다.

---

## 2. 핵심 주장과 근거 (표)

| # | 핵심 주장 | 근거 | 위치 | 근거 강도 [내 평가] |
|---|---|---|---|---|
| C1 | itemwise는 독립 점수화라는 근본 한계가 있고, listwise(조합) 최적화가 필요하다 | 개념 도식 (실험 근거는 아님) | Fig. 1 (p.2), p.2 | 개념적 |
| C2 | 문맥인식(CA) 예측이 문맥무시(CF)보다 정확하다 | NE 상대 개선: IG Continue 1.57%, Skip 0.74%, Like 0.29%, Comment 0.21%, Share 0.30%, Watch completion 0.43%. FB는 Skip 0.46~Comment 1.59%. FB 17과제 중 15개 개선, 2개는 ≤0.07% 악화 | Table 1 (p.11), §5.1 | 중간 (분산·CI 없음) |
| C3 | LIGE-GR은 기존 시스템을 엄밀히 일반화하며, 설정만으로 복원·폴백 가능하다 | 구성 논증(CF 대체, $b=1$, $p_\text{continue}\equiv1$, $\widehat F=0$). b=1 재현 검증: 11,965개 재생 요청에서 평균·사분위 점수가 0.15% 이내로 일치 | p.5, p.8, p.10, Alg. 1–2, App. A (p.20) | 중~강 (오프라인 재생 기준) |
| C4 | b=1 기본 구성이 온라인 지표를 개선한다 | IG: time spent +1.14%†, 조회 +2.28%†, 좋아요 +2.65%†, 리셰어 +1.77%†. FB: time spent +0.72%†, 좋아요(반응) +1.59%†, **조회 −0.52%†** | Table 2 (p.12) | 중간 (단일 7일 A/B, 소수 트래픽) |
| C5 | b=6 + $ListVM_\text{golden}$ + $\widehat F_\text{dur}$가 b=1보다 추가 이득을 준다 | b=1 대비 time spent +0.69%†, 조회 +1.82%†, 좋아요 +2.93%†, 리셰어 +1.41%†. 빔만 넓힌 팔( $ListVM_\text{vanilla}$ , b=6)은 time spent +0.05%, 조회 +0.00% | Table 3 (p.12) | 중간 (두 변경이 묶여 효과 분리 불가) |
| C6 | 추가 추론 자원이 작다 | CF의 약 10%(b=1), b=6은 b=1의 약 2.1배로 약 20% "추정". 후보 풀을 약 1/3로 줄이면 처리량 60–80% 향상. 지연은 IG 약 7%, FB 약 2.2% 증가 | p.3, p.10–11, p.12–13 | 약~중 ("roughly", 정의 불일치) |
| C7 | 리스트 구성이 다양해지고 신선도만 약간 후퇴한다 | 토픽 엔트로피 +1.14~2.37%, 동일 토픽 연속 −4.03~−4.74%, 인접 영상 유사도 −3.68%, 72시간 미만 영상 −0.66% | Table 4 (p.13), Fig. 4 (p.14), §5.4 | 약~중 (3일, 활동적 사용자 편향) |
| C8 | 빔 폭 b=6은 이득 포화 구간이다 | 오프라인 누적 VM+CL 점수 이득이 b=2에서 3.16%, b=6에서 6.72%, b=8에서 7.42% | Fig. 5 (p.20), App. A | 약 ("포화"는 판단) |
| C9 | 별도 리랭킹 단계나 전면 교체 없이 G–E 이점을 얻는다 | Related Work의 서술적 비교. **실험적 직접 비교 없음** | p.15 | 서술적 |

> 📘 **용어 풀이**
> - **NE (Normalized Entropy)**: 로그손실(교차엔트로피)을 기준 엔트로피로 정규화한 예측 품질 지표입니다. 낮을수록 좋습니다. 논문은 정의를 직접 쓰지 않고 Liu et al. (2023a)를 인용하며, 위 설명은 일반적 정의입니다.
> - **A/B 테스트 † (p<0.001)**: 무작위 대조 실험에서 통계적으로 유의한 차이를 표시한 것입니다.
> - **리셰어 / VPV**: 공유 횟수 / 논문 표기상 영상 조회 수입니다.

---

## 2-1. 상세 설명

### (A) 해결하려는 문제

**문제 정의 (Eq. 1, p.4).** 요청마다 후보 집합 $\mathcal C$에서 길이 $T$(약 10)의 순서 있는 리스트를 뽑습니다.

$$V_T=[v_1,v_2,\dots,v_T],\quad v_t\in\mathcal C,\quad V_t=[v_1,\dots,v_t],\quad V_0=\emptyset$$

- $V_T$: 최종 추천 리스트
- $V_t$: 앞 $t$개를 고른 접두부(prefix)
- $T$: 리스트 길이 (약 10)
- $\mathcal C$: 후보 집합 (크기 약 $10^2$, p.11)
- 리스트 안의 아이템은 서로 달라야 합니다.

**기존 itemwise 시스템 (Eq. 2–5, p.4–5).**

$$\text{itemVM}(p)=w_1 p_\text{like}+w_2 p_\text{follow}+w_3 p_\text{share}+\cdots \tag{3}$$

$$\arg\max_{V_T}\sum_{t=1}^{T}\text{itemVM}\big(\text{CF}(u,v_t)\big) \tag{4}$$

$$\arg\max_{V_T}\sum_{t=1}^{T}\Big[\text{itemVM}\big(\text{CF}(u,v_t)\big)+\text{CL}(v_t\mid V_{t-1})\Big] \tag{5}$$

- $p_\text{like}, p_\text{follow},\dots$: 랭킹 모델이 예측한 참여 신호 확률
- $w_k$: 가치모델 가중치 (값은 비공개)
- $u$: 사용자 특징, $\text{CF}(u,v_t)$: 사용자와 아이템만 보는 문맥무시 예측기
- $\text{CL}(v_t\mid V_{t-1})\in\mathbb R\cup\{-\infty\}$: 컨트롤 레이어의 가산 보정. $-\infty$는 불가 후보를 마스킹합니다.

**컨트롤 레이어 예시 (p.4).**
- 하드 룰: $\text{CL}=-\infty$
- 간격 감점: $\text{CL}=-\exp(t'-t+1)\cdot\text{const}$ ($t'$는 직전 동일 카테고리 아이템 위치)
- DPP 다양성: $\text{CL}=\log\det(\Phi_t^\top\Phi_t)-\log\det(\Phi_{t-1}^\top\Phi_{t-1})$ ($\Phi_t=[\phi_t,\dots,\phi_1]$, $\|\phi\|=1$인 임베딩)

기존 디코더는 위치마다 $v_t=\arg\max_{c\in\mathcal C\setminus V_{t-1}}\text{itemVM}(\text{CF}(u,c))+\text{CL}(c\mid V_{t-1})$를 고르는 greedy입니다 (p.5).

> 📘 **용어 풀이**
> - **Greedy decoder**: 매 위치에서 현재 최고 점수 후보만 고르고 되돌아보지 않습니다.
> - **DPP(Determinantal Point Process)**: 임베딩 행렬식(부피)이 클수록, 즉 아이템들이 서로 다를수록 높은 점수를 주는 다양성 모델입니다.
> - **Control layer**: 다양성·무결성·비즈니스 제약을 점수 가감이나 마스킹으로 반영하는 규칙 계층입니다.

### (B) 제안 방법

**B-1. 문맥인식 목적함수 (Eq. 6, 7, p.5).**

$$\arg\max_{V_T}\sum_{t=1}^{T}\Big[\text{itemVM}\big(\text{CA}(u,v_t\mid V_{t-1})\big)+\text{CL}(v_t\mid V_{t-1})\Big] \tag{6}$$

$$\arg\max_{V_T}\ \text{ListVM}\Big(\{\text{CA}(u,v_t\mid V_{t-1})\}_{t=1}^{T}\Big) \tag{7}$$

- $\text{CA}(u,v_t\mid V_{t-1})$: 앞서 선택된 아이템 $V_{t-1}$을 조건으로 한 참여 신호 예측. 반복·포화·보완·다양성·피로 같은 리스트 효과를 학습하려는 목적입니다 (p.6).
- Eq. 7의 ListVM은 아이템 메타데이터와 컨트롤 레이어 보정도 쓸 수 있는 가장 일반적인 형태입니다.

**B-2. Listwise VM (Eq. 8, 9, p.7).**

$$\text{ListVM}_\text{vanilla}(V_T)=\sum_{t=1}^{T}\big[\text{itemVM}(\text{CA}(u,v_t\mid V_{t-1}))+\text{CL}(v_t\mid V_{t-1})\big] \tag{8}$$

$$\text{ListVM}_\text{golden}(V_T)=\sum_{t=1}^{T}p_\text{continue}(V_{t-1})\cdot\big[\text{itemVM}(\text{CA}(u,v_t\mid V_{t-1}))+\text{CL}(v_t\mid V_{t-1})\big] \tag{9}$$

$$p_\text{continue}(V_t)=p_\text{continue}(V_{t-1})\cdot \text{CA}_\text{continue}(u,v_t\mid V_{t-1}),\quad p_\text{continue}(\emptyset)=1$$

- $p_\text{continue}(V_{t-1})$: 사용자가 $V_{t-1}$을 소비하고 $v_t$에 도달할 누적 생존확률
- $\text{CA}_\text{continue}$: CA 모델이 예측하는 "해당 아이템 후 계속 볼 확률"
- $ListVM_\text{vanilla}$ 는 $p_\text{continue}\equiv1$인 특수형입니다 (p.7).

> 📘 **용어 풀이**
> - **누적 생존확률**: 사용자가 이탈하지 않고 해당 위치까지 도달할 확률입니다. 뒤쪽 아이템일수록 노출 가능성이 낮으니 가치를 그만큼 할인합니다.
> - **"golden"**: 저자들이 붙인 이름입니다. 의미 설명은 문서에 없습니다.

**B-3. Palette 디코더 (Eq. 10, Alg. 1, p.7–8).**

$$Q(V{+}c)=\text{ListVM}_\text{golden}(V{+}c)+\widehat F(V{+}c) \tag{10}$$

각 단계 $t=1,\dots,T$는 다음과 같이 진행됩니다 (Alg. 1).

1. 빔 $\mathcal B$의 각 접두부 $V$에 대해 가능한 후보 $c$를 붙여 확장 집합 $\mathcal E$를 만듭니다 ($\text{CL}(c\mid V)>-\infty$인 것만).
2. 누적 가치를 갱신합니다.

$$\text{ListVM}_\text{golden}(V{+}c)=\text{ListVM}_\text{golden}(V)+p_\text{continue}(V)\big[\text{itemVM}(\text{CA}(u,c\mid V))+\text{CL}(c\mid V)\big]$$

3. $p_\text{continue}(V{+}c)=p_\text{continue}(V)\cdot\text{CA}_\text{continue}(u,c\mid V)$로 갱신합니다.
4. $Q$ 상위 $b$개만 남깁니다 ($b$는 빔 폭).
5. 마지막에 $\arg\max_{V\in\mathcal B}\text{ListVM}_\text{golden}(V)$를 반환합니다.

기호는 다음과 같습니다.
- $V{+}c$: 접두부 $V$에 후보 $c$를 붙인 새 접두부
- $\widehat F$: 남은 위치에서 기대되는 가치(value-to-go)의 추정치
- RL 해석(p.8): 상태 = 접두부, 행동 = 다음 후보, 이미 쌓인 ListVM = 실현 보상, $\widehat F$ = 미래 가치 추정

**복원 조건 (p.8).** CF 사용, $\text{CA}_\text{continue}\equiv1$, $b=1$, $\widehat F=0$으로 두면 기존 greedy 디코더가 됩니다.

> 📘 **용어 풀이**
> - **빔서치(Beam search)**: 각 단계에서 상위 $b$개 부분해만 유지하며 탐색합니다. $b=1$이면 greedy입니다.
> - **value-to-go(미래 가치)**: 현재 상태에서 앞으로 얻을 것으로 기대되는 보상입니다.
> - **absorbing state(흡수 상태)**: 진입하면 빠져나오지 못하는 종료 상태입니다 (p.7 표현). 문서는 설명을 생략했고, 사용자 이탈로 보는 것은 **[내 해석]**입니다.
> - **[내 해석] Q**: A*의 $f=g+h$(지금까지 비용 + 휴리스틱 잔여)와 구조가 같습니다. 저자들도 학습된 가치망이 아니라 닫힌 형태의 추정을 쓴다고 밝힙니다 (p.8).

**B-4. 미래가치 추정 (Eq. 11–14, p.8–9).**

$$\bar s(V_t)=\frac1t\sum_{\tau=1}^{t}\big[\text{itemVM}(\text{CA}(u,v_\tau\mid V_{\tau-1}))+\text{CL}(v_\tau\mid V_{\tau-1})\big] \tag{11}$$

(원문은 $v_i$로 적혀 있어 $v_\tau$와 표기가 어긋납니다. 의도는 접두부의 위치당 평균 점수로 보입니다.)

$$\widehat F_\text{step}(V_t)=\bar s(V_t)\sum_{j=1}^{T-t}\text{CA}_\text{continue}(u,v_t\mid V_{t-1})^{\,j} \tag{12}$$

$$\widehat F_\text{dur}(V_t)=\bar s(V_t)\sum_{j=1}^{T-t}\text{CA}_\text{continue}(u,v_t\mid V_{t-1})^{\,j\,\bar d/d_t},\qquad \bar d=\frac{D(V_t)}{t} \tag{14}$$

(Eq. 13은 지수 $\bar d/d_t$를 설명하는 식입니다.)

- $\bar s$: 접두부의 위치당 평균 점수(연속확률 가중 전)
- $d_t$: 가장 최근 선택된 아이템의 길이(초)
- $D(V_t)$: 선택된 $t$개 아이템의 총 길이
- $\bar d$: 평균 길이
- 합의 상한 $T-t$: 남은 위치 수. $t=T$이면 합이 비어 $\widehat F=0$입니다.

**동기 (p.9).** $\widehat F_\text{step}$은 현재 아이템의 연속확률을 모든 미래 위치에 반복 적용하므로, 긴 영상으로 끝나는 접두부가 빔에서 불리하게 탈락합니다. $\widehat F_\text{dur}$는 지수 $\bar d/d_t$로 평균 길이 기준으로 재스케일해 이 편향을 줄입니다.

> 📘 **용어 풀이**
> - **지수 $\bar d/d_t$ 재스케일**: 길이당 이탈 위험이 일정하다는 가정 아래 생존확률이 $q^{\text{길이}}$ 꼴로 변한다는 사고입니다. 논문은 "same per-second exit propensity"라고 서술합니다 (p.9). 이 가정이 실제로 맞는지는 검증되지 않았습니다.
> - **Monte Carlo Tree Search / learned value model**: 더 정교한 대안으로 언급만 되고 후속 과제로 남겨졌습니다 (p.8).

### (C) 모델 구조 (Fig. 2, Fig. 3, Alg. 2)

| 구성 | 내용 | 위치 |
|---|---|---|
| CF 모듈 (기존 랭커) | 사용자 $u$와 후보 $v_t$를 Interaction NN으로 결합, 과제별 head(p_like, p_follow 등)로 예측. 중간 표현 $v'_t$ 생성. 요청당 1회, 후보 순서와 무관하게 계산·캐시 | Fig. 3 좌, p.6, p.9 |
| CA 모듈 | $v'_1,\dots,v'_T$를 입력으로 하는 GPT식 decoder-only causal Transformer (4 layers, 4 heads). 이전 위치만 참조하며 과제별 head가 예측을 재정제. 모든 head는 위치 간 가중치 공유 | Fig. 3 우, p.6–7 |
| 서빙 Phase 1 | 기존 랭킹과 동일(추가 자원 없음) | p.9, Alg. 2 |
| 서빙 Phase 2 | Palette가 $T$번의 배치 경량 패스 수행. 빔을 넓혀도 순차 패스 수는 불변, 배치만 증가 | p.9 |
| 후보 풀 축소 | CF 점수 상위 약 1/3만 CA에 전달. 처리량 60–80% 향상 (두 세대 GPU 벤치마크) | p.10 |
| 폴백 | 지연 예산 $\tau$ 초과 또는 CA 호출 실패 시 요청 단위로 itemwise 디코더로 복귀 | Alg. 2, p.10 |

> 📘 **용어 풀이**
> - **Causal Transformer**: 어텐션 마스크로 이전(과거) 위치만 보게 한 자기회귀형 Transformer입니다.
> - **Interaction NN / task head**: 사용자-아이템 특징 상호작용 네트워크 / 각 참여 신호별 출력 층입니다.
> - **[내 해석]** Fig. 3의 어텐션 행렬은 하삼각(대각 포함)처럼 보입니다. 자기 자신의 $v'_t$와 이전 아이템을 문맥으로 쓰는 구조로 읽힙니다.

### (D) 성능 향상 (저자 보고)

- **예측 품질 (Table 1).** CA는 IG Reels 6개, FB Video 5개 표시 과제 모두에서 NE가 낮았습니다.
- **온라인 b=1 (Table 2).** IG Reels는 Sessions +0.11%†, DAU +0.05%†, Time spent +1.14%†입니다. FB Video는 Sessions +0.07%, DAU +0.01%(둘 다 † 없음), Time spent +0.72%†입니다.
- **b=6 duration-aware (Table 3).** 빔만 넓히면 좋아요 +0.74%†, 리셰어 +1.21%†뿐이고 time spent는 +0.05%입니다. golden+ $\widehat F_\text{dur}$를 더하면 time spent +0.69%†, 조회 +1.82%†가 됩니다. 저자들은 "빔 확대만으로는 오프라인 점수 개선이 광범위한 소비 개선으로 전환되지 않는다"고 서술합니다 (p.20).
- **생태계 (Table 4).** 토픽 다양성은 늘고 반복은 줄었으며 신선도는 후퇴했습니다.

### (E) 한계

**저자가 명시한 한계 (p.16, p.8, p.7).**
- 현재 후보 풀은 수백 개 수준이어서, Semantic ID 등과 결합해야 대규모 후보로 확장됩니다.
- 이 작업은 최종형이 아니라 "여지가 있는 전환 프레임워크"입니다.
- 아키텍처는 핵심이 아니며, 학습된 가치모델과 MCTS는 미구현입니다.
- 신선도 소폭 하락(−0.66%)과 탐색 비용으로 인한 친숙 콘텐츠 감소를 트레이드오프로 인정합니다 (p.13).

**[내 해석] 추가 한계.** 5절(통계 취약점)과 6절(미답 질문)을 참고하세요.

---

## 4. 저자 보고 vs. 내 해석 (분리)

| 구분 | 저자가 직접 보고 (위치) | 내 해석 |
|---|---|---|
| **연구 주제** | itemwise에서 listwise·생성형으로 가는 "업그레이드 경로"를 제시한다. 완전 대체가 아니라 가산적·되돌릴 수 있는 일반화다 (p.1–3, p.14) | 학술적 새로움은 개별 알고리즘보다 "성숙 시스템 이관 전략과 엄밀한 일반화 구조"에 있다. 개별 요소(문맥 리랭킹, 빔서치, 생존 가중)는 기존 아이디어의 조합으로 보인다 |
| **방법** | CA(causal Transformer) + $ListVM_\text{golden}$ (Eq. 9) + Palette(Eq. 10, 14) (p.5–9) | Palette는 RL이라기보다 휴리스틱 lookahead를 가진 빔서치에 가깝다 (학습되는 정책·가치망 없음). 저자도 학습 없는 닫힌 추정이라고 밝힌다 (p.8). 생존확률 가중은 "도달하지 못할 아이템에 가치 부여 안 함"이라는 합리적 보정이지만, $p_\text{continue}$ 예측의 보정(calibration)이 정확해야 효과가 난다 |
| **방법 (설계 의도)** | duration-aware 추정이 긴 영상으로 끝나는 접두부의 부당한 탈락을 완화한다 (p.9) | 조회 수(+1.82%)가 time spent(+0.69%)보다 크게 늘어난 점은 영상 길이 분포 변화와 관련될 수 있다. 문서에 길이 변화 수치는 없으므로 **[가설]**이다 |
| **결과 (예측)** | CA가 모든 표시 과제에서 NE 개선 (Table 1) | 크기는 0.2~1.6%로 작다. 큰 이득은 Continue(1.57%)이고, 이는 $p_\text{continue}$ 설계와 일관된다. 다만 NE 개선이 곧 순위 품질 개선은 아니다 |
| **결과 (온라인)** | b=1에서 IG +1.14%, FB +0.72% time spent. b=6 golden은 추가 +0.69% (Table 2–3) | b=6 구성의 incumbent 대비 총효과는 문서에 없다 (기준선이 b=1 구성). Table 3의 +0.69%를 Table 2에 단순 합산하면 안 된다 |
| **결과 (빔)** | 빔 확대만으로는 time spent 개선이 없다 (p.20) | 오프라인 VM 점수가 +6.72%(Fig. 5)인데도 온라인 이득이 거의 없는 것은 **모델 예측 오차를 옵티마이저가 활용할 가능성(Goodhart/optimizer's curse)**과 일치한다 **[가설]**. 생존 가중과 길이 보정이 이를 완화한 것일 수 있으나, 분리 실험이 없어 단정할 수 없다 |
| **결과 (생태계)** | 다양성 증가, 신선도 감소 (Table 4) | "개선"(녹색) 표시는 가치판단이다. 저자도 탐색 트레이드오프라고 단서를 단다 (p.13). 사용자 반응과 연결되지 않은 구성 진단이다 |

---

## 5. 통계적으로 취약한 부분과 비교 불가 수치

### 5-A. 통계적 취약점

| 대상 | 문제 | 위치 |
|---|---|---|
| Table 2 (A/B) | 단일 7일 실험이고, 신뢰구간·표준오차 없이 † 표시만 있다. IG 1.5%, FB 약 2% 트래픽이다. 다중 지표 비교(6개)에 대한 다중검정 보정은 언급이 없다 | p.12 |
| Table 2 FB | Sessions, DAU, Reshares(+0.99%)에는 †가 없다 (유의성 불명). **조회 −0.52%†는 유의한 감소**인데 "broadly improves"로 요약된다 | p.12 |
| Table 3 | 세 구성이 각각 독립 A/B이다. $ListVM_\text{golden}$ 과 $\widehat F_\text{dur}$ 효과가 묶여 있어 분리 불가하고, golden b=6과 vanilla b=6의 직접 비교도 없다. b=6 vanilla의 time spent +0.05%, 조회 +0.00%는 † 없음 | p.12 |
| 장기 효과 | 7일 읽기만 있어 신기성 효과와 장기 리텐션을 알 수 없다 | p.12 |
| Table 1 (NE) | 반복 실행·분산·CI가 없다. IG는 "한 번의 온라인 학습 실행 누적값"이다. 0.21~0.30% 수준 차이의 신뢰성은 판단할 수 없다 | p.11 |
| Table 4 | 3일, 13,197 요청, 알려진 사용자 7,110명이다. 1,498 요청은 ID 없이 각각 1명으로 센 8,608명(7,110+1,498)이고, 활동적 사용자에 편향된 샘플이다. "95% 유의"만 있고 구간·사전분포·모형 상세가 없다. 13개 지표의 다중비교 보정도 불명확하다 | p.13 |
| Table 4 해석 | 반사실 로깅이라 사용자가 실제로 본 것은 한 리스트뿐이다. 두 리스트에 대한 반응은 관측되지 않았다 | p.13 |
| Fig. 4 | 단일 "대표" 사례이고 썸네일은 AI 생성 대체 이미지다. 일화적 근거다 | p.14 |
| Fig. 5 | 5,815건 "이상치 제거"의 기준이 없다. 오프라인 점수는 디코더가 직접 최적화하는 지표라 순환적이다. 오차막대가 없다. "포화"라 했지만 값은 계속 증가한다 (b=6 6.72 → 7: 7.10 → 8: 7.42) | p.20 |
| 외부 비교 | PRM, PEAR, NAR4Rec, GFN4Rec, OneRec 등과의 실험 비교가 전혀 없다. 우월성은 서술적 주장이다 | p.15 |

### 5-B. 비교 불가 수치

| 수치 | 이유 |
|---|---|
| IG Reels vs. FB Video 효과 크기 | 기준 시스템, 트래픽 비율(1.5% vs 약 2%), 지표 정의가 다르다. 직접 비교 불가 |
| 지연 +7%(IG) vs +2.2%(FB) | 저자 스스로 "정의와 기준이 달라 직접 비교 불가"라 명시 (p.11) |
| Table 1의 IG vs FB 열 | IG는 온라인 학습 누적, FB는 CA 대 CF 평가로 방식이 다르다 |
| Table 2 vs Table 3 | 기준선이 다르다 (Table 2는 incumbent, Table 3은 b=1 구성). 합산 불가 |
| Fig. 5 (%, VM 점수) vs Table 3 (%, 온라인 지표) | 서로 다른 단위의 지표다 |
| 추론 자원 "약 20%" (b=6) | 2.1배 × 10%로 계산한 "derived estimate"이고 실측이 아니다. b=6의 지연은 "not available" (p.12–13) |
| Table 4 범위 ("+1.14% to +2.37%") | 여러 분류체계에 걸친 범위이며 단일 효과가 아니다 |
| 초록의 +1.14%, +0.72% | b=1 기본 구성의 결과이며, 가장 정교한 구성(b=6 golden)의 incumbent 대비 총효과가 아니다 |

---

## 6. 문서가 답하지 않는 질문

1. CA는 어떻게 학습되는가? 손실, 라벨, 학습 시 리스트의 순서 출처(incumbent가 만든 로그인가), teacher forcing 여부가 문서에 없습니다.
2. 학습 시 접두부 분포(incumbent 로그)와 디코딩 시 접두부 분포(빔서치 탐색)가 다를 때 성능이 유지되는가 (분포 이동)?
3. $\text{CA}_\text{continue}$의 정의, 라벨(어떤 이벤트를 "계속"으로 보는가), 보정 정확도는?
4. itemVM 가중치 $w_k$와 가치모델 설계 방식, 가중치 민감도는?
5. $ListVM_\text{golden}$과 $\widehat F_\text{dur}$ 각각의 기여는? $\widehat F_\text{step}$ 대 $\widehat F_\text{dur}$의 온라인 비교는?
6. b=6 golden 구성의 incumbent 대비 총효과와 지연은?
7. 영상 길이 분포와 시청 완료 구조는 어떻게 변했는가 (조회 수 +2.28%와 time spent +1.14%의 차이 해석)?
8. FB Video 조회 −0.52%†의 원인과 사용자 경험 영향은? FB에서 b=6 실험은 왜 없는가?
9. 후보 풀 1/3 축소의 품질 영향(풀 크기 스윕)은? 리스트 길이 $T$가 커져도 성립하는가?
10. 후보 후속 평가의 구현: KV 캐시 사용 여부, 배치 구성은?
11. 신선도 하락의 장기 영향과 크리에이터 생태계 영향은? Table 4의 "familiar-creator 노출 −6.23%"의 사용자 만족 영향은?
12. 반복 실험 분산, 신뢰구간, 다중검정 처리는?
13. 데이터셋·코드·모델의 공개 여부와 재현성은? (문서에 없음. 사내 시스템으로 보임)
14. 다른 도메인(광고·검색·피드·이커머스)으로의 전이는?
15. 개인정보·공정성(크리에이터 규모별 노출) 평가의 범위는?

---

## 7. 가장 중요한 그림 5개 (논문의 그림은 총 5개)

> 📘 **용어 풀이 (공통)**
> - **Lattice(격자)**: 위치×후보 조합의 탐색 공간 그림입니다.
> - **Counterfactual logging(반사실 로깅)**: 같은 요청에 대해 기준 리스트와 신규 리스트를 동시에 기록해 사후 비교하는 방식입니다.

**① Fig. 2 (p.3): 세 구성요소 업그레이드 총괄도**
- 세 행: 랭킹 모델(CF에서 CA로), 목적함수(합산에서 ListVM으로), 디코더(greedy 단일 경로에서 RL 디코더의 다중 경로 탐색).
- 해석: 논문 전체의 구조도입니다. 좌측 열이 기존 시스템이고 우측 열이 확장이라 "대체가 아닌 일반화" 주장이 시각화됩니다.
- 우측의 점선 대안 경로는 greedy가 놓치는 조합을 빔이 발견한다는 뜻입니다.
- **[내 해석]** 각 구성요소의 개별 기여는 이 그림이 보여 주지 않으며, ablation이 필요합니다.

**② Fig. 3 (p.6): 리스트와이즈 모델 구조**
- 좌측 CF가 각 후보 $v_t$와 $u$로 $v'_t$를 만들고, 우측 4층 causal Transformer가 $v'_t$ 시퀀스를 입력받아 과제별 예측을 정제합니다.
- 해석: 비용이 작은 이유(Phase 1 재사용, 경량 CA)와 롤백이 쉬운 이유(CF 경로 보존)를 보여 줍니다.
- **[내 해석]** CA의 입력이 CF의 $v'$뿐이라, 원 입력 특징이 CA에서 직접 쓰이지 않습니다. CA의 표현력은 CF 표현의 정보량에 상한이 있습니다.

**③ Fig. 5 (p.20): 빔 폭별 오프라인 점수 이득**
- b=2~8에서 이득은 3.16, 4.72, 5.63, 6.27, **6.72**, 7.10, 7.42%입니다.
- 증분은 1.56, 0.91, 0.64, 0.45, 0.38, 0.32%포인트로 줄어듭니다.
- 해석: 수확체감은 분명하지만 plateau는 아닙니다. b=6은 비용 대비 선택의 "무릎"이라는 실용적 판단입니다.
- **[내 해석]** 이 그래프는 $p_\text{continue}\equiv1$, $\widehat F=0$의 단순 목적으로 그렸고, 실제 배포 목적(golden+ $\widehat F_\text{dur}$ )에는 적용되지 않았습니다. 오프라인 이득이 온라인 time spent로 이어지지 않은 결과(Table 3)와 함께 읽어야 합니다.

**④ Fig. 4 (p.14): 반사실 로그의 대표 요청 사례**
- 베이스라인 7개와 LIGE-GR 7개를 비교합니다. 새 토픽(Sports)이 도입되고, 27.5초 영상이 2위에서 4위로 이동하며, 70일 된 영상이 마지막으로 강등되고, Internet Culture 항목 하나가 제거됩니다.
- 해석: Table 4의 통계적 효과(다양성, 반복 감소, 신선도 변화)를 구체적으로 보여 줍니다.
- **[내 해석]** 이 사례는 단일 요청이고 AI 생성 이미지이므로 증거가 아니라 설명 도구입니다. 같은 사례에서 Comedy & Humor 항목이 6, 7위에 연속 배치된 것도 보입니다. 모든 반복이 사라지는 것은 아니라는 뜻입니다.

**⑤ Fig. 1 (p.2): itemwise 대 listwise 개념도**
- 5개 독립 결정 대 하나의 결합 결정을 대비합니다.
- 해석: 문제 설정을 직관적으로 보여 주지만 정량 정보는 없습니다. 이 논문의 정량 핵심은 Table 3입니다.

---

## 8. 결론

### 8-0. 저자가 제시한 시사점과 후속 계획 (p.16)

**시사점.**
- 기존 인프라, 모델, 조직 소유권을 유지한 채 점진적으로 listwise 생성형으로 전환할 수 있습니다.
- 롤백이 가능하고 비용이 낮아 산업 환경(지연, 신뢰성, 팀 소유권)에 현실적입니다.

**후속 계획.**
- 더 표현력 있는 아키텍처
- 아이템별 시스템에서는 정의하기 어려운 풍부한 리스트 수준 신호를 목적함수에 반영
- 더 발전된 디코딩과 강화학습
- Semantic ID로 대규모 후보 지원
- 스트리밍 추론: 첫 결과를 CA 패스 이전에 즉시 반환
- 컴퓨팅 공급·수요에 따른 동적 복잡도 조절
- (p.8) 학습된 가치모델, MCTS

### 8-0'. 내가 제안하는 추가 후속 연구 방향 **[내 제안]**

1. **ablation 설계**: $ListVM_\text{golden}$과 $\widehat F_\text{dur}$를 분리하고, $\widehat F_\text{step}$, 학습된 $\widehat F$와 비교합니다.
2. **학습된 가치 추정**: 오프라인 로그로 $\widehat F$를 학습하고, 분포 이동을 줄이는 off-policy 보정을 도입합니다.
3. **리스트 수준 학습 목표**: 리스트 보상으로 CA를 직접 미세조정하고, 빔 탐색으로 생성된 접두부로 데이터를 보강합니다.
4. **$p_\text{continue}$ 보정**: calibration 평가와 위험함수(hazard) 모델링(영상 길이 가정 검증)을 수행합니다.
5. **강건성**: 예측 불확실성을 반영한 보수적 디코딩(점수에서 불확실도 차감)으로 옵티마이저 편향을 억제합니다.
6. **평가 체계**: 외부 베이스라인(리랭커·G–E 방식)과 동일 조건으로 비교하고, 장기(≥4주) 리텐션과 크리에이터 지표를 측정합니다.
7. **서빙**: KV 캐시, 동적 빔 폭, 스트리밍 출력을 실험합니다.
8. **확장**: Semantic ID 기반 대규모 후보와 결합해 CF가 하던 pre-filter 역할을 대체합니다. 이 경우 후보 풀 1/3 축소 가정이 유지되는지도 점검해야 합니다.

### 8-1. 모델 일반화 성능 향상 가능성 (중점)

**(1) 논문이 실제로 보여 준 것**
- 두 가지 표면(IG Reels, FB Video)에서 CA의 NE 개선이 일관됩니다 (Table 1). FB 17개 과제 중 15개가 개선되고 2개는 ≤0.07% 악화입니다 (p.11).
- 두 표면에서 time spent가 유의하게 증가했습니다 (Table 2).
- CF 모델 자체는 보존되므로, 기존 모델이 가진 일반화 능력이 CA의 기반으로 유지됩니다 (p.7).
- 저자는 "목적 가중치나 VM 형태를 바꿔 생성을 제어할 수 있다"고 서술합니다 (p.15). 즉 CA 재학습 없이 목적을 바꾸는 유연성입니다.

**(2) 논문이 보여 주지 않은 것 (일반화의 빈틈)**
- 다른 도메인, 표면 간 전이, 후보 풀 크기와 리스트 길이 변화, 시간적 분포 이동, 사용자 세그먼트별(신규·저활동 사용자) 성능은 보고되지 않았습니다.
- 두 표면 모두 Meta의 숏폼 비디오이며, FB 조회 −0.52%†처럼 표면별 이질성이 존재합니다.
- Table 4는 활동적 사용자에 편향되어 있습니다 (p.13).
- 오프라인 지표는 NE와 자체 목적 점수뿐이며, 보류 데이터 일반화 평가는 문서에 없습니다.

**(3) 일반화를 높일 수 있는 메커니즘 [가설]**
- CA가 반복·포화·피로 같은 구조적 리스트 효과를 학습하면, 아이템 ID에 의존하지 않는 일반적 패턴이 되어 새 후보에도 적용될 수 있습니다.
- 생존 가중(Eq. 9)은 노출되지 않을 위치의 오차를 줄여 목적함수를 현실과 가깝게 합니다.
- CF 표현 재사용과 경량 CA 구조는 학습 효율 면에서 유리할 수 있습니다. 다만 과적합 감소는 검증되지 않았습니다.

**(4) 일반화를 해칠 수 있는 위험 [가설]**
- **학습-서빙 불일치(exposure bias류)**: CA가 incumbent 리스트로 학습되었다면, 빔이 만든 새로운 접두부는 학습 분포 밖일 수 있습니다.
- **옵티마이저 편향**: 넓은 빔이 예측 오차가 큰 조합을 찾아낼 수 있습니다. 오프라인 +6.72%가 온라인 time spent +0.05%로 이어지지 않은 것(p.20, Table 3)과 일관됩니다. 다만 인과적 입증은 아닙니다.
- **피드백 루프**: 새 디코더가 만든 로그로 재학습하면 탐색 분포가 다시 바뀝니다.
- **풀 축소 의존**: 성능이 CF의 pre-filter 품질에 의존합니다.

**(5) 제안하는 검증 실험 [내 제안]**
- 표면 간 교차 평가(IG에서 학습한 CA를 FB에 적용)와 시간적 보류 평가
- 후보 풀 크기, 리스트 길이, 빔 폭의 스윕
- 사용자 활동 수준별 층화 분석
- 의도적 OOD 접두부에 대한 CA 보정(calibration) 검사
- 불확실성 반영 디코딩 대 기본 디코딩 비교

### 8-2. 2020년 이후 관련 연구 비교 분석

> 아래는 **이 논문의 Related Work(p.14–15)와 참고문헌에 근거**한 비교입니다. 내용은 해당 논문의 서술을 요약한 것으로, 원 논문은 직접 확인하지 않았습니다. 이 논문 안에서 아래 방법들과의 실험적 직접 비교는 **없습니다**.

| 연구 (연도) | 계열 | 이 논문이 기술한 핵심 | LIGE-GR과의 차이 (저자 주장) |
|---|---|---|---|
| SetRank (Pang et al., 2020) | 리랭킹 | 후보 집합에 양방향 self-attention을 한 번에 적용해 상호 영향을 포착 | LIGE-GR은 별도 리랭킹 단계가 아니라 기존 랭킹 단계 안에서 causal attention으로 접두부에 조건화 |
| PURS (Pan et al., 2020), Sliding Spectrum Decomposition (Wang et al., 2021a) 등 | 다양성 제어 | DPP 등을 컨트롤 레이어 접근으로 인용 (p.4) | LIGE-GR은 CL을 보존하되, CA와 ListVM으로 학습 기반 listwise 최적화를 추가 |
| Edge-cloud polarized reranking (Gong et al., 2021) | 간격 감점 | gap demotion 규칙으로 인용 (p.4) | 규칙 기반 휴리스틱을 CL로 유지하면서 학습 기반 문맥 예측으로 보완 |
| Pivot-CVAE (Liu et al., 2021) | 생성형 슬레이트 | List-CVAE를 개선해 리스트 변이 보장, 과집중 완화 | LIGE-GR은 명시적 목적함수 기반 자기회귀 |
| PEAR (Li et al., 2022), MIR (Xi et al., 2022) | 리랭킹 | 개인화 문맥 Transformer / 후보 집합과 사용자 이력 공동 모델링 | 동일하게 별도 리랭킹 머신 없이 랭킹 단계 내 구현을 주장 |
| P5 (Geng et al., 2022) | LLM식 통합 | 다양한 추천 과제를 텍스트-투-텍스트로 통합 | 아이템 토큰화가 아닌 기존 랭커 확장 |
| TIGER (Rajput et al., 2023) | 생성형 검색 | 멀티모달 Semantic ID를 자기회귀로 예측 | 검색 단계 대체 vs 랭킹 단계 업그레이드. 결합은 후속 과제 (p.16) |
| GFN4Rec (Liu et al., 2023b) | 생성형 슬레이트 | GFlowNet으로 리스트 보상에 비례하는 확률로 시퀀스 샘플링. 자기회귀이나 이전 선택 아이템에 대한 직접 어텐션 없음 | LIGE-GR은 이전 선택 전체에 full causal attention |
| PIER (Shi et al., 2023) | 리랭킹 | 후보 순열 중 end-to-end 선택 | LIGE-GR은 부분 리스트를 평가하고 가지치기하는 교차형 생성-평가 |
| TransAct (Xia et al., 2023) | 아이템별 + 시퀀스 | 실시간 사용자 행동 모델 (후보를 독립 점수화) | 사용자 이력 모델링이 아니라 후보 간 문맥 모델링 |
| NAR4Rec (Ren et al., 2024) | G–E | 여러 슬레이트를 만들고 슬레이트 평가자로 최선 선택 | 생성 후 평가(generate-then-evaluate)가 아니라 부분 리스트 평가 |
| HSTU (Zhai et al., 2024) | 산업 생성형 | 조 단위 파라미터 순차 트랜스듀서 | 후보 간 문맥(list-level)을 직접 다루는 방식이 아님 (저자 서술: 아이템별 점수화 계열) |
| OneRec (Deng et al., 2025) | 전면 교체 | 생성형 검색과 반복 선호 정렬로 전체 퍼널을 하나의 모델로 대체 | LIGE-GR은 퍼널 교체가 아닌 기존 랭킹의 in-place 업그레이드 |
| GReF (Lin et al., 2025), HiGR (Pang et al., 2025) | 생성형 리랭킹 | 순서형 멀티 토큰 예측 + 선호 기반 학습 / 계층적 계획 + 다목적 선호 정렬 | LIGE-GR은 명시적 가치함수 목적이라 가중치·VM 형태로 제어 |
| Prompt-to-Slate (Tomasi et al., 2025) | 확산 | 슬레이트를 병렬 디노이징으로 구성 (순차 지연 감소) | 단계별 제약을 강제하기 어렵다고 저자가 평가 |
| Streaming VQ (Bin et al., 2025) | 인덱싱 | VQ-VAE 기반 실시간 갱신 인덱스 | 후보 규모 확장에서 보완 관계 가능 |

**8-2-1. 이 논문이 이후 연구에 미칠 영향 [내 해석]**
- **"이관 경로" 설계가 연구 주제가 된다**: 정확도뿐 아니라 롤백 가능성, 엄밀한 일반화, 폴백이 설계 요구사항으로 부각됩니다.
- **생성-평가 통합 패턴의 확산**: 부분 리스트를 평가하고 가지치기하는 방식은 다른 서빙 제약 환경에도 응용될 수 있습니다.
- **평가 관행**: 반사실 로깅 기반 구성 진단(Table 4)은 listwise 효과를 측정하는 참조 틀이 될 수 있습니다. 다만 5절에서 지적한 통계적 엄밀성 개선이 필요합니다.
- **한계**: 사내 시스템이라 재현이 어렵고 외부 베이스라인 비교가 없어, 학계 후속 연구가 이 결과를 기준선으로 쓰기는 제한적입니다.

**8-2-2. 앞으로 연구할 때 고려할 점 [내 제안]**
1. **공정한 비교**: 동일 후보 풀·동일 VM에서 리랭커, G–E, 확산, 생성형 모델과 맞비교합니다.
2. **효과 분리**: 문맥 예측(CA), 목적(ListVM), 탐색(빔)의 기여를 단계별 ablation으로 나눕니다.
3. **측정 엄밀성**: 신뢰구간, 다중검정 보정, 장기 효과, 이질성(사용자·크리에이터) 분석을 포함합니다.
4. **목적 정합성**: 오프라인 점수와 온라인 성과의 괴리를 줄이는 평가 지표, 보수적 디코딩, off-policy 평가를 설계합니다.
5. **생태계 책임**: 신선도, 크리에이터 노출 편중, 탐색 비용 등 가드레일을 목적에 반영합니다.
6. **확장성**: 대규모 후보(Semantic ID)와의 결합 시 pre-filter 가정과 지연 예산을 재검증합니다.
7. **최신 문헌 보강**: 이 보고서는 논문 인용 목록 범위만 다루었으므로, 별도 문헌 조사로 그 외 2020년 이후 연구를 보완해야 합니다.

---

## 참고자료 (출처)

**1차 자료 (직접 분석한 문서)**
- Srinivas, V. et al., *LIGE-GR: A Smooth Leap from Ranking to Generative Recommendation in the LLM Era*, Meta Platforms, arXiv:2609.18148v2 (첨부 PDF, 20 Sep 2026 표기)

**위 논문의 참고문헌 중 본 보고서에서 언급한 것 (제목은 논문 참고문헌 기준, 내용은 이 논문의 서술에 의존)**
- Pang et al., *SetRank: Learning a permutation-invariant ranking model for information retrieval* (SIGIR 2020)
- Pan et al., *PURS: Personalized unexpected recommender system for improving user satisfaction* (RecSys 2020)
- Wang et al., *Sliding spectrum decomposition for diversified recommendation* (KDD 2021)
- Gong et al., *Edge-cloud polarized reranking system for web-scale video recommendation* (SIGIR 2021)
- Liu et al., *Variation control and evaluation for generative slate recommendations* (WebConf 2021)
- Li et al., *PEAR: Personalized re-ranking with contextualized transformer for recommendation* (2022)
- Xi et al., *Multi-level interaction reranking with user behavior history* (SIGIR 2022)
- Geng et al., *Recommendation as language processing (RLP): A unified pretrain, personalize, prompt and predict paradigm (P5)* (RecSys 2022)
- Rajput et al., *Recommender systems with generative retrieval* (NeurIPS 2023)
- Liu et al., *Generative flow network for listwise recommendation* (KDD 2023)
- Shi et al., *PIER: Permutation-level interest-based end-to-end re-ranking framework in e-commerce* (KDD 2023)
- Xia et al., *TransAct: Transformer-based realtime user action model for recommendation at Pinterest* (KDD 2023)
- Liu et al., *Learning to rank normalized entropy curves with differentiable window transformation* (arXiv 2023, NE 지표 인용)
- Ren et al., *Non-autoregressive generative models for reranking recommendation* (KDD 2024)
- Zhai et al., *Actions speak louder than words: trillion-parameter sequential transducers for generative recommendations* (ICML 2024)
- Deng et al., *OneRec: Unifying retrieve and rank with generative recommender and iterative preference alignment* (arXiv 2025)
- Lin et al., *GReF: A unified generative framework for efficient reranking via ordered multi-token prediction* (CIKM 2025)
- Pang et al., *HiGR: Efficient generative slate recommendation via hierarchical planning and multi-objective preference alignment* (arXiv 2025)
- Tomasi et al., *Prompt-to-slate: Diffusion models for prompt-conditioned slate generation* (RecSys 2025)
- Bin et al., *Real-time indexing for large-scale recommendation by streaming vector quantization retriever* (KDD 2025)

웹 사이트는 참조하지 않았습니다. 마지막으로, 2020년 이후 비교 표의 각 방법 설명은 해당 논문들의 원문이 아니라 LIGE-GR 논문의 서술을 요약한 것이므로, 인용 시에는 원 논문을 확인하시기 바랍니다.
