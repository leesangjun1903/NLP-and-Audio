# SCRIBES: Web-Scale Script-Based Semi-Structured Data Extraction with Reinforcement Learning

**참고 범위**
- 이 보고서는 제공하신 PDF(arXiv:2510.01832v2)만 근거로 합니다. 외부 사이트는 검색하지 않았습니다.
- 위치 표기는 PDF 페이지(p.), Figure, Table 번호입니다.
- 논문에 없고 제 배경지식에서 가져온 내용은 **[논문 외]**로 표시했습니다. 서지 확인이 필요합니다.
- 제 추론이나 계산은 **[해석]** 또는 **[추정]**으로 표시했습니다.
- Figure의 막대 높이처럼 텍스트로 정확히 읽을 수 없는 값은 근사치라고 밝혔습니다.

---

## 1. Executive Summary (10문장)

1. SCRIBES는 HTML 표·리스트·infobox 같은 반정형(semi-structured) 웹 콘텐츠에서 (subject, predicate, object) triple을 추출하는 문제를 다룹니다 (p.1).
2. 기존 wrapper/레이아웃 기법은 새 사이트와 스키마에 취약하고, 페이지마다 LLM을 호출하는 방식은 웹 규모에서 비쌉니다 (p.1).
3. 핵심 아이디어는 LLM이 한 페이지를 보고 BeautifulSoup(bs4) 기반 Python 추출 스크립트를 생성하고, 같은 사이트의 구조가 비슷한 페이지 그룹 전체에 재사용하는 것입니다 (Fig.1, p.2).
4. 학습은 RLVR(검증 가능한 보상 강화학습)과 GRPO로 합니다. 보상은 "한 페이지로 만든 스크립트를 그룹 내 모든 페이지에 실행한 점수의 평균"이며, 자기 점수 비중은 1/|G|뿐입니다 (Eq.1, p.4).
5. 긴 HTML은 반복 블록을 "n more … elements"로 접는 dedup으로 줄입니다 (평균 114k→17k 토큰, Table 6, p.17).
6. 소규모 주석 데이터(34그룹·192페이지)로 먼저 학습한 뒤, CommonCrawl의 무라벨 페이지에 LLM 직접추출을 합성 라벨로 붙입니다. 그리고 기존 모델의 스크립트가 빈 결과를 낸 "실패 사례"에서 추가 학습합니다 (Fig.3, p.5).
7. Q-32B(+CC)의 F1^LM은 33.2로, 같은 크기의 agentic 2-shot 기준선(19.4) 대비 +13.8pt이고 GO-120B agentic(34.3)과 비슷합니다 (Table 1, p.6). 다만 페이지별 직접추출 GO-120B(40.4)보다는 낮습니다.
8. 한 사이트에 유사 페이지가 약 4개 이상이면 토큰 효율이 앞서고(speedup = k/ρ), 학습비용을 포함해도 FLOPs가 절감됩니다 (Fig.4, p.7).
9. 추출된 triple을 QA 문맥에 추가하면 GPT-4o 정확도가 82.5→86.6%로 오릅니다 (Table 5, p.9).
10. 한계는 평가 규모가 작고(테스트 22그룹·65예제), 복잡 구조(중첩 리스트, free-form)에서 약하며, 웹 규모 추출물의 정확도는 직접 검증되지 않았다는 점입니다 (Fig.5, p.9; p.5, p.7).

### 1-1. 연구 목적과 필요성
- **목적**
  - 반정형 웹 콘텐츠에서 구조화 지식을 효과적이고 효율적으로 대규모 추출하는 것입니다 (p.1).
  - 이를 위해 사이트 내 레이아웃 유사성을 RL 보상 신호로 쓰는 "스크립트 생성 모델"을 학습합니다.
- **필요성**
  - 웹의 사실 데이터 상당수가 반정형 형식이지만, 서식 때문에 QA 등에서 활용이 어렵습니다 (p.1).
  - flatten/Trafilatura 계열은 표·리스트 구조를 버립니다 (p.3).
  - 페이지별 LLM 추출은 품질은 좋아도 웹 규모에서 자원 소모가 큽니다 (p.1, p.3).
  - 스크립트 정답 라벨은 전문가도 만들기 어렵습니다. 그래서 시연(SFT) 대신 실행 결과로 채점하는 RL을 씁니다 (p.2, p.4).
  - 오픈 사전학습 말뭉치는 반정형 콘텐츠를 체계적으로 걸러냅니다 (C4, Dolma, FineWeb 언급, p.10).

> 💡 **용어: semi-structured content / infobox** – 완전한 DB 표는 아니지만 표·속성-값 목록처럼 구조가 있는 웹 콘텐츠입니다. infobox는 위키백과 우측의 요약 박스 같은 것입니다.
> 💡 **용어: triple / KG** – (주어, 서술어, 목적어) 3개 조합입니다. 이를 모은 지식 표현이 Knowledge Graph(KG)입니다.
> 💡 **용어: RLVR (RL with Verifiable Rewards)** – 정답을 자동으로 검증해 보상을 주는 강화학습입니다. 여기서는 스크립트 실행 결과를 정답 triple과 비교해 점수를 줍니다.

---

## 2. 핵심 주장과 근거 (표)

| # | 핵심 주장 | 근거(저자 보고 수치) | 위치 |
|---|---|---|---|
| C1 | 스크립트 재사용은 페이지별 LLM 추론보다 자원 효율적이다 | 토큰 비율 ρ=8879/2399≈3.7, speedup=k/ρ, 유사 페이지 ≥4개면 이득. 100페이지/그룹·10⁵그룹에서 1.12×10²¹ FLOPs 절감 | p.7, Fig.4, App. D.3 (p.18) |
| C2 | RL 학습 모델이 agentic 스크립트 기준선을 크게 능가한다 | Q-14B(+CC) 21.8 vs 8.0, Q-32B(+CC) 33.2 vs 19.4 (F1^LM, All): 둘 다 +13.8pt. Q-32B(+CC)는 GO-120B agentic(34.3)과 on-par | Table 1 (p.6), p.7 |
| C3 | "그룹 내 교차 점수" 보상이 일반화의 핵심이다 | self-only 보상: Example +1.2, Holdout −7.2, All −4.2 (F1^LM) | Table 2, p.8 |
| C4 | CC 합성 라벨 추가 학습이 성능을 더 올린다 | All F1^LM: 14B 19.9→21.8(+1.9), 32B 28.1→33.2(+5.1) | Table 1, Table 3 (p.8–9) |
| C5 | 실패 사례 서브셋이 유리하다 (32B 기준) | 32B: All CC 29.7 vs Failure-Case CC 33.2 (+3.5). 14B는 비슷(22.0 vs 21.8) | Table 3, p.9 |
| C6 | 노이즈 보상은 gold 보상 학습 이후에 써야 한다 | CC만 9.2, 1:1 혼합 6.5, 순차(주석→CC) 21.8 (All F1^LM, 14B) | Table 10 (p.21), p.8–9 |
| C7 | 도메인 간 전이가 가능하다 | 제품·백과를 test로 제외 학습: 19.4 vs agentic 8.8 (All F1^LM, +10.6) | Table 4 (p.8), App. G.3 (p.21) |
| C8 | HTML dedup이 입력 길이와 성능을 개선한다 | 114,318→16,985 토큰(14.9%). GPT-4o Non-Empty Rate 63.8→94.9% | Table 6 (p.17), Table 7 (p.18) |
| C9 | 추출 triple이 하위 QA를 향상시킨다 | GPT-4o 82.5→86.6(+4.1), Q-14B 74.2→77.3 | Table 5 (p.9), p.10 |
| C10 | 구조가 많고 복잡할수록 어렵다 | Structure Ratio가 높을수록 F1 하락. HT>A-VP>F-F. 중첩 리스트 F1 19.7, 다중 컬럼 18.6 | Fig.5 (p.9), p.10 |
| C11 | fuzzy F1 보상이 LLM-judge F1의 대리 지표로 타당하다 | 상관계수 0.957 (p=1.4×10⁻⁵) | App. F.1 (p.20), Table 9 |
| C12 | 기존 LLM-KG 파이프라인(HippoRAG, AutoSchemaKG)은 이 과제에서 더 낮다 | 2-shot 단순 기준선이 20pt 이상 앞섬(GO-120B) | App. G.1 (p.20–21), Table 11 (p.22) |

---

## 2-1. 상세 설명

### (1) 해결하려는 문제

- 구조가 유사한 반정형 웹페이지 그룹 $G=\{p_1,\dots,p_n\}$이 있습니다 (Fig.2, p.3).
- 각 페이지를 triple 리스트로 파싱하며, 정답은 $y^\star_{p_i}$입니다 (p.3).
- 모델 $LM$이 임의의 $p\in G$로부터 스크립트 $\hat y_p = LM(p)$를 만듭니다. 이를 그룹 모든 페이지에 적용했을 때 정답 triple에 가까운 결과가 나와야 합니다 (p.4).
- 어려운 점은 세 가지입니다.
  - 스크립트 정답 라벨이 없습니다.
  - 새 사이트에도 일반화해야 합니다.
  - HTML이 너무 깁니다.

> 💡 **용어: wrapper induction** – 소수의 라벨 예시에서 특정 사이트용 추출 규칙("wrapper")을 자동으로 학습하는 고전 기법입니다 (p.1, p.3). 사이트 구조가 바뀌면 쉽게 깨집니다.
> 💡 **용어: DOM** – HTML을 트리 구조로 표현한 Document Object Model입니다 (App. E, p.18).

### (2) 제안 방법 (수식 포함)

**① 점수 함수와 교차 보상**

스크립트 $\hat y_p$를 페이지 $q$에 실행한 점수는 다음과 같습니다 (p.4).

$$r(p\to q)=S\big(\hat y_p(q),\,y^\star_q\big)\in[0,1]$$

- $\hat y_p(q)$: $p$에서 생성한 스크립트를 $q$에 실행한 triple 집합
- $y^\star_q$: $q$의 정답 triple
- $S$: 유사도 함수. 학습 시 $S=F_1^{\text{fuzzy}}$를 씁니다.

자기 점수와 교차 점수는 다음과 같이 정의합니다.

$$r_{\text{self}}(p)=r(p\to p),\qquad r_{\text{cross}}(p,q)=r(p\to q)\ (q\neq p)$$

최종 보상 (Eq.1, p.4)은 다음과 같습니다.

$$r_{\text{SCRIBES}}(p)=\frac{1}{|G(p)|}\sum_{q\in G(p)} r(p\to q)=\frac{1}{|G(p)|}\Big(r_{\text{self}}(p)+\sum_{q\in G(p),\,q\ne p} r_{\text{cross}}(p,q)\Big)$$

- $G(p)$: $p$가 속한 그룹. $|G(p)|$는 그룹 크기입니다.
- 자기 점수의 비중은 $1/|G(p)|$입니다. 그룹이 3개면 1/3, 13개면 1/13입니다 **[해석: 계산]**.
- ⚠️ **표기 주의**: 논문 Eq.1의 두 번째 등식은 교차항에 $\frac{|G|-1}{|G|}$ 계수가 붙은 합으로 인쇄되어 있습니다. 이는 합이 아니라 평균일 때만 첫 번째 등식과 같습니다. 위 식은 첫 번째 등식과 일치하는 형태로 제가 정리한 것입니다.

**② ablation용 보상 (Eq.3, p.8)**

$$r_0(p)=r_{\text{self}}(p)$$

**③ 평가 점수 (Eq.2, p.7)**

$$S(p)=\frac{1}{|G(p)|}\sum_{q\in G(p)} S(\hat y_p,y^\star_q)$$

$$S_{\text{example}}(p)=S(\hat y_p,y^\star_p),\qquad S_{\text{holdout}}(p)=\frac{1}{|G(p)|-1}\sum_{q\in G(p),\,q\ne p} S(\hat y_q,y^\star_q)$$

- ⚠️ 논문의 $S_{\text{holdout}}$은 $\hat y_q$로 적혀 있습니다. 문맥상 " $p$의 스크립트 $\hat y_p$를 $q$에 적용"이어야 하는 표기 오류로 보입니다 **[해석]**.

**④ fuzzy 매칭 지표 (App. F.1, p.19)**

gold $G=\{g_i\}$ ($|G|=m$), 예측 $P=\{p_j\}$ ($|P|=n$)에 대해, 유사도 $f^{\text{fuzzy}}(g_i,p_j)\in[0,1]$(문자 수준 매칭 비율)을 가중치로 하는 최대가중 이분매칭 $M$을 구합니다.

$$P^{\text{fuzzy}}=\frac{\sum_{(g,p)\in M}f^{\text{fuzzy}}(g,p)}{|P|},\quad R^{\text{fuzzy}}=\frac{\sum_{(g,p)\in M}f^{\text{fuzzy}}(g,p)}{|G|},\quad F_1^{\text{fuzzy}}=\frac{2P^{\text{fuzzy}}R^{\text{fuzzy}}}{P^{\text{fuzzy}}+R^{\text{fuzzy}}}$$

LLM-judge 지표는 매칭된 쌍에 LLM의 0/1 판정 $f^{\text{LM}}(g,p)\in\{0,1\}$을 넣어 같은 방식으로 계산합니다 (p.20).

$$P^{\text{LM}}=\frac{\sum_{(g,p)\in M}f^{\text{LM}}(g,p)}{|P|},\quad R^{\text{LM}}=\frac{\sum_{(g,p)\in M}f^{\text{LM}}(g,p)}{|G|},\quad F_1^{\text{LM}}=\frac{2P^{\text{LM}}R^{\text{LM}}}{P^{\text{LM}}+R^{\text{LM}}}$$

- 학습 중에는 정확한 매칭 대신 탐욕(greedy) 근사를 씁니다. 후보쌍을 점수 내림차순으로 정렬해 충돌 없이 추가하고, 60초 timeout 시 외삽합니다 (App. F.2, p.20).

> 💡 **용어: bipartite matching / Jonker–Volgenant** – 예측 triple과 정답 triple을 1:1로 짝지어 총점을 최대화하는 문제입니다. Jonker–Volgenant는 이를 효율적으로 푸는 알고리즘이며, 논문은 scipy `linear_sum_assignment`를 사용했습니다.
> 💡 **용어: fuzzy matching / Levenshtein** – 문자열이 정확히 같지 않아도 편집거리 기반으로 비슷함을 0~1로 재는 방식입니다 (fuzzywuzzy `ratio`).
> 💡 **용어: LLM-as-a-judge** – LLM이 "두 triple이 의미상 같은가"를 Yes/No로 판정하는 평가 방식입니다. 여기서는 Llama-3.3-70B-Instruct를 씁니다.
> 💡 **용어: macro average** – 예제별 점수를 먼저 구한 뒤 평균을 내는 방식입니다. 전체 P, R 평균으로 F1을 계산하는 harmonic 방식($F_1^H=\frac{2\bar P\bar R}{\bar P+\bar R}$, Eq.4, p.22)과 값이 다릅니다.

**⑤ GRPO** (논문은 Shao et al., 2024를 인용만 하고 수식을 쓰지 않음 → 아래는 **[논문 외] 일반 정의**)

롤아웃 $K$개(논문 설정 $K=8$, p.18)의 보상 $r_i$로 그룹 정규화 이점을 구합니다.

$$\hat A_i=\frac{r_i-\text{mean}(r_{1:K})}{\text{std}(r_{1:K})}$$

- 이점은 PPO식 클리핑 목적함수에 쓰이고, KL 정규화가 붙습니다.
- 논문 설정: KL 계수 0.001, 엔트로피 손실 없음, lr 1e-6 상수 (p.18).
- KL 항은 k3 근사입니다. 논문 표기 그대로 $k_3=\frac{\pi_{\text{new}}}{\pi_{\text{old}}}-\log\frac{\pi_{\text{new}}}{\pi_{\text{old}}}-1$ (p.18).
- ⚠️ 여기서 GRPO의 "group"(롤아웃 묶음)과 SCRIBES의 "group"(유사 웹페이지 묶음)은 서로 다른 개념입니다.

> 💡 **용어: GRPO** – 같은 입력에 대한 여러 샘플 보상의 평균·표준편차로 이점을 정규화해 별도 가치함수(critic) 없이 정책을 업데이트하는 PPO 계열 알고리즘입니다.
> 💡 **용어: rollout / KL loss / k3** – rollout은 모델이 샘플링한 출력 1개입니다. KL loss는 학습 중 모델이 기준 모델에서 너무 멀어지지 않게 하는 항이고, k3는 그 KL의 분산 낮은 추정식입니다.

**⑥ 효율 식 (p.7)**

$$\rho=\frac{8879}{2399}\approx3.7,\qquad \text{speedup}=\frac{k}{\rho}$$

- $k$: 구조가 유사한 페이지 수
- 8,879: dedup HTML의 평균 토큰 수
- 2,399: flatten HTML의 평균 토큰 수
- 해석: 페이지당 flatten LLM 호출은 $k\times2399$ 토큰이고, 스크립트 방식은 대표 페이지 1개 $8879$ 토큰입니다. 이 식은 출력 토큰·재시도·모델 크기 차이를 무시한 단순화입니다 **[해석]**.
- 계산량은 순전파 FLOPs ≈ $2\times$파라미터 수×토큰으로 추정하고, 학습 비용은 Q-14B $1.71\times10^{20}$, Q-32B $3.56\times10^{20}$ FLOPs입니다 (App. D.3, p.18).
- 제 검산 **[추정]**: Q-32B, 10⁷페이지에서 flatten은 $2\times32\text{B}\times2399\times10^7\approx1.5\times10^{21}$ FLOPs이고, SCRIBES는 학습 $3.6\times10^{20}$에 추론 $\sim6\times10^{19}$을 더해 약 $4\times10^{20}$입니다. 차이는 약 $1.1\times10^{21}$로 논문의 1.12×10²¹과 일치합니다. 출력 토큰 포함 여부는 확인되지 않았습니다.

> 💡 **용어: FLOPs / TFLOPs** – 부동소수점 연산 횟수(총량)입니다. TFLOPs는 초당 10¹² 연산(처리속도)입니다.

**⑦ HTML Dedup (Algorithm 1, p.17; Fig.6, p.16)**

- script/style/meta 등 태그와 대부분 속성·주석을 제거합니다.
- 부모 노드(ul, ol, div, section, tbody, thead, select)의 자식을 $\text{sig}(c)=(c.\text{tag},\text{sort}(c.\text{class}))$로 묶습니다.
- 한 묶음이 $z$개(기본 $z=3$)를 넘으면 처음 $z$개만 두고 "… |G|−z more <tag> elements …" 주석을 넣습니다.

### (3) 모델/시스템 구조

| 구성 | 내용 | 위치 |
|---|---|---|
| 정책 모델 | Qwen2.5-Instruct 14B / 32B, LoRA 없이 전체 파라미터 FSDP 미세조정 | p.6, p.18 |
| 입력 | dedup된 대표 HTML 1개 + 프롬프트(Prompt 19), 최대 프롬프트 28,672 / 응답 4,096 토큰 (총 32,768, YaRN 미사용) | p.17, p.28 |
| 출력 | `def main(html)->List[tuple(str,str,str)]` 형태의 bs4 파이썬 스크립트 (예: Table 13) | p.24, p.28 |
| 보상 | 그룹 내 모든 페이지에 실행, $F_1^{\text{fuzzy}}$ 평균 (Eq.1) | p.4 |
| 학습 | 주석 데이터 50 epoch → CC 데이터 1 epoch, 롤아웃 8개, 32B는 grad clip 0.5 | p.18 |
| CC 파이프라인 | blacklist→영어→도메인 그룹화→n≥30→LLM 분류(m≥90%)→1:k 구성(k=13)→LLM 직접추출(합성 라벨)→빈 예측(실패) 필터 | Fig.3, p.5 |
| CC 임계값 | n=30, m=90, k=13. 그룹 19,566→2,003→직접추출 후 1,898개 | p.5 |
| 추론 | 대표 페이지 1개→스크립트 생성→유사 페이지 전체에 적용 | Fig.1, p.2 |

> 💡 **용어: CommonCrawl (CC)** – 웹을 주기적으로 크롤링해 공개하는 대규모 웹 아카이브입니다.
> 💡 **용어: BeautifulSoup(bs4)** – HTML을 파싱해 태그·텍스트를 뽑는 Python 라이브러리입니다.
> 💡 **용어: FSDP / YaRN / LoRA** – FSDP는 파라미터를 GPU들에 쪼개 저장하는 분산 학습 기법입니다. YaRN은 모델 컨텍스트 길이를 확장하는 기법이며, 저자들은 이를 쓰면 학습이 불안정해 32k 이내로 제한했다고 합니다 (p.17 각주). LoRA는 일부 저차원 파라미터만 학습하는 방식으로, 여기서는 쓰지 않았습니다.
> 💡 **용어: ReAct식 agentic-n-iter** – 스크립트가 실패하거나 빈 결과일 때 실행 피드백을 LLM에게 되돌려 최대 n번 재시도하는 기준선입니다 (p.6).
> 💡 **용어: n-shot / flatten** – n-shot은 예시 HTML과 정답을 프롬프트에 n개 넣는 방식입니다. flatten은 HTML을 `get_text()`로 평문화해 입력하는 방식입니다. 일반화 요구와 dedup이 없습니다 (p.6).
> 💡 **용어: HippoRAG / AutoSchemaKG** – 모두 LLM으로 KG를 만드는 파이프라인입니다. 페이지마다 LLM을 여러 번 호출합니다 (App. G.1, p.20–21).

### (4) 성능 향상 (저자 보고)

**Table 1 (p.6), F1^LM (All / Example / Holdout)**

| 모델 | All | Example | Holdout |
|---|---|---|---|
| Q-14B SCRIBES | 19.9 | 26.7 | 16.7 |
| Q-14B (+CC) | 21.8 | 30.0 | 17.7 |
| Q-32B SCRIBES | 28.1 | 30.3 | 26.8 |
| Q-32B (+CC) | 33.2 | 34.6 | 32.4 |
| 최강 script-gen 기준선 GO-120B agentic 2-shot | 34.3 | 36.6 | 33.3 |
| GPT-4o agentic 2-shot | 24.4 | 31.2 | 21.1 |
| 직접추출 GO-120B 2-shot flatten | 40.4 | – | – |

- 효율: 113,129 페이지 중 직접 모델 예측은 4,661건이며, 나머지는 스크립트로 추출해 2,788,760 triple을 얻었습니다 (p.7).
- QA (Table 5, p.9): GPT-4o 82.5→86.6, Q-14B 74.2→77.3, Q-32B 70.8→73.2.
  - gold triple 추가 시에는 GPT-4o 87.4, Q-14B 78.2, Q-32B 74.8입니다.

### (5) 한계

- **저자가 인정한 것**
  - 복잡 구조에서 성능이 하락합니다 (Fig.5).
  - 직접추출 상위 기준선보다 IE 정확도가 약간 낮습니다 (p.10).
  - Q-3B, Q-7B QA에서는 직접추출 triple보다 못합니다 (Table 5).
  - 웹 규모 검증은 CC의 <1%만 사용했으며 "feasibility" 수준입니다 (p.6).
  - 도메인 다양성이 필요합니다 (p.9).
- **제가 추가로 본 한계**는 5장에서 다룹니다.

---

## 3. 각 주장의 위치 표시

위 2장 표와 아래 4장에서 모든 주장에 [p./Fig./Table]을 병기했습니다.

---

## 4. 저자 보고 vs 내 해석 (분리)

### 4-1. 연구 주제

| 저자가 직접 보고 | 내 해석 [해석] |
|---|---|
| 반정형 웹 콘텐츠 추출을 "스크립트 생성 + 그룹 재사용"으로 해결한다 (p.1–2) | 문제를 "페이지→triple" 변환에서 "사이트 레이아웃→프로그램 합성"으로 바꾼 점이 기여다. 단, 추출 품질은 한 대표 페이지가 그룹 전체를 대표한다는 가정에 의존한다. |
| 레이아웃 유사성을 보상으로 쓴다 (abstract) | 라벨 없는 "일반화"를 직접 최적화 목표로 만든 설계로, 새 사이트에 약한 wrapper의 문제를 학습 목표에 반영했다. |

### 4-2. 방법

| 저자가 직접 보고 | 내 해석 [해석] |
|---|---|
| Eq.1: 자기 점수는 1/|G|만 기여, 교차 점수가 다수 (p.4) | 그룹 크기가 3이면 자기 비중 33%, 13이면 7.7%라 그룹 크기에 따라 보상 성격이 달라진다. 가중치 변화 실험은 없다. |
| CC에서는 LLM 직접추출을 합성 정답으로 쓴다 (p.5) | 선생(GO-120B 직접추출, F1 40.4)이 학생(33.2)보다 강한 지식증류와 유사한 구도다. "자기 개선"이라기보다 강한 비싼 모델의 지식을 스크립트 형태로 압축한다는 해석이 자연스럽다. |
| 실패(빈 예측) 사례만 학습 (p.5, Table 3) | 노이즈 보상 위험을 줄이는 휴리스틱이다. "빈 결과"는 쉬운 실패 신호인데, 이는 "틀린 결과"와는 다른 집합이다. |
| Dedup으로 성능이 개선된다 (p.4, Table 7) | 증거는 기준선(L-70B, GPT-4o)에 대한 것이다 (Table 7). 학습된 모델에서 dedup on/off 비교는 본문에 제시되지 않았다. 개선의 상당 부분은 컨텍스트 초과로 인한 빈 출력 감소(Non-Empty Rate)로 보인다. |

### 4-3. 결과

| 저자가 직접 보고 | 내 해석 [해석] |
|---|---|
| "강한 기준선을 13% 이상 능가" (abstract, p.1) | 정확히는 같은 크기 모델의 agentic 2-shot 대비 +13.8 F1 포인트다 (Table 1). 최강 script-gen 기준선(GO-120B, 34.3)이나 직접추출(40.4)에는 못 미친다. 따라서 "가장 강한 기준선을 능가"는 아니다. |
| Q-32B가 GO-120B agentic과 on-par (p.7) | 33.2 vs 34.3, 차이 −1.1은 22개 테스트 그룹 규모에서 통계적으로 구분하기 어렵다. 파라미터 수 대비 효율은 강점이다. |
| 14B/32B 기준선이 매우 낮다 (8.0, 19.4) | 2-shot 예시 HTML 2개 + 입력을 합치면 dedup 후에도 평균 ~17k 토큰×3 ≈ 51k로 32k 컨텍스트를 넘을 수 있다 (Table 6 기준). 기준선의 낮은 점수에는 컨텍스트 초과/빈 출력이 섞였을 가능성이 있다. Table 1 기준선의 Non-Empty Rate는 보고되지 않았다. |
| QA +4.1 (GPT-4o) (Table 5) | 효과는 있어 보이나 416문항에서 쌍체 검정이 없다. GPT-4o에서 SCRIBES triple(86.6)이 더 높은 F1의 직접추출 triple(82.7)보다 도움이 된 이유는 논문이 설명하지 않는다. triple의 길이·형태·재현율 차이 등 가설만 가능하다. |
| 교차 보상이 일반화에 필요 (Table 2) | Holdout −7.2는 큰 효과이며 방향성은 설득력 있다. 다만 14B 단일 실험이다. |
| CC 학습이 14B +1.9, 32B +5.1 (Table 3) | 14B에서는 Example(+3.3) > Holdout(+1.0)라 "일반화"보다 "Example 적합" 이득이 컸다. 32B는 Holdout +5.6 > Example +4.3이다. |
| 도메인 전이 +10.6 (Table 4) | Holdout만 보면 12.2 vs 7.2로 격차가 5pt뿐이다. 전이의 상당 부분은 Example 스크립트 품질(30.4 vs 20.0)에서 나온다. |

---

## 5. 통계적으로 취약한 부분 & 비교 불가능한 수치

### ⚠️ 통계적 취약

1. **평가 규모가 작다.**
   - 테스트는 22그룹, 76페이지이며, 컨텍스트 제한 필터 후 65예제만 남습니다 (App. D.1, p.18).
   - 학습은 192페이지 중 141개입니다.
   - 필터로 테스트의 약 14%가 제외되어 짧은 페이지로 편향될 수 있습니다.
2. **신뢰구간, 시드 반복, 유의성 검정이 전혀 없습니다.**
   - 표 전체가 단일 실행 결과로 보입니다.
   - 1~2pt 차이는 해석이 곤란합니다. 예: Table 2 Example +1.2, Table 3의 Q-14B All CC vs Failure CC 22.0 vs 21.8.
3. **예제 간 독립성 위반.**
   - 그룹 크기 n마다 n개 예제를 만들므로 같은 그룹의 예제는 상관됩니다 (p.5).
   - 13페이지 그룹 10개가 전체 268페이지 중 130페이지(약 49%)입니다 (p.5 계산 **[해석]**).
   - 실효 표본은 페이지 수가 아니라 그룹 수(22)에 가깝습니다.
4. **단일 60/40 분할만 사용했습니다.** 교차검증이 없어 분할 운에 따른 분산을 모릅니다.
5. **Table 5 (QA):**
   - 416문항에서 정확도 약 80% 부근의 단순 표준오차는 약 2pt입니다 **[추정: 이항분포 근사]**.
   - 0.2~1pt 차이(예: Q-3B, Q-7B 비교)는 노이즈일 수 있습니다.
   - Q-32B(70.8)가 Q-14B(74.2)보다 낮은 비단조성도 노이즈를 시사합니다.
6. **Fig.5 오류 분석:** 5개 bin에 65예제를 나누면 bin당 약 13개이고, 중앙값만 보고됩니다. 유형별 표본 수는 없습니다.
7. **상관 0.957 (App. F.1):** Table 9의 10개 설정에 대한 상관이며, p=1.4×10⁻⁵는 n=10과 정합합니다 **[추정: 역산]**. 예제 수준 보상 타당성을 보장하지는 않습니다. 인간 합의 95%도 Sun et al. 인용값입니다 (p.20).
8. **CC 분류기:** 정밀도 90%, 재현율 72%가 50개 샘플 추정이라 구간이 매우 넓습니다 (p.5).
9. **Table 10 이상치:** 주석 + CC 1:1 혼합(6.5)이 기준선(약 7~8)보다도 낮습니다. 학습 붕괴/불안정 가능성이 있는데 원인 분석이 없습니다. 또한 "CC only"와 "mixed"가 All CC인지 Failure-Case CC인지 명시가 없습니다.
10. **Failure-Case vs All CC 비교 교란:** 두 집합의 학습 예제 수가 보고되지 않아 데이터 양과 선택 효과가 분리되지 않습니다.
11. **판정자 동일 계열:** 평가 판정자가 Llama-3.3-70B이고, QA 쌍 생성도 Llama 70B입니다 (App. H, p.22). 같은 계열 편향 가능성이 있습니다.
12. **기준선 설정 선택:** "각 모델의 가장 강한 기준선만 표시"했습니다 (Table 1 캡션). 이는 기준선에 유리한 보수적 선택이지만, 설정 선택이 테스트셋 점수로 이뤄진 것으로 보입니다.
13. **모델/체크포인트 선택 기준 미공개:** 50 epoch 학습 후 어느 체크포인트를 보고했는지 불명확합니다. 테스트 기반 선택이라면 누수 가능성이 있습니다.

### ⛔ 비교 불가능한 수치

| 항목 | 이유 | 위치 |
|---|---|---|
| Table 1의 ∗ 행 (L-70B, Fine-tuned L-70B, GPT-4o; Sun et al.) | 전체 세트 기준이며 F1이 harmonic 방식입니다. 예: GPT-4o R 35.1, P 23.8이면 $2PR/(P+R)=28.3$으로 표의 28.3과 일치하고, 다른 행은 예제별 F1 평균(macro)입니다 **[검산]**. | p.6 |
| 직접추출 행과 script-gen 행 | 직접추출은 일반화 요구와 dedup이 없고 Example/Holdout 열이 없습니다. 비용 구조도 다릅니다. | p.6 |
| Macro F1 vs harmonic $F_1^H$ | 같은 모델에서도 값이 다릅니다 (Q-14B flatten 29.87 vs 33.19). 두 표를 섞어 비교하면 안 됩니다. | Table 11 (p.22) |
| Table 4 vs Table 1 | 테스트셋이 다릅니다 (제품·백과 59예제). Table 4의 SCRIBES 모델이 +CC인지 명시가 없습니다. | p.8, App. G.3 |
| Table 5의 "Best Q-32B triples" | 어떤 Q-32B(+CC 여부)인지 명시가 없습니다. 맥락에 들어간 triple 수·길이도 방법마다 다릅니다. | p.9 |
| 효율 비교 (Fig.4) | 비교 대상이 Q-32B 페이지별 flatten입니다. 정확도가 더 높은 GO-120B 직접추출 기준의 비용이 아닙니다. FLOPs는 $2\times N\times$토큰의 추정치입니다. | p.7, p.18 |
| 토큰 통계 | 8,879/2,399(CC 잔여 세트, p.7)와 16,985/114,318(SemiBench, Table 6)은 다른 데이터셋입니다. | p.7, p.17 |
| Fig.5b "All" 행 | 값(39.5, 35.5, 34.6)이 Table 1 Q-32B(+CC)의 **Example** 열(R=39.5, P=35.5, F1=34.6)과 일치합니다. 헤더는 P, R, F1 순서이므로 열 이름이 뒤바뀌었거나 Example 기준일 가능성이 있습니다 **[해석]**. | p.9 |
| 본문 참조 오류 | "Table 5b"는 Figure 5b의 오기로 보입니다 (p.10). 5.1절의 "validation examples"도 test와 혼용된 표현입니다. | p.10 |

---

## 6. 문서가 답하지 않는 질문

1. CC 서브셋에서 추출된 2,788,760 triple의 **정확도**는 얼마인가? 정답이 없어 측정되지 않았습니다 (p.7).
2. CC 학습 도메인과 SemiBench 테스트 사이트의 **중복/누수** 여부는? 둘 다 CC 기반입니다 (p.5).
3. Failure-Case CC의 **샘플 수**와 비율은? Table 3 설명에도 없습니다.
4. 여러 **시드**에서의 분산과 신뢰구간은?
5. 체크포인트 선택 기준과 검증 세트는 무엇인가? (50 epoch, 소규모 데이터에서의 과적합 가능성)
6. 자기 점수 가중치(1/|G|), 그룹 크기, k, dedup의 $z$ 값에 대한 **민감도 ablation**은? Dedup 효과는 기준선에서만 검증됐고, 학습 모델에서의 on/off 비교는 없습니다.
7. 32B 이상 모델이나 더 큰 CC로 **스케일링**하면 어떻게 되나? (p.6은 가설만 제시)
8. 사이트 **템플릿이 바뀌면**(시간적 변화) 스크립트는 어떻게 수명을 관리하나? 자바스크립트 렌더링 페이지, 비영어 페이지, 그룹화 휴리스틱(URL prefix)의 오류율도 다뤄지지 않았습니다.
9. LLM이 생성한 코드를 임의의 웹 HTML에서 **실행**할 때의 보안/샌드박스는? 논문에 언급이 없습니다.
10. 사이트 간 **predicate 정규화**(예: "Seq" vs "sequence")와 KG 통합은 어떻게 하나? (Table 14에 불일치 사례가 나타남)
11. 고전/전용 기법(wrapper induction, ZeroShotCeres 등)이나 HTML→Markdown/JSON 모델(ReaderLM-v2 등)과의 **직접 비교**는? 관련 연구로만 언급됩니다 (p.3).
12. SCRIBES triple이 더 낮은 F1에도 GPT-4o QA를 더 개선한 **원인**은?
13. "Multi-page 복잡 QA"와 "사전학습 데이터 확보" 주장은 **실험 없이 논의만** 되었습니다 (p.10).
14. 크롤링 윤리(robots.txt, 라이선스)는? 성인물 필터만 언급됩니다 (p.5). 코드는 "공개 예정"으로 기술되어 있습니다 (p.1 각주).
15. 기준선 Q-14B/32B의 Table 1 Non-Empty Rate는? 컨텍스트 초과 영향이 분리되지 않았습니다.

---

## 7. 가장 중요한 그림 5개 해석

Table 1, 2, 3, 5는 그림이 아니지만 핵심 수치를 담고 있어 2장 표에 반영했습니다.

### Figure 1 (p.2) – 프레임워크 개요
- **내용:** 학습 시에는 그룹마다 대표 페이지 1개를 dedup해 모델에 넣고(①) 스크립트 1개를 생성하며(②), 그룹 내 모든 페이지에 적용한 결과를 사람/합성 주석과 비교해 보상을 계산하고 가중치를 갱신합니다(③). 추론 시에는 같은 흐름으로 보지 못한 사이트에 일반화합니다(④).
- **해석 [해석]:** 핵심은 "입력은 1페이지, 보상은 N페이지"의 비대칭입니다. 이 구조가 일반화를 학습 신호로 만듭니다. 라벨 소스(사람/합성)만 바꾸고 파이프라인은 동일하다는 점도 보여줍니다.

### Figure 3 (p.5) – CC 합성 데이터 파이프라인
- **내용:** 10단계 필터입니다 (blacklist→영어→그룹화→n≥30→분류기 m≥90%→1:k 구성→LLM 직접추출→체크포인트 빈 예측만 선택→학습 데이터).
- **해석 [해석]:** 수치상 깔때기는 19,566→2,003→1,898이며, 마지막 실패 필터 후 크기는 보고되지 않았습니다. 마지막 필터가 "노이즈 보상 완화"의 핵심이지만, 어떤 종류의 실패인지(빈 출력)가 편향을 만들 수 있습니다.

### Figure 4 (p.7) – FLOPs 추정
- **내용:** x축은 총 페이지 수 $k\cdot g$, y축은 총 FLOPs(로그)입니다. 점선은 페이지별 LLM 추론이고, SCRIBES는 $k=1,10,100,1000$ 곡선입니다. SCRIBES 곡선은 초기에 학습비용으로 평평하다가 페이지가 늘수록 완만히 오릅니다.
- **해석 [해석]:**
  - $k=1$에서는 SCRIBES가 오히려 더 비쌉니다 ($\rho\approx3.7$배 입력 때문). $k$가 클수록 손익분기점이 앞당겨지고 격차가 벌어집니다.
  - 이 결과는 "유사 페이지가 많은 사이트"라는 전제와 스크립트 정확도(F1≈0.33)가 충분하다는 전제를 암묵적으로 포함합니다.
  - 비교 대상은 Q-32B 페이지별 호출이며, 곡선은 추정치입니다 (실측 아님).

### Figure 5 (p.9) – 오류 분석
- **내용:**
  - (a) 왼쪽은 Structure Ratio(HTML 길이/평문 길이) 5개 bin별 F1입니다. 큰 구조일수록 하락하며, 막대 높이는 대략 0.2~0.5 사이입니다. 정확한 값은 읽을 수 없습니다.
  - (a) 오른쪽은 페이지 유형별 F1입니다. Horizontal Table > Attribute-Value > Free-Form 순입니다.
  - (b) 표에서 중첩 리스트의 F1은 19.7, 다중 컬럼은 18.6으로, 표시된 "All" 34.6보다 낮습니다.
- **해석 [해석]:**
  - 복잡도가 오를수록 단일 스크립트로 포괄하기 어렵습니다.
  - bin당 약 13예제, 중앙값만 사용하므로 경향 확인 용도이며 정량 결론은 약합니다.
  - 5b "All" 행의 열 라벨은 Table 1과 맞지 않습니다 (5장 참조).

### Figure 6 (p.16) – HTML dedup 예시
- **내용:** 왼쪽 원본의 script/style 블록은 삭제되고, 반복되는 product-card 5개 중 3개만 남은 뒤 "2 more div class=product-card elements" 주석으로 압축됩니다.
- **해석 [해석]:** 구조를 유지하면서 "반복되는 행"을 요약하는 장치입니다. 모델이 반복 패턴을 보고 루프(`for row in rows`)를 쓰도록 유도하는 효과가 있을 것으로 보입니다. 학습된 모델에서의 인과 검증은 없습니다. 반복 요소 간 편차가 큰 표(헤더 행과 데이터 행의 class가 같은 경우)에서는 정보 손실 위험이 있을 수 있습니다.

> 💡 **용어: Structure Ratio / Tag Count** – 각각 HTML 길이 대비 텍스트 길이 비율과 전체 태그 수로, 구조 복잡도의 대리 지표입니다 (App. E, p.18–19).

---

## 8. 결론: 시사점과 후속 연구

### 저자가 제시한 시사점과 계획 (p.10, p.6, p.9)
- **시사점**
  - 레이아웃 유사성 기반 RL로 일반화 가능한 스크립트를 학습할 수 있습니다.
  - CC 합성 데이터가 성능을 더 올립니다.
  - QA 개선과 자원 효율이 입증됩니다.
- **후속 방향**
  - 여러 페이지에 걸친 집계·랭킹형 복잡 QA (예: "가장 최근 보고서는?").
  - 반정형 콘텐츠를 사전학습 말뭉치에 반영 (C4/Dolma/FineWeb은 이를 거의 제외함).
  - CC의 더 큰 부분으로 파이프라인 확장 (현재는 <1%).
  - 다양한 도메인·레이아웃 데이터로 학습 권장 (p.9).

### 8-1. "모델의 일반화 성능 향상 가능성" 중점 분석

**① 논문이 보여준 증거**

| 증거 | 수치 | 의미 |
|---|---|---|
| 교차 보상 vs self-only (Table 2) | Holdout F1 16.7 vs 9.5 | 일반화를 직접 보상으로 쓰는 것이 핵심입니다. |
| Example–Holdout 격차 (Table 1, F1) | Q-32B(+CC) 34.6/32.4 (격차 2.2), Q-14B 26.7/16.7 (10.0), GPT-4o agentic 31.2/21.1 (10.1) | 32B SCRIBES가 격차가 가장 작습니다 **[해석: 계산]**. 큰 모델일수록 일반화가 잘 될 가능성을 시사합니다. |
| CC 학습 (Table 3) | 32B Holdout 26.8→32.4 (+5.6) | 분포 확장이 일반화에 도움이 됩니다 (Table 8: CC는 태그 수가 적고 단순한 편). |
| 14B + CC | Example +3.3, Holdout +1.0 (Table 1) **[해석: 계산]** | 작은 모델은 CC 데이터에서 일반화보다 Example 적합 이득이 컸습니다. |
| 도메인 held-out (Table 4) | All 19.4 vs 8.8, Holdout 12.2 vs 7.2 | 전이는 있으나 Holdout 격차는 작고 표본이 작습니다 (59예제). |
| 순차 학습 (Table 10) | 주석→CC 21.8, 혼합 6.5 | 일반화 향상에는 "정확한 보상으로 먼저 기초를 쌓은 뒤 확장"이라는 순서가 중요합니다. |
| 복잡 구조 (Fig.5) | 중첩/다중 컬럼 F1 ≈ 19 | 일반화 천장은 구조 복잡도에 의해 제한됩니다. |

**② 해석 및 가능성 [해석]**
- 일반화 향상의 두 축은 (a) 보상 설계(교차 보상)와 (b) 데이터 다양성(CC)입니다. 둘 다 규모를 키울 여지가 있습니다.
- 다만 합성 라벨의 상한은 "선생" 성능입니다 (GO-120B 직접추출 약 40 F1). 현재 학생(33.2)은 상한에 근접하지 않았지만, 상한이 병목이 될 수 있습니다.
- CC가 상대적으로 단순한 페이지라는 점 (Table 8) 때문에, 어려운 구조(Fig.5)에 대한 일반화는 CC 확장만으로 해결되기 어려울 수 있습니다.

**③ 일반화를 더 높이기 위한 후속 방향 (제안)**
1. **보상 변형:** 그룹 내 교차 점수의 평균 대신 최솟값/분위수(worst-case)를 쓰거나, 자기 점수 가중치를 스케줄링하고 가중치 ablation을 수행합니다.
2. **그룹 구성 개선:** 그룹 크기와 이질성을 커리큘럼으로 조절하고, URL prefix 휴리스틱 대신 DOM 유사도로 그룹화합니다. 그리고 레이아웃이 다른 변형 페이지를 합성합니다 (DOM augmentation).
3. **실패 기반 확장 확대:** "빈 출력" 외에 낮은 교차 점수, 자기-교차 불일치 사례를 선택 기준으로 쓰고 이를 ablation합니다.
4. **합성 라벨 품질 향상:** 여러 LLM의 합의나 일관성으로 정제합니다 (TTRL류 합의 보상과 결합). 라벨 신뢰도를 보상 가중치로 사용합니다.
5. **하이브리드 추론:** 스크립트가 빈 결과나 낮은 신뢰도를 내는 페이지만 페이지별 LLM으로 폴백합니다.
6. **평가 강화:** 사이트 단위 k-fold, 도메인 leave-one-out, 시드 반복과 부트스트랩 CI, 시간 경과에 따른 템플릿 변화 테스트를 도입합니다.
7. **구조 특화:** 중첩 리스트·다중 컬럼용 중간 표현(예: 표 정규화 후 추출)을 도입합니다.

### 8-2. 2020년 이후 관련 최신 연구 비교

논문 내 인용 연구와 [논문 외] 연구를 구분했습니다. 아래 서술은 논문이 설명한 범위 또는 제 기억의 개략적 설명이며, 세부 수치 비교는 하지 않았습니다.

| 연구(연도) | 요지 | SCRIBES와의 관계 |
|---|---|---|
| ZeroShotCeres (Lockard et al., ACL 2020) | 반정형 웹페이지의 zero-shot 관계 추출 | 신경망 기반 사이트 일반화를 시도합니다. SCRIBES는 실행 가능한 코드를 생성하므로 해석 가능성과 비용 면에서 다릅니다. 직접 비교는 없습니다. |
| ReAct (Yao et al., 2022) | 추론+행동(피드백 반영) 에이전트 | SCRIBES의 agentic-n-iter 기준선의 틀입니다. SCRIBES는 이 반복 없이 단일 생성으로 학습한 모델이 이를 능가한다고 주장합니다. |
| Trafilatura (Barbaresi, 2021), Firecrawl, newspaper4k | HTML→텍스트/마크다운 변환 | 구조를 평탄화하는 방식으로, SCRIBES는 구조를 보존해 추출합니다 (p.3). |
| HtmlRAG (Tan et al., WWW 2025) | RAG에서 평문보다 HTML이 낫다 | SCRIBES의 동기를 뒷받침합니다 (p.3). |
| ReaderLM-v2 (Wang et al., 2025) | 소형 LM으로 HTML→Markdown/JSON, SFT+RL | 페이지별 변환이며 SCRIBES는 사이트 단위 스크립트 재사용입니다. 실험 비교는 없습니다. |
| olmOCR (Poznanski et al., 2025) | VLM으로 PDF를 읽기 쉬운 형식으로 변환 | 유사 목적(구조 유지 변환)이나 페이지별 추론입니다. |
| HippoRAG (2024), AutoSchemaKG (2025), EDC (Zhang & Soh, 2024) | LLM 기반 KG 구축 | 페이지별 LLM 호출이며 SCRIBES 실험에서 HippoRAG·AutoSchemaKG는 이 과제에서 낮은 점수를 받았습니다 (Table 11). EDC는 인용만 되고 실험 비교는 없습니다. |
| GRPO/DeepSeekMath (Shao et al., 2024) | 그룹 상대 정책 최적화 | SCRIBES의 RL 알고리즘입니다. |
| TTRL (2025), Intuitor 계열 (Zhao et al., 2025), Prabhudesai et al. (2025), Spurious Rewards (Shao et al., 2025) | 외부 라벨 없는/약한 보상 RL | SCRIBES는 내부 신호 대신 LLM 직접추출을 합성 라벨로 쓰고 레이아웃 유사성을 구조적 보상으로 씁니다 (p.3). |
| **[논문 외]** Evaporate (Arora et al., 2023; arXiv 2304.09433) | LLM이 반정형 문서에서 추출 함수를 코드로 합성해 재사용 | SCRIBES와 개념적으로 가장 가까운 선행 연구로 기억합니다. 논문은 비교·인용하지 않았습니다. |
| **[논문 외]** MarkupLM (Li et al., ACL 2022) | HTML 마크업을 사전학습에 반영한 모델 | 구조 인식 모델 계열로, 비교 대상이 될 수 있습니다. |
| **[논문 외]** DeepSeek-R1 (2025), Tülu 3 (2024; RLVR 용어 제안) | 검증 가능한 보상 RL로 추론 능력 향상 | SCRIBES는 RLVR를 "코드 실행 검증" 영역에 적용한 사례로 볼 수 있습니다 **[해석]**. |

### 앞으로의 연구에 미치는 영향 [해석]
1. "프로그램 합성 + 구조적 일관성 보상"은 웹 추출 외 PDF·로그·표 정규화 등 반복 템플릿이 있는 다른 과제에도 이식 가능한 설계 원리입니다.
2. 핵심 교훈은 "추론 비용을 학습 비용으로 이전(amortize)"하는 것입니다. 코드를 중간 산출물로 쓰면 RL 보상을 실행으로 검증할 수 있습니다.
3. 사전학습 데이터에서 누락된 반정형 데이터를 복원하는 데이터 엔진으로 활용될 수 있습니다 (저자 제안, 미검증).
4. 합성 라벨 + 실패 사례 중심 커리큘럼 + 순차 학습(Table 10)은 노이즈 보상 RL 설계의 참고 사례입니다.

### 앞으로 연구 시 고려할 점
- **평가:** 더 큰 테스트셋, 그룹 단위 통계 분석, 시드/CI, 정확도 vs 비용 파레토 곡선, 직접추출 상위 기준선과의 동일 조건 비교가 필요합니다.
- **지표 일치:** macro vs harmonic F1, 판정자 편향(Llama 계열 판정자와 QA 생성), 결과 중복(누수) 점검이 필요합니다.
- **안전·법·윤리:** 생성 코드 샌드박싱, robots.txt/라이선스, 개인정보 포함 페이지 필터링이 필요합니다.
- **견고성:** 템플릿 변경 감지·재생성, JS 렌더링 페이지, 다국어 대응이 필요합니다.
- **하위 활용:** 사이트 간 스키마 정규화와 KG 통합, 다중 페이지 질의 실험으로 주장을 실증해야 합니다.

---

## 참고자료 (출처)

**주 자료 (직접 분석)**
- Shicheng Liu et al., *"SCRIBES: Web-Scale Script-Based Semi-Structured Data Extraction with Reinforcement Learning"*, ICLR 2026, arXiv:2510.01832v2 (제공된 PDF 2510.01832v2.pdf)

**위 논문의 참고문헌 목록에서 언급한 자료 (논문 서술 범위 내에서만 사용)**
- Lockard et al., *ZeroShotCeres: Zero-shot relation extraction from semi-structured webpages* (ACL 2020)
- Yao et al., *ReAct: Synergizing reasoning and acting in language models* (2022)
- Barbaresi, *Trafilatura: A Web Scraping Library and Command-Line Tool for Text Discovery and Extraction* (2021)
- Tan et al., *HtmlRAG: HTML is better than plain text for modeling retrieved knowledge in RAG systems* (WWW 2025)
- Wang et al., *ReaderLM-v2: Small language model for HTML to Markdown and JSON* (2025)
- Poznanski et al., *olmOCR: Unlocking Trillions of Tokens in PDFs with Vision Language Models* (2025)
- Gutiérrez et al., *HippoRAG: Neurobiologically inspired long-term memory for large language models* (NeurIPS 2024)
- Bai et al., *AutoSchemaKG: Autonomous knowledge graph construction through dynamic schema induction from web-scale corpora* (2025)
- Zhang & Soh, *Extract, Define, Canonicalize: An LLM-based framework for knowledge graph construction* (EMNLP 2024)
- Shao et al., *DeepSeekMath: Pushing the limits of mathematical reasoning in open language models* (2024)
- Zuo et al., *TTRL: Test-time reinforcement learning* (2025)
- Zhao et al., *Learning to reason without external rewards* (2025)
- Prabhudesai et al., *Maximizing confidence alone improves reasoning* (2025)
- Shao et al., *Spurious rewards: Rethinking training signals in RLVR* (2025)
- Sun et al., *Knowledge extraction on semi-structured content: Does it remain relevant for question answering in the era of LLMs?* (2025, SemiBench)
- Kushmerick et al., *Wrapper induction for information extraction* (IJCAI 1997)
- Schulman, *Approximating KL divergence* (2020 블로그)
- Peng et al., *YaRN: Efficient context window extension of large language models* (2023)
- Zhao et al., *PyTorch FSDP: Experiences on scaling fully sharded data parallel* (2023)
- Penedo et al., *The FineWeb datasets: Decanting the web for the finest text data at scale* (2024)
- Soldaini et al., *Dolma: an open corpus of three trillion tokens for language model pretraining research* (2024)
- Raffel et al., *Exploring the limits of transfer learning with a unified text-to-text transformer* (C4, 2023)

**[논문 외] 제 기억 기반 (웹 검색 없음, 서지 확인 필요)**
- Arora et al., *Language Models Enable Simple Systems for Generating Structured Views of Heterogeneous Data Lakes* (Evaporate, arXiv:2304.09433)
- Li et al., *MarkupLM: Pre-training of Text and Markup Language for Visually-rich Document Understanding* (ACL 2022)
- DeepSeek-AI, *DeepSeek-R1* (2025)
- Lambert et al., *Tülu 3* (2024)

**정확성 고지**
- 논문 외 항목의 상세 수치, 정확한 서지, 세부 주장은 확인하지 못해 서술하지 않았습니다.
- Figure 5의 막대 높이 등 이미지 값은 텍스트로 정확히 읽을 수 없어 근사만 언급했습니다.
- Eq.1과 $S_{\text{holdout}}$의 표기 문제, Fig.5b의 열 라벨 문제는 제공된 텍스트 추출본 기준의 관찰입니다. 원본 PDF 렌더링에서 확인하시길 권합니다.
