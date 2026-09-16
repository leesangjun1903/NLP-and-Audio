# Language Models Can Control Their Own Attention

---

## 1. Executive Summary (10문장 이내)

**Declarative Attention (DA)**는 LLM이 추론 과정에서 자신이 주목할 문맥 영역을 명시적으로 선언하도록 유도하는 제로샷(zero-shot) 프로토콜이다.  
기존 Transformer는 디코딩 시 전체 KV 캐시를 매 스텝 읽어야 하는 $O(N)$ 비용 문제를 가진다. DA는 `<global>`, `<focus>`, `<local>` 세 가지 모드를 통해 모델의 Chain-of-Thought 내에서 어텐션 범위를 구조화한다.  
인퍼런스 엔진은 이 선언을 파싱하여 KV 캐시 블록 테이블을 동적으로 재구성하고, 불필요한 블록 읽기를 건너뛴다.  
15개 장문 컨텍스트 태스크에서 Gemma-4-31B 기준 평균 어텐션 토큰 52.0% 감소, 정확도는 1.27pp 하락에 그쳤다.  
DA는 별도 학습 없이 기성 모델(off-the-shelf)에 즉시 적용 가능하며, 현재 결과는 하한선(lower bound)에 해당한다.  
모델 스케일이 커질수록 정확도 손실이 좁혀지는 양의 스케일링 특성을 보인다.  
컨텍스트가 길어질수록 절대적 토큰 절감량이 증가하여 최대 21M 토큰/응답을 절약한다.  
이론적 wall-clock time 기준 Gemma-4-31B에서 $0.71\times$, Qwen-3.6-27B에서 $0.77\times$ 수준의 디코딩 시간 단축이 예측된다.  
DA는 희소 어텐션(sparse attention)의 새로운 축을 열며, 향후 훈련 기반 방법과 결합 시 추가 성능 향상 여지가 크다.

> 📌 **용어 설명**
> - **KV 캐시(Key-Value Cache):** Transformer의 어텐션 연산 시 이전 토큰들의 키(Key)와 값(Value) 벡터를 저장해두는 메모리 버퍼. 디코딩 시 매 스텝 이를 전부 읽어야 하므로 긴 컨텍스트에서 병목이 됨.
> - **제로샷(Zero-shot):** 특정 태스크에 대한 추가 학습 없이, 프롬프트만으로 원하는 동작을 유도하는 방식.
> - **Chain-of-Thought (CoT):** 모델이 중간 추론 과정을 명시적으로 텍스트로 출력하도록 유도하는 기법.

---

### 1-1. 연구의 목적과 필요성

**배경 (p.2, Introduction):**
Transformer 기반 LLM은 디코딩 시 매 스텝마다 누적된 전체 KV 캐시를 HBM(High Bandwidth Memory)에서 읽어야 한다. 예컨대 Qwen-3.5-397B-A17B 모델로 1M 토큰 컨텍스트를 처리할 경우, 매 스텝 약 15GB의 KV 캐시를 로드해야 하며 이는 모델의 17B 활성 파라미터를 로딩하는 것과 비슷한 대역폭 요구량이다.

**문제 정의:**
- 어텐션 가중치는 소수의 토큰에 집중되지만(Child et al., 2019; Zhang et al., 2023), 이 패턴은 사전에 알 수 없음
- 기존 희소 어텐션 방법(Quest, SparQ 등)은 모든 KV를 경량 스코어로 스캔하므로 여전히 $O(N)$ 비용 발생
- 정적 휴리스틱(recency, 과거 어텐션 크기)은 미래 쿼리가 필요로 하는 토큰을 사전 예측하는 데 실패함

**연구 질문:**
> "모델 스스로 어느 부분에 주목해야 하는지 이미 알고 있지 않을까?"

**필요성:**
장문 컨텍스트 서빙이 보편화되는 상황에서, 추가 학습 없이도 LLM이 자신의 어텐션 범위를 직접 제어함으로써 추론 비용을 줄이는 접근이 필요하다.

> 📌 **용어 설명**
> - **HBM(High Bandwidth Memory):** GPU에 탑재된 고속 메모리. KV 캐시 읽기는 HBM 대역폭에 의해 병목이 발생함.
> - **희소 어텐션(Sparse Attention):** 전체 토큰이 아닌 선택된 일부 토큰에만 어텐션을 수행하는 기법.

---

## 2. 핵심 주장과 근거 표

| # | 핵심 주장 | 근거 | 위치 |
|---|-----------|------|------|
| 1 | DA는 제로샷으로 어텐션 비용을 크게 줄인다 | Gemma-4-31B 52.0%, Qwen-3.6-27B 31.1% 어텐션 토큰 감소 | Table 2, p.8 |
| 2 | 정확도 손실은 소폭이다 | 평균 1.27pp (Gemma), 2.75pp (Qwen) 하락 | Table 2, Section 5.1 |
| 3 | 어텐션 마스크가 효율성의 원천이다 | DA vs DA-nm 비교: 마스크가 Gemma에서 71.1% 토큰 절감 | Section 5.1, p.7-8 |
| 4 | 모델 스케일이 클수록 성능 격차가 좁혀진다 | 4B→31B로 갈수록 상대 정확도 29%→99% (Gemma 계열) | Figure 3, Section 5.2 |
| 5 | 컨텍스트가 길수록 절대 절감량이 커진다 | 짧은 컨텍스트 -1M 토큰 vs 긴 컨텍스트 -21M 토큰/응답 | Figure 4, Section 5.3 |
| 6 | 이론적 wall-clock 시간이 단축된다 | $0.71\times$ (Gemma), $0.77\times$ (Qwen) | Table 3, Section 5.4 |
| 7 | 프로토콜 준수율은 모델 크기에 따라 향상된다 | Focus success rate: 58% (E4B) → 99% (31B) | Figure 6, Section 6.2 |

---

### 2-1. 해결 문제 / 제안 방법 / 모델 구조 / 성능 및 한계

#### ① 해결하고자 하는 문제

디코딩 시 매 스텝마다 전체 KV 캐시를 읽어야 하는 $O(N)$ 메모리 대역폭 병목 문제. 기존 방법들도 경량 스캔을 통해 상수 인자를 줄이지만 복잡도는 여전히 $O(N)$.

#### ② 제안 방법: Declarative Attention (DA)

**핵심 아이디어:** 모델의 CoT 출력 스트림에서 어텐션 마스크를 직접 파싱함.

**세 가지 모드:**
- `<global>`: 전체 컨텍스트 세그먼트 어텐션 (내비게이션용)
- `<focus magic_chunks="K">`: 지정된 세그먼트 K만 어텐션 (정보 추출용)
- `<local>`: 컨텍스트 세그먼트 없음, 응답 내용만 어텐션 (추론/합성용)

**어텐션 메트릭 수식 (Section 7, Metrics):**

$$\text{Attended Tokens} = \sum_{t} \text{attended}(t)$$

- $t$: 디코딩 스텝 인덱스
- $\text{attended}(t)$: 스텝 $t$에서 어텐션된 KV 포지션 수

**Roofline Wall-time 수식 (Section 3, Appendix C.4):**

$$T_{\text{roofline}} = \frac{\text{work}}{R \times u}$$

- $\text{work}$: 계산량(FLOPs) 또는 메모리 접근량(bytes)
- $R$: 하드웨어 피크 성능 (Peak FLOPS 또는 Peak BW)
- $u$: 달성 활용률 (MFU 또는 MBU)

**FFN Roofline Wall-time:**

$$T_{\text{FFN}} = \frac{\text{FLOPs}}{\text{Peak FLOPS} \times \text{MFU}}$$

- MFU(Model FLOPs Utilization): 피크 컴퓨팅 대비 실제 달성 비율 (채택값: 40%)

**Attention Roofline Wall-time:**

$$T_{\text{attn}} = \frac{\text{KV bytes}}{\text{Peak BW} \times \text{MBU}}$$

- MBU(Model Bandwidth Utilization): 피크 메모리 대역폭 대비 실제 달성 비율 (채택값: 70%)

**Decode Wall-time 분해 (Appendix C.8):**

$$T_{\text{decode}} = T_{\text{matmul}} + T_{\text{global}} + T_{\text{local}}$$

$$T_{\text{matmul}} = \frac{2PD}{C}, \quad T_{\text{global}} = \frac{A \cdot b_{\text{kv}}}{\text{BW}_{\text{eff}}}, \quad T_{\text{local}} = \frac{D \cdot s_{\text{local}}}{\text{BW}_{\text{eff}}}$$

- $P$: 활성 파라미터 수
- $D$: 디코딩 스텝 수
- $C$: 유효 컴퓨팅 처리량 ($= \text{Peak FLOPS} \times \text{MFU}$)
- $A$: 전체 어텐션된 토큰 포지션 수 (DA 마스크가 줄이는 값)
- $b_{\text{kv}}$: 토큰당 KV 바이트 수
- $\text{BW}_{\text{eff}}$: 유효 메모리 대역폭 ($= \text{Peak BW} \times \text{MBU}$)
- $s_{\text{local}}$: 효율적 레이어(SWA/GDN)의 스텝당 고정 메모리 읽기량

**1M 토큰 컨텍스트에서의 각 항목 분해 (Appendix C.10):**

$$T_{\text{global}} = \frac{b_{\text{kv}} N}{\text{BW}_{\text{eff}}}, \quad T_{\text{matmul}} = \frac{2P}{C}, \quad T_{\text{local}} = \frac{s_{\text{local}}}{\text{BW}_{\text{eff}}}$$

- $N$: 컨텍스트 토큰 수

> 📌 **용어 설명**
> - **Roofline Model:** 하드웨어의 피크 성능 한계를 기준으로 연산 시간을 추정하는 분석 모델. 메모리 한계 vs 컴퓨팅 한계를 구분함.
> - **MFU (Model FLOPs Utilization):** 이론적 피크 FLOPs 대비 실제 달성 FLOPs의 비율.
> - **MBU (Model Bandwidth Utilization):** 이론적 피크 메모리 대역폭 대비 실제 달성 비율.

#### ③ 모델 구조 (프로토콜 메커니즘)

```
[프롬프트 구조]
A: System Instruction (Attention Sink 역할, 고정 16토큰)
B1~BN: Context Segments (Magic Chunks, ~2K 토큰씩 분할)
C: Question
D: DA Instruction (모드 사용법 안내)

[DA State Machine]
- 기본 상태: global
- <focus magic_chunks="K"> 파싱 → focus 모드 전환
- </focus> 파싱 → global 복귀
- <local> 파싱 → local 모드 전환

[KV 블록 테이블 재구성]
- vLLM attention metadata builder에 hook 추가
- 블록 단위(16~32토큰)로 정렬하여 마스크 적용
- FlashAttention 등 기존 커널 수정 없이 동작
```

> 📌 **용어 설명**
> - **Attention Sink:** 어텐션이 집중되는 초기 토큰들. StreamingLLM에서 처음 관찰된 현상으로, 의미 없는 고정 토큰이어도 어텐션 안정화에 기여함.
> - **vLLM:** 대규모 LLM 서빙을 위한 오픈소스 추론 엔진. PagedAttention으로 KV 캐시를 효율적으로 관리함.
> - **FlashAttention:** IO 인식 알고리즘으로 어텐션 연산의 메모리 효율을 크게 개선한 커널.

#### ④ 성능 향상 및 한계

**성능 향상:**

| 모델 | 어텐션 토큰 감소 | 정확도 변화 | 이론적 Wall-time |
|------|-----------------|------------|-----------------|
| Gemma-4-31B | 52.0% (13.43M→6.45M) | −1.27pp | $0.71\times$ |
| Qwen-3.6-27B | 31.1% (22.54M→15.52M) | −2.75pp | $0.77\times$ |

**주요 한계 (Section 8, p.13-14):**
1. **비최적 제로샷 전략:** DA는 vanilla 대비 ~30-35% 더 많은 디코딩 스텝 실행
2. **인위적 세그먼트:** 고정 벤치마크를 위해 컨텍스트를 인위적으로 분할
3. **thinking 모드 미지원:** 내부 추론(thinking trace) 내에서 DA 프로토콜 미작동
4. **글로벌 모드 비용:** 글로벌 모드는 여전히 전체 어텐션 비용 발생, DA 어텐션 토큰의 80% 이상 차지
5. **세그멘테이션에 의한 정보 파괴:** 전체 세그먼트를 아우르는 집계 태스크에서 정확도 급락 (Table 10)

---

## 3. 각 주장 위치 표시

| 주장 | 위치 |
|------|------|
| DA의 평균 어텐션 토큰 감소 52.0% (Gemma), 31.1% (Qwen) | Abstract, Table 2 (p.8) |
| 정확도 손실 1.27pp, 2.75pp | Abstract, Table 2 (p.8), Section 5.1 (p.7) |
| DA-nm과의 비교로 마스크가 효율성 원천임 확인 | Section 5.1 (p.7-8) |
| 모델 크기별 상대 정확도 29%→99% | Figure 3 (p.9), Section 5.2 (p.8) |
| 컨텍스트 길이 증가 시 절대 절감량 -1M→-21M | Figure 4 (p.10), Section 5.3 (p.10) |
| Wall-clock 0.71×, 0.77× 추정 | Table 3 (p.11), Section 5.4 (p.10) |
| focus success rate 58%→99% | Figure 6 (p.12), Section 6.2 (p.11) |
| global 모드 토큰 비중 27%, focus+local 73% | Figure 5 (p.12), Section 6.1 (p.11) |

---

## 4. 저자 보고 결과 vs. 해석 분리

### 연구 주제

**저자 보고:** "DA는 모델이 CoT 내에서 어텐션 범위를 선언하게 함으로써, 추론 엔진이 텍스트에서 직접 마스크를 읽어 KV 캐시 읽기를 줄인다." (Abstract, p.1)

**내 해석:** 이는 어텐션 희소화를 모델의 언어적 추론 능력으로 위임하는 시도로, 기존 내부 활성화 기반 방법과 근본적으로 다른 패러다임이다. 그러나 이 위임이 얼마나 정확한지—즉, 모델이 실제로 필요한 토큰을 항상 올바르게 지목하는지—는 이 논문에서 직접 검증하지 않는다.

### 방법

**저자 보고:** 컨텍스트를 ~2K 토큰 단위로 분할하여 Magic Chunk로 명명하고, 모델 네이티브 tool-use 형식으로 제공. DA state machine이 태그를 파싱하여 vLLM의 KV 블록 테이블을 재구성. (Section 2, p.3-5)

**내 해석:** Magic Chunk의 크기(~2K 토큰)는 고정 휴리스틱이며, 태스크나 컨텍스트 구조에 따라 최적값이 달라질 수 있다. 또한 tool-use 형식은 해당 형식으로 post-training된 모델에 유리하게 작용할 수 있어 일반화 가능성에 의문이 남는다.

### 결과

**저자 보고 (Table 2, p.8):**
- Gemma-4-31B: 87.01% → 85.74% (−1.27pp), 13.43M → 6.45M 토큰 (−52.0%)
- Qwen-3.6-27B: 85.31% → 82.56% (−2.75pp), 22.54M → 15.52M 토큰 (−31.1%)
- Wall-clock 추정: Gemma $0.71\times$, Qwen $0.77\times$ (Table 3, p.11)

**내 해석:**
- Wall-clock 수치는 측정값이 아닌 **이론적 상한 추정치(ceiling estimate)**임을 저자들도 명시 (Section 5.4, p.11). 실제 서빙 환경에서는 배치 크기, 프리필-디코딩 분리 여부, 하드웨어 실효율에 따라 크게 달라질 수 있음.
- "modest accuracy drop"이라는 표현은 평균 기준이며, 일부 태스크(multidoc_qa: −7.1pp on Qwen)에서는 상당한 손실 발생.

---

## 5. 통계적으로 취약한 부분 및 비교 불가능한 수치

| 항목 | 문제점 |
|------|--------|
| **Wall-clock time 수치 (0.71×, 0.77×)** | ⚠️ 실측값 아닌 Roofline 이론 추정치. MFU=40%, MBU=70% 가정에 민감 (저자들도 "ceiling estimate"이라 명시, Appendix C.8) |
| **128개 샘플 per source** | ⚠️ 소규모 샘플. 일부 소스(code_repo: 107개)는 더 작음. 신뢰구간 미제공 |
| **태스크 간 단순 평균** | ⚠️ 컨텍스트 길이(6.4K~1071K), 태스크 유형이 매우 이질적임에도 동일 가중치로 평균 |
| **LLM judge 평가 (Qwen-3.5-4B)** | 🔍 frontier judge와 Pearson r=0.992의 높은 상관이나, judge 자체의 편향 가능성 완전 배제 불가 |
| **code_repo 결과 (Attended Tokens)** | ⚠️ Vanilla Qwen: 134.00M, DAnm: 162.25M, DA: 81.96M — 이상치적 수치, 컨텍스트 길이 평균 1071K로 극단적 |
| **Gemma-4-12B 관련 수치** | ⚠️ 비종료(non-terminating) 응답 ~6%로 인해 attended token 수치가 인플레이션됨 (Figure 3b 주석) |
| **이론적 글로벌 어텐션 비중** | 🔍 Qwen-3.5-397B-A17B에서 1M 토큰 어텐션 vs FFN 비용 비교는 직접 실험 대상 모델과 다른 모델 사용 |

---

## 6. 논문이 답하지 않는 질문

1. **실제 wall-clock time 측정 결과는?** 논문은 Roofline 이론 추정만 제공하며, 실제 B200 GPU에서의 end-to-end 지연 측정은 없음.

2. **Magic Chunk 크기(~2K 토큰)의 최적화 근거는?** 2K 선택이 왜 최적인지, 다른 크기 대비 성능 비교가 없음.

3. **DA의 선택이 실제로 "올바른" 토큰을 포함하는가?** 모델이 선언한 focus 영역이 실제 어텐션 필요 영역과 얼마나 일치하는지 분석 없음.

4. **Multi-turn/Agentic 환경에서의 실제 성능은?** 저자들이 가능성을 언급하나 실험적 검증 없음.

5. **다른 모델 패밀리(GPT, Claude, LLaMA)에서의 일반화 가능성은?** Gemma와 Qwen 두 패밀리만 평가.

6. **Thinking mode에서 DA를 적용하는 구체적 방법론은?** 한계로 언급하지만 해결 방향만 제시.

7. **DA의 mode selection이 최적인가?** 모델이 zero-shot 프롬프트로 결정하는 mode 순서/빈도가 태스크별로 최적화되지 않음.

8. **Post-training(SFT/RL)으로 실제 얼마나 개선되나?** 이론적 여지를 언급하나 실험 없음.

9. **Global mode에서의 비용을 줄이는 방법론?** In-context index 아이디어 제안에 그침.

10. **벤치마크 오염(contamination) 가능성은?** Synthetic QA를 Gemini-3-Flash로 생성했는데, 평가 모델과의 데이터 오염 여부 불명확.

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.1) — DA 전체 개요

**내용:** 25,466 토큰 프롬프트에서 DA가 동작하는 예시. R1(global), R2(focus), R3(local) 각 단계에서 어텐션 마스크가 어떻게 달라지는지 시각화.

**해석:**
- R1(global): 25,466 토큰 전체 어텐션 (0% 감소)
- R2(focus, chunk 1): 3,435 토큰만 어텐션 (**86.5% 감소**)
- R3(local): 1,124 토큰만 어텐션 (**95.6% 감소**)
- 스캐폴드(A, C, D)는 항상 어텐션 유지, 컨텍스트 세그먼트(B1~B12)는 모드에 따라 마스킹

**시사점:** 대부분의 추론 스텝이 focus/local 모드에서 이루어진다면, 전체 어텐션 비용이 급격히 줄어든다. 이 예시에서 단일 응답으로 91% 이상의 어텐션 감소를 달성함.

---

### Figure 2 (p.9) — 효율성의 원천 분석

**내용:** Vanilla, DA, DA-nm의 (a) 상대 정확도, (b) 상대 디코딩 스텝, (c) 상대 어텐션 토큰 비교.

**해석:**
- **(a) 정확도:** 세 방법 모두 거의 유사 (DA는 1-3pp 하락), 포맷 변환(DA-nm) 자체의 정확도 비용 거의 없음
- **(b) 디코딩 스텝:** DA와 DA-nm 모두 vanilla 대비 ~15-35% 증가 — DA 프로토콜의 순비용
- **(c) 어텐션 토큰:** DA는 48%(Gemma), 69%(Qwen)로 크게 감소, DA-nm은 오히려 166%(Gemma), 129%(Qwen)로 증가

**핵심 메시지:** DA의 효율성은 더 짧은 생성에서 오는 것이 **아니라** 어텐션 마스크에서 온다. DA-nm이 오버헤드를 발생시키는 것은 더 많은 디코딩 스텝을 full attention으로 실행하기 때문.

---

### Figure 3 (p.9) — 모델 크기 스케일링

**내용:** 6개 모델(Gemma 4: E4B/12B/31B, Qwen 3.5/3.6: 4B/9B/27B)에서 (a) 상대 정확도, (b) 상대 어텐션 토큰.

**해석:**
- **(a) 정확도 스케일링:** Gemma 29%(E4B) → 91%(12B) → 99%(31B), Qwen 64%(4B) → 89%(9B) → 97%(27B). **모노토닉 증가**로 스케일링 법칙과 일치
- **(b) 토큰 절감:** 크기와 무관하게 roughly 46-69% 범위. Gemma-4-12B의 이상치(183%)는 비종료 응답 아티팩트로, 제외 시 98%
- **Gemma-4-E4B의 실패(29%):** Focus success rate 58%로 프로토콜 자체를 따르지 못하는 것이 주원인

**시사점:** 정확도 gap은 스케일로 해결 가능하지만, 토큰 절감 효율은 이미 작은 모델에서도 안정적 — DA의 효율성 메커니즘이 모델 크기와 독립적임을 시사.

> 📌 **용어 설명**
> - **Focus success rate:** `<focus>` 태그 내에 유효한 chunk 번호를 올바르게 참조하는 비율. 프로토콜 준수도의 지표.

---

### Figure 4 (p.10) — 컨텍스트 길이 스케일링

**내용:** Gemma-4-31B에서 컨텍스트 길이 구간별로 (a) 상대 정확도, (b) 절대 어텐션 토큰 delta.

**해석:**
- **(a) 정확도:** 32K 이하에서 vanilla와 거의 동일(~1pp 내), 64-256K에서 약 96%로 소폭 하락. DA-nm 라인이 이 하락을 보이지 않으므로 **마스크가 원인** (컨텍스트 포맷이 아님)
- **(b) 절대 절감량:** 짧은 컨텍스트 -1M 토큰 → 긴 컨텍스트 약 -21M 토큰. DA-nm은 반대로 양의 오버헤드 증가
- **핵심 패턴:** DA의 per-step 마스킹이 ~50%의 일정 비율을 절감하므로, 컨텍스트가 길수록 절대 이득이 선형 증가

**시사점:** DA는 정확히 가장 비싼(긴 컨텍스트) 상황에서 가장 큰 절감을 제공하는 이상적인 특성을 보임.

---

### Figure 5 (p.12) — 모드별 효율성 분석

**내용:** Gemma-4-31B에서 컨텍스트 길이별 (a) 모드별 토큰 비중, (b) focus/local 모드의 per-token 어텐션 절감률.

**해석:**
- **(a) 모드 비중:** global 27%, focus 38%, local 35% (<32K 기준). 길이 증가 시 global 비중이 45%(>128K)까지 증가 — 이것이 긴 컨텍스트에서 절감 효율이 다소 낮아지는 원인
- **(b) 절감 효율:** focus 76-96%, local 88-99%의 per-token 절감. 길이가 길수록 두 모드 모두 효율 향상
- `<global>` 모드는 정의상 0% 절감

**시사점:** DA의 병목은 global 모드. 저자들이 제안하는 in-context index나 경량 sparse attention과의 결합이 global 모드 비용 감소에 직접 도움이 될 수 있음. 또한 post-training으로 모델이 global 모드 사용을 줄이도록 학습시키면 전체 효율이 크게 향상될 수 있음.

---

## 8. 결론: 시사점, 후속 연구 계획, 추가 방향 제시

### 저자들이 제시한 시사점 (Section 7, p.12)

1. LLM은 텍스트로 읽을 수 있는 형태로 자신의 어텐션 계획을 표현할 수 있음
2. 선택적 어텐션이 네트워크 내부에서 추론되는 패턴이 아닌, 모델이 명시적으로 선언하는 패턴이 될 수 있음
3. 이 접근은 해석 가능성(legibility)과 효율성을 동시에 달성

### 저자들이 제시한 후속 연구 계획 (Section 8, p.13-14)

| 항목 | 방향 |
|------|------|
| Post-training (SFT/RLVR) | DA 프로토콜에 최적화된 응답 패턴 학습 |
| Thinking mode 통합 | tool declaration 형식으로 thinking trace 내 DA 적용 |
| In-context index | Global 모드를 위한 압축된 세그먼트 인덱스 활용 |
| Agentic 설정 | 자연적으로 구조화된 컨텍스트(tool calls, user turns)에서 평가 |
| Speculative decoding 결합 | DA + speculative decoding의 상보적 결합 |
| KV cache offloading | 선언된 out-of-focus 세그먼트를 호스트 메모리로 오프로드 |

---

### 8-1. 모델 일반화 성능 향상 가능성

**현재 상황:**
- Zero-shot DA의 정확도 gap은 모델 크기에 따라 단조 감소: Gemma E4B 29% → 31B 99% (Figure 3a)
- 그러나 소규모 모델(E4B)은 프로토콜 자체를 따르지 못하는 기본 능력 부족이 주원인 (focus success 58%)

**일반화 성능 향상을 위한 경로:**

**① Supervised Fine-Tuning (SFT):**
DA 프로토콜에 맞게 최적화된 reasoning trace를 포함한 데이터로 fine-tuning하면, 소규모 모델에서도 프로토콜 준수율이 크게 향상될 수 있다. 저자들은 이를 명시적으로 언급하며 (Section 8.2), vanilla CoT가 SFT를 통해 크게 개선된 것처럼 DA도 유사한 경로를 따를 것으로 예상한다.

**② Reinforcement Learning (RLVR):**
정확도와 어텐션 효율 모두를 보상으로 설계한 RL이 가능하다. DeepSeek-R1의 방식처럼, 정답률과 attended token 수의 트레이드오프를 보상 함수에 명시적으로 포함:

$$r = \alpha \cdot \text{accuracy} - \beta \cdot \frac{\text{attended tokens}}{\text{vanilla attende tokens}}$$

- $\alpha, \beta$: 정확도와 효율성 간의 균형 하이퍼파라미터

**③ 태스크 특화 generalization:**
현재 DA는 retrieval-oriented QA에서 강하고, multi-span reasoning에서 약하다 (Gemma: single-span −0.78pp vs multi-span −2.28pp). 태스크 유형별로 최적 mode selection 전략을 학습하면 다양한 태스크로의 일반화가 개선될 수 있다.

**④ 세그먼트 granularity 적응:**
2K 토큰 고정 분할 대신, 태스크나 문서 구조에 따라 세그먼트 크기를 동적으로 조절하면 summarization, structured data 등에서 발생하는 segmentation failure를 줄일 수 있다.

**⑤ Cross-architecture 일반화:**
현재 Gemma와 Qwen만 평가. MLA를 사용하는 DeepSeek 계열이나 purely dense architecture에서의 성능은 미검증. KV 구조가 다른 모델에서는 $b_{\text{kv}}$ 값이 달라 wall-clock 이득도 달라진다.

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

#### 비교 연구 맵

| 연구 | 방식 | DA와의 관계 | 주요 차이점 |
|------|------|------------|------------|
| **H2O** (Zhang et al., 2023, NeurIPS) | 어텐션 크기 기반 KV 영구 eviction | 대조적 | eviction은 비가역적; DA는 마스킹으로 가역적 유지 |
| **SnapKV** (Li et al., 2024, NeurIPS) | 관찰된 어텐션 통계로 KV 선택 | 대조적 | 활성화 기반 선택; DA는 텍스트 기반 선언 |
| **Quest** (Tang et al., 2024, ICML) | per-page key bound로 KV 스캔 | 보완적 | $O(N)$ 스캔 여전히 필요; DA global 모드에 결합 가능 |
| **Self-Selected Attention Span (SSAS)** (Jin et al., 2024) | 태스크별 훈련으로 span 선택 | 선구자 | 태스크별 fine-tuning, 2K 컨텍스트; DA는 zero-shot, 244K |
| **Native Sparse Attention** (Yuan et al., 2025, ACL) | 모델에 block-sparse selection을 pre-training에서 학습 | 대조적/보완적 | 사전 훈련 필요; DA는 기성 모델에 즉시 적용 |
| **System 2 Attention** (Weston & Sukhbaatar, 2023) | 모델이 입력을 재생성하여 어텐션 결정 | 개념적 선구자 | 입력 전체 재생성 비용; DA는 마스킹으로 더 효율적 |
| **DeepSeek V3.2/V4 Sparse Attention** (DeepSeek-AI, 2025b, 2026) | 경량 indexer로 top-k 선택 ( $O(N)$ ) | 보완적 | $O(N)$ 스캔 유지; DA의 focus 해석은 $O(1)$ |
| **ThinKV** (Ramachandran et al., 2025) | 어텐션 희소성 패턴으로 reasoning phase 감지 후 eviction | 대조적 | 잠재 활성화 기반; DA는 텍스트 기반, 반대 방향 |
| **SpotAttention** (Ahmad & Yun, 2026) | frozen backbone에 block-sparse routing distillation | 보완적 | 추가 학습 필요; DA는 훈련 불필요 |

#### 이 논문이 앞으로의 연구에 미치는 영향

**1. 새로운 희소 어텐션 패러다임 확립:**
기존 "내부 활성화로 마스크 예측" 패러다임과 달리 "텍스트로 마스크 선언"이라는 새 방향을 제시. 이는 interpretable AI와 efficient inference를 동시에 추구하는 연구 흐름에 부합.

**2. Model-controlled inference의 확장:**
ReAct, Self-RAG 등 모델이 행동을 선언하는 흐름에서, 어텐션이라는 내부 연산도 선언 대상이 될 수 있음을 보여줌. 향후 모델이 메모리 관리, 계산 할당 등을 더 넓게 제어하는 연구로 이어질 가능성.

**3. LLM 평가의 새로운 차원:**
"모델이 자신의 추론에 필요한 정보를 얼마나 잘 파악하는가"라는 metacognitive 능력을 측정하는 새 벤치마크 개발이 필요해짐.

**4. Post-training 최적화의 새 목표:**
어텐션 효율성을 보상으로 하는 RLVR이 정확도 최적화와 함께 연구될 가능성. 이는 inference 비용 인식 모델 학습이라는 새 분야를 열 수 있음.

#### 앞으로 연구 시 고려할 점

1. **실측 실험의 필요성:** Roofline 이론치가 아닌 실제 서빙 스택에서의 측정 필요. 배치 크기, 하드웨어 특성, 선점/비선점 스케줄링에 따른 실제 지연 변동을 측정해야 함.

2. **Thinking mode 통합:** 현재 thinking mode 비활성화는 추론 집약적 태스크에서 DA의 실용성을 제한. Thinking trace 내에서 DA가 동작하도록 하는 방법론 개발이 시급.

3. **더 넓은 태스크 커버리지:** 현재 벤치마크는 retrieval-oriented QA에 편중. 코드 생성, 수학 추론, 창작 등 diverse task에서 DA 성능 검증 필요.

4. **Semantic 세그멘테이션:** 현재의 길이 기반 분할 대신 의미 기반(semantic) 분할이 특히 table, code, structured data 처리에서 중요. LLM 기반 자동 세그멘테이션 연구 필요.

5. **Multi-modal 확장:** 텍스트 외 이미지, 오디오, 비디오 토큰에서도 유사한 "선언적 어텐션" 메커니즘이 적용 가능한지 탐색.

6. **Privacy & Security:** 어텐션 패턴이 텍스트로 노출되므로, 모델의 추론 전략이 외부에 드러남. 이는 해석 가능성의 장점이지만, 동시에 adversarial attack의 새로운 표면이 될 수 있음.

7. **Consistency 검증:** 동일 질문에 대해 DA가 항상 같은 chunk를 선택하는지, 아니면 stochastic한 변동이 있는지의 분석이 신뢰성 평가에 중요.

---

## 참고자료

**논문 원문:**
- Ho, N., Ahmad, H., Koh, W., Yun, S.-Y., Schuster, T., & dos Santos, C. N. (2026). *Language Models Can Control Their Own Attention*. arXiv:2609.02737v1.

**논문 내 주요 참고문헌:**
- Wei, J., et al. (2022). Chain-of-Thought prompting elicits reasoning in large language models. *NeurIPS 2022*.
- Kwon, W., et al. (2023). Efficient memory management for large language model serving with PagedAttention. *SOSP 2023*.
- Dao, T., et al. (2022). FlashAttention: Fast and memory-efficient exact attention with IO-awareness. *NeurIPS 2022*.
- Tang, J., et al. (2024). QUEST: Query-aware sparsity for efficient long-context LLM inference. *ICML 2024*.
- Jin, T., et al. (2024). Self-selected attention span for accelerating large language model inference. arXiv:2404.09336.
- Zhang, Z., et al. (2023). H2O: Heavy-hitter oracle for efficient generative inference of large language models. *NeurIPS 2023*.
- Yuan, J., et al. (2025). Native sparse attention: Hardware-aligned and natively trainable sparse attention. *ACL 2025*.
- Williams, S., Waterman, A., & Patterson, D. (2009). Roofline: An insightful visual performance model for multicore architectures. *Communications of the ACM*.
- Weston, J., & Sukhbaatar, S. (2023). System 2 Attention (is something you might need too). arXiv:2311.11829.
- DeepSeek-AI. (2025b). DeepSeek-V3.2. arXiv:2512.02556.
- Hsieh, C.-P., et al. (2024). RULER: What's the real context size of your long-context language models? *COLM 2024*.
- Bai, Y., et al. (2024). LongBench. *ACL 2024*; (2025). LongBench v2. *ACL 2025*.
- Li, J., et al. (2024). LooGLE. *ACL 2024*.
- Shaham, U., et al. (2023). ZeroSCROLLS. *EMNLP Findings 2023*.
- Agarwal, M., et al. (2023). LLM inference performance engineering: Best practices. *Databricks Mosaic Research Blog*.
- Ramachandran, A., et al. (2025). ThinKV. arXiv:2510.01290.
