# Model-Distributed Inference for Large Language Models at the Edge

## Executive Summary

**MDI-LLM**은 저성능 엣지 기기 위에서 상태-of-the-art 대규모 언어모델(LLM)의 배포를 가능하게 하는 혁신적인 프레임워크이다. 모델을 여러 파티션으로 분할하고, 각 파티션을 서로 다른 엣지 장치에 할당하는 **모델 병렬화(Model Parallelism)** 접근법을 채택한다. 핵심 기여는 LLM의 자기회귀 특성을 처리하기 위한 **재귀적 파이프라인 병렬화(Recurrent Pipeline Parallelism, RPP)** 기술이다. 본 논문은 KV 캐싱과 그룹화된 쿼리 주의(GQA)를 분산 환경에 적응시켜, 저비용 하드웨어에서도 효율적인 LLM 추론을 가능하게 한다.

---

## 1. 핵심 주장과 주요 기여

### 1.1 핵심 주장
MDI-LLM의 근본적인 주장은 **물리적으로 분산된 저성능 엣지 기기들의 네트워크가 협력하여 개별 장치의 메모리 용량을 초과하는 LLM 추론을 수행할 수 있다**는 것이다. 이는 다음 세 가지 주요 관찰에 기초한다:

1. **클라우드 의존성 문제**: 기존 LLM 배포는 클라우드 중심이며, 프라이버시, 가용성, 고비용 문제 야기
2. **병렬화 선택의 문제**: 텐서 병렬화는 높은 통신 오버헤드로 인해 엣지에 부적합; 모델 병렬화는 전통적 CNN/RNN에 적용되었으나 자기회귀 LLM의 특수성 미반영
3. **자기회귀 특성**: LLM은 출력을 입력으로 되먹이는 자기회귀 모델로, 기존 파이프라인 병렬화가 비효율적

### 1.2 주요 기여
1. **재귀적 파이프라인 병렬화(RPP)**: 자기회귀 LLM을 위해 특별히 설계된 파이프라인 병렬화 기법
   - 여러 텍스트 샘플을 동시에 처리하여 생성 시간 단축
   - 장치 유휴시간 최소화
   - 기존 파이프라인 병렬화보다 자기회귀 특성에 최적화

2. **KV 캐싱의 분산 적응**: 회전하는 KV 캐시 메커니즘
   - 각 샘플의 KV 행렬을 별도로 저장
   - 메시지 크기 감소 (전체 컨텍스트 ×임베딩 차원 → 마지막 토큰만)
   - 처리 속도 향상 및 메모리 소비 감소

3. **GQA 통합**: 그룹화된 쿼리 주의(Grouped Query Attention)의 분산 활용
   - 중앙화된 설정과 유사하게 성능 유지
   - 엣지 환경에서 효율성 증진

---

## 2. 해결하는 문제

### 2.1 기술적 문제
| 문제 | 전통적 해결책 | MDI-LLM의 해결책 |
|------|-------------|-----------------|
| **메모리 부족** | 모델 양자화, 단일 기기 | 모델 분할, 다중 기기 협력 |
| **처리 속도 저하** | 강력한 하드웨어 (비용 증가) | 병렬 처리로 시간 단축 |
| **자기회귀 특성 처리** | 기존 파이프라인 병렬화 (기기 유휴) | 재귀적 파이프라인 병렬화 |
| **통신 오버헤드** | 텐서 병렬화 (높은 동기화 비용) | 모델 병렬화 (중간 활성화만 전송) |
| **KV 캐시 관리** | 각 기기가 전체 캐시 유지 | 분산 회전 캐시 |

### 2.2 실제 애플리케이션 맥락
- **프라이버시**: 입력 프롬프트가 사용자 기기에만 유지
- **레이턴시**: 클라우드 왕복 시간 제거
- **네트워크 비용**: 중간 활성화만 전송 (대역폭 효율)
- **가용성**: 인터넷 연결 없이도 작동

---

## 3. 제안 방법: 기술 상세 분석

### 3.1 시스템 아키텍처

```
Ring Overlay Network:
┌─────────────┐    ┌─────────────┐
│ Starter     │───→│ Secondary   │───→ ...
│ (Master)    │←───│ Node 1      │←───
└─────────────┘    └─────────────┘
```

**장치 역할**:
- **Starter 노드**: 애플리케이션 입구점, 프롬프트 수신, 생성 텍스트 반환, 조정 역할
- **Secondary 노드**: 워커, 이전 노드로부터 입력 수신, 로컬 모델 청크 처리, 다음 노드에 전송

### 3.2 모델 분할 전략

$$\text{Model Partition} = \{\text{Initial Layers}, \text{Transformer Blocks}_{1..n}, \text{Output Layers}\}$$

**구조**:
- **Starter 노드**: 초기 레이어 + 처음 몇 트랜스포머 블록 + 최종 레이어
- **Secondary 노드**: 트랜스포머 블록 (장치의 계산 능력에 비례)

**목표**: 통신 병목 제거, 균형잡힌 워크로드 분배

### 3.3 재귀적 파이프라인 병렬화 (RPP) 알고리즘

**전통 파이프라인 병렬화의 문제점**:
```
시간 0: Device1(sample1) → Device2(idle) → Device3(idle)
시간 1: Device1(idle) ← Device2(sample1) → Device3(idle)
시간 2: Device1(idle) ← Device2(idle) ← Device3(sample1)
→ 많은 유휴 시간
```

**RPP 해결책**:
```
시간 0: Device1(sample1) → Device2(idle) → Device3(idle)
시간 1: Device1(sample2) → Device2(sample1) → Device3(idle)
시간 2: Device1(sample3) → Device2(sample2) → Device3(sample1)
→ 모든 장치가 동시에 계산
```

**동작 원리**:
- 여러 샘플(텍스트 조각)을 순차적으로 처리
- 각 기기가 처리를 마친 직후 다음 샘플을 받음
- n개 샘플과 n개 노드: 파이프라인 충전 후 유휴시간 0

### 3.4 회전하는 KV 캐시 메커니즘

**전통적 KV 캐싱** (중앙화):
- 모든 토큰의 K, V 행렬을 저장
- 메시지 크기: (context_length) × (embedding_dimension)
- 예: 2048 × 4096 × 2 = 16MB

**분산 RPP KV 캐싱**:
```python
# 각 샘플별 캐시 분리 저장
KV_caches = {
    sample_0: [cached_K[0], cached_V[0]],
    sample_1: [cached_K[1], cached_V[1]],
    ...
}

# 샘플 전환 시 활성 캐시 교체
for sample_n in samples:
    activate_cache(KV_caches[sample_n])
    forward_pass(sample_n)  # 마지막 토큰만 처리
    cache_new_token(sample_n)
```

**이점**:
- 메시지 크기: (embedding_dimension) 만 전송 (기하급수적 감소)
- 예: 4096 차원 = ~16KB (기존 16MB에서 1000배 감소)
- K, V 재계산 불필요 (캐시된 값 사용)

### 3.5 수식 표현

**Attention 블록 (분산)**:

$$H^l_{attn} = \text{allreduce}\left(\text{softmax}\left(\frac{Q_{h,l}(K_{h,l})^T}{\sqrt{d}}\right)V_{h,l}\right) + H^{l-1}_{ffn}$$

여기서:
- $Q_{h,l}, K_{h,l}, V_{h,l}$ = 기기 $h$에서 계산한 쿼리, 키, 값
- $d$ = 어텐션 헤드 차원
- $\text{allreduce}$ = 모든 기기에서 결과 집계

**FFN 블록 (분산)**:

$$H^l_{ffn} = \text{allreduce}\left(W^{h,l}_{down} \cdot (\sigma(W^{h,l}_{gate} \cdot H^l_{norm}) \odot (W^{h,l}_{up} \cdot H^l_{norm}))\right) + H^l_{attn}$$

여기서:
- $W^{h,l}\_{up}, W^{h,l}\_{gate}, W^{h,l}_{down}$ = 분할된 FFN 가중치
- $\odot$ = 원소곱
- $\sigma$ = SiLU 활성화 함수

**Starter 노드 알고리즘**:
```python
for iteration k in range(n_samples × n_tokens):
    sample_n = samples[k mod n_samples]
    
    if k < n_samples:
        # 첫 반복: 초기화
        initialize_KV_cache(sample_n)
        H_output = forward_input_layers(full_sample)
    else:
        # 이후 반복: 마지막 토큰만
        activate_cache(sample_n)
        H = forward_input_layers(last_token_embedding)
        token = sample_output_layers(H)
        append_token(sample_n, token)
    
    if k < n_samples × (n_tokens - 1):
        activate_cache(sample_n)
        output = forward_through_network(H)
        send_to_next_node(output)
```

---

## 4. 모델 구조

### 4.1 Llama 2 / Llama 3 기반 구현
- **아키텍처**: Decoder-only Transformer
- **특성**: RoPE (회전 위치 인코딩), GQA (그룹화 쿼리 주의)
- **테스트 모델**: 
  - NanoLlama (304M 파라미터)
  - TinyLlama Chat v1.0 (1.1B 파라미터)

### 4.2 분할 구성 예시

**2개 노드 (NanoLlama)**:
- Starter: 초기 5개 블록 + 처음 5개 변환 블록
- Secondary: 7개 변환 블록

**3개 노드 (NanoLlama)**:
- Starter: 초기 2개 블록 + 처음 2개 변환 블록
- Secondary-1: 5개 변환 블록
- Secondary-2: 5개 변환 블록

**프롬프트 처리 메커니즘**:
1. 프롬프트 토큰화
2. 임베딩 생성
3. 모든 기기에 초기 임베딩 브로드캐스트
4. 각 기기가 로컬 트랜스포머 블록 실행
5. 마지막 기기에서 Starter로 결과 전송
6. Starter가 최종 토큰 샘플링

---

## 5. 성능 향상 분석

### 5.1 토큰 생성 속도 (Token Generation Rate)

**실험 설정**:
- 모델: NanoLlama (304M 파라미터)
- 샘플: 3개, 각각 800 토큰 생성
- 하드웨어: Nvidia Jetson TX2 (1.33 TFLOPS)

**결과** (수치):
| 디바이스 수 | 생성 속도 (tokens/sec) | 개선도 |
|-----------|---------------------|----|
| 1 | ~1 | baseline |
| 2 | ~2.5 | 2.5배 |
| 3 | ~5 | 5배 |

**관찰**:
- 초기 트랜지언트: KV 캐시 및 내부 상태 초기화로 인한 느린 속도
- 정상 상태 도달 후: 선형적 성능 향상
- 네트워크 지터의 영향 관찰됨

### 5.2 메모리 사용량 감소

**NanoLlama (304M, 12 블록)**:
| 노드 수 | Node 1 (GB) | Node 2 (GB) | Node 3 (GB) | 평균 |
|--------|-----------|-----------|-----------|-----|
| 1 | 2.15 | - | - | 2.15 |
| 2 | 1.76 | 1.76 | - | 1.76 |
| 3 | 1.34 | 1.46 | 1.46 | 1.42 |

**TinyLlama (1.1B, 22 블록)**:
| 노드 수 | Node 1 (GB) | Node 2 (GB) | Node 3 (GB) | 평균 |
|--------|-----------|-----------|-----------|-----|
| 2 | 4.57 | 4.57 | - | 4.57 |
| 3 | 3.26 | 3.26 | 3.26 | 3.26 |

**감소 분석**:
- 2개 → 3개 노드: 1.76GB → 1.42GB (19.3% 감소)
- 수확 체감: 추가 기기 추가 시 감소량 감소
- 오버헤드: Python 라이브러리(450MB), HTTP 서버(150-200MB), 통신 채널(150-200MB)

### 5.3 한계점

**파이프라인 초기화 오버헤드**:
- 모든 노드가 첫 샘플을 처리할 때까지 기기 유휴
- 오버헤드 = $(n_{nodes} - 1) \times (processing\_time\_per\_node)$

**통신 지연**:
- TCP/IP 오버헤드 존재
- 네트워크 대역폭보다는 지연 시간이 병목
- 고지연 네트워크에서 성능 저하

---

## 6. 일반화 성능 향상 가능성

### 6.1 모델 일반화와의 관계

MDI-LLM이 모델의 출력 정확도(일반화 성능)에 미치는 영향은 **최소화**되도록 설계되었다:

#### 6.1.1 정확도 보존 메커니즘
```
전통적 추론:
Input → [Block1] → [Block2] → ... → [BlockN] → Output

MDI-LLM 추론:
Input → [Block1~k (device1)] → [Block(k+1)~m (device2)] → ... → Output
        (동일한 가중치, 동일한 계산순서)
```

**핵심**: 모델 가중치와 계산 순서가 동일하므로 이론적으로 정확도 손실 없음

#### 6.1.2 분산 KV 캐싱의 정확도 영향
- 각 샘플의 KV 캐시가 독립적으로 유지
- 캐시 교환은 순전히 기술적 메커니즘 (수학적 무해)
- 같은 중간값 사용 → 같은 출력

#### 6.1.3 GQA 적응의 영향
- GQA는 이미 Llama 2/3에 통합된 설계
- 분산 환경에서도 동일하게 작동
- 실제로는 추론 속도만 향상

### 6.2 장점: 일반화 성능 향상의 간접적 가능성

#### 6.2.1 더 큰 모델 사용 가능
```
단일 기기 제약:
- 메모리 한계 → 작은 모델만 가능 (예: 300M 파라미터)
- 작은 모델의 일반화 성능 ↓

MDI-LLM:
- 여러 기기 협력 → 큰 모델 가능 (예: 1.1B 파라미터)
- 더 큰 모델 = 더 나은 일반화 성능 ↑
```

**예시**: NanoLlama(304M)는 TinyLlama(1.1B)보다 많은 작업에서 성능 저하

#### 6.2.2 더 높은 정밀도 사용 가능
```
단일 기기 (메모리 제약):
- FP32 불가능 → FP16 또는 INT8 양자화 필요
- 양자화로 인한 정확도 손실 가능

MDI-LLM (메모리 분산):
- FP32 사용 가능 → 정확도 보존
```

#### 6.2.3 배치 처리 가능
- 단일 기기: 배치 크기 1 (메모리 부족)
- MDI-LLM: 여러 샘플 동시 처리 (RPP)
  - 더 안정적인 그래디언트 추정 (이론상)
  - 더 일관된 모델 거동

### 6.3 한계: 일반화 성능 저하 요인

#### 6.3.1 네트워크 지연의 영향
- 높은 지연 = 느린 추론 = 사용자 불만족
- 그러나 **정확도에는 직접 영향 없음**

#### 6.3.2 장치 간 불균형
```
균형 잡힌 분할:
Device1 처리 시간 = Device2 = Device3 = 동일
→ 최적 성능

불균형 분할:
Device1 처리 시간 = 10ms
Device2 처리 시간 = 50ms (병목)
→ 파이프라인 거품 발생
→ 속도만 저하, 정확도는 유지
```

#### 6.3.3 프라이버시의 트레이드오프
- Starter 노드만 프라이버시 보장
- Secondary 노드는 중간 활성화 관찰 가능
  - 그러나 원본 프롬프트/출력 보호
  - 모델 가중치 분산 (완전 역엔지니어링 방지)

---

## 7. 2020년 이후 관련 연구 비교 분석

### 7.1 병렬화 기법 비교

| 특성 | 데이터 병렬화 | 모델 병렬화 | 텐서 병렬화 | MDI-LLM (RPP) |
|-----|-----------|----------|----------|-----------|
| **원리** | 데이터 분산 | 모델 분할 | 레이어 내 분할 | 모델 분할 + RPP |
| **메모리 요구** | 각 기기 전체 모델 필요 | 낮음 | 낮음 | 낮음 ✓ |
| **통신 오버헤드** | 높음 (그래디언트 수집) | 중간 | 매우 높음 | 중간 ✓ |
| **엣지 환경 적합성** | ✗ (메모리 문제) | ✓ | ✗ (높은 동기화) | ✓✓ |
| **LLM 자기회귀 최적화** | - | 기본 | - | 전용 ✓ |
| **파이프라인 효율** | - | 낮음 (유휴) | - | 높음 ✓ |

### 7.2 엣지 분산 추론 시스템 비교

#### PipeEdge (2021, Hu et al.)
- **방식**: 기본 파이프라인 병렬화
- **목표**: 비전 모델 (ViT, BERT)
- **한계**: 자기회귀 LLM 미고려
- **성능**: 16개 기기에서 12.78배 향상
- **MDI-LLM vs PipeEdge**: 
  - PipeEdge는 이미지 처리에 최적화
  - MDI-LLM은 텍스트 생성에 최적화된 RPP 사용

#### TPI-LLM (2024, Li et al.)
- **방식**: 텐서 병렬화 + 슬라이딩 윈도우 메모리 스케줄러
- **대상**: 70B 모델, 저메모리 엣지 (4-16GB)
- **성능**: Llama 2-70B를 3.1GB 메모리에 수행
- **통신**: 링-기반 AllReduce → 링크 지연이 병목
- **MDI-LLM vs TPI-LLM**:
  
| 항목 | TPI-LLM | MDI-LLM |
|-----|---------|---------|
| 병렬화 | 텐서 (모든 기기 동시 계산) | 모델 (순차적 단계) |
| 통신 패턴 | AllReduce (복잡) | 링 (단순) |
| 자기회귀 최적화 | 없음 | 있음 (RPP) |
| 메모리 효율 | 우수 (3.1GB/70B) | 중간 |
| 처리량 | 중간 | 높음 (다중 샘플) |

#### DeTransformer (2024)
- **혁신**: 블록 병렬화 (레이어 디커플링)
- **통신 감소**: 2.81배 개선
- **MDI-LLM과 차이**:
  - DeTransformer: 레이어 내 병렬화
  - MDI-LLM: 레이어 간 순차적 분할
  - 통신 패턴 다름

#### Galaxy (2024, Ye et al.)
- **방식**: 텐서 + 시퀀스 병렬화 결합
- **장점**: 높은 처리량
- **문제**: 링-기반 AllReduce의 높은 링크 지연
- **MDI-LLM의 개선**: 
  - 더 단순한 파이프라인 구조
  - 자기회귀 특성 직접 활용

#### EdgeShard (2024)
- **특징**: 적응형 기기 선택 + 모델 분할 최적화
- **공통점**: 파이프라인 병렬화 기반
- **차이점**: 이질적 기기 자동 조율 없음 (MDI-LLM은 수동 분할)

### 7.3 시간별 발전 추이

```
2020-2021: 기초 연구
├─ PipeEdge (2021): 파이프라인 병렬화 소개
└─ Megatron (2019-2020): 클라우드 중심 병렬화

2022-2023: 엣지 최적화 시작
├─ 모델 양자화 연구 활발
├─ 클라우드-엣지 협력 모델
└─ 적응형 분할 알고리즘

2024: 분산 LLM 추론 폭발적 증가
├─ TPI-LLM: 텐서 병렬화 + 메모리 스케줄링
├─ Galaxy: 텐서 + 시퀀스 병렬화
├─ DeTransformer: 통신 효율화
├─ EdgeShard: 적응형 분할
└─ CE-CoLLM: 클라우드-엣지 협력

2025: 고도화 및 다양화
├─ LIME: 인터리빙 파이프라인
├─ SLED: 추측 디코딩
├─ MDI-LLM: 자기회귀 파이프라인 병렬화 (본 논문)
└─ 종합 설문: Edge LLM 전체 생태계 분석
```

### 7.4 핵심 기술 발전

**통신 효율성 진화**:
1. **초기 (2021-2022)**: 링-기반 AllReduce (높은 지연)
2. **중기 (2023-2024)**: 
   - 블록 병렬화 (레이어 디커플링)
   - 적응형 토폴로지
3. **최근 (2024-2025)**:
   - MDI-LLM: 모델 분할 + RPP (자기회귀 특화)
   - 인터리빙 파이프라인 (LIME)
   - 추측 디코딩 (SLED)

**메모리 효율 진화**:
1. **단일 기기 압축** (2020-2022): 양자화, 프루닝
2. **다중 기기 분산** (2023-2024): 슬라이딩 윈도우, 적응형 스케줄링
3. **하이브리드** (2024-2025): 압축 + 분산 결합

---

## 8. 논문의 한계와 미래 연구 방향

### 8.1 이론적 한계

**1. 이질적 기기 처리 부재**
```
현재: 모든 기기가 유사한 성능 가정
실제: 엣지 환경은 매우 이질적 (스마트폰, 라즈베리파이, NPU 등)

필요: 자동 로드 밸런싱, 적응형 분할
```

**2. 네트워크 동적 특성 미반영**
- 고정 대역폭, 지연 가정
- WiFi/4G의 변동성 미처리

**3. 장애 복구 메커니즘 부재**
- 기기 또는 링크 실패 시 처리 방안 없음
- 클라우드 폴백 필요

### 8.2 실제 배포 고려사항

**1. 프라이버시-성능 트레이드오프**
```
현재:
- Starter 노드: 완전 프라이버시
- Secondary 노드: 중간 활성화 노출

개선 필요:
- Secondary 노드의 중간 활성화 암호화 (동형 암호화 등)
- 추가 계산 비용 vs 프라이버시 균형
```

**2. 실시간 요구사항**
```
현재: 배치 처리 최적화
필요:
- 단일 요청 지연 시간 최소화
- SLA(Service Level Agreement) 보장
```

**3. 확장성 제약**
```
현재: 링 토폴로지 (O(n) 지연)
개선 방향:
- 트리 또는 메시 토폴로지
- 비동기 패턴
```

### 8.3 추천 미래 연구

#### 8.3.1 단기 (1-2년)
1. **적응형 분할 알고리즘**
   - 기기 성능 프로파일링
   - 자동 최적 분할점 결정

2. **동적 네트워크 대응**
   - 지연/대역폭 변화 감지
   - 실시간 재분할

3. **LLM 다양성 지원**
   - Llama뿐 아니라 GPT, BERT, Qwen 등
   - 다양한 아키텍처 호환성

#### 8.3.2 중기 (2-4년)
1. **동형 암호화 통합**
   - Secondary 노드의 중간값 보호
   - 계산 비용 vs 프라이버시 최적화

2. **강화학습 기반 분할**
   - 네트워크 상태에 따른 동적 최적화
   - 멀티 에이전트 강화학습

3. **프로토콜 레벨 최적화**
   - gRPC, QUIC 등 현대적 통신 프로토콜
   - 압축 + 암호화 통합

#### 8.3.3 장기 (4년 이상)
1. **이기종 컴퓨팅 지원**
   - CPU, GPU, NPU, TPU 통합
   - 작업 유형별 최적 실행 장소 선택

2. **연합 학습과의 통합**
   - 분산 추론 + 지역 적응학습
   - 프라이버시 보존 모델 개인화

3. **6G 네트워킹 활용**
   - 극저지연 네트워크 기대
   - 실시간 협력 추론

---

## 9. 결론 및 영향 평가

### 9.1 주요 성과

**기술적 기여**:
1. ✓ 자기회귀 LLM을 위한 최초 상세한 분산 추론 프레임워크
2. ✓ 재귀적 파이프라인 병렬화의 이론화 및 구현
3. ✓ 실제 엣지 환경 (Jetson TX2)에서의 동작 증명
4. ✓ KV 캐싱의 분산 적응으로 통신 비용 극감

**실무 가치**:
1. 저비용 하드웨어에서 큰 모델 실행 가능
2. 프라이버시 보호 (프롬프트 로컬 유지)
3. 인터넷 독립성 (클라우드 의존성 제거)

### 9.2 학술 영향력

**동시대 연구에 대한 영향**:
- 2024-2025년의 여러 분산 LLM 논문에서 참조됨
- 엣지 LLM 종합 설문 (Zheng et al., 2025)에 포함

**향후 연구 방향 제시**:
- 자기회귀 모델의 특성을 고려한 병렬화 필요성 강조
- 엣지 환경의 이질성 처리 과제 도출

### 9.3 실제 적용 가능성

**현실적 제약**:
- ✗ 몇 개의 저성능 기기의 네트워킹 복잡성 (설정/유지)
- ✗ 초기 파이프라인 충전 오버헤드 (소규모 작업에 부적합)
- ✗ Secondary 노드의 보안 취약점

**적합한 시나리오**:
- ✓ IoT 게이트웨이에서의 정기적 추론 (기상 예측, 센서 데이터 분석)
- ✓ 개인 기기의 협력 네트워크 (집 내 스마트 기기)
- ✓ 공장/산업 현장 분산 제어 시스템
- ✓ 프라이버시가 중요한 의료/금융 애플리케이션

### 9.4 최종 평가

**과학적 기여도**: ⭐⭐⭐⭐☆ (4/5)
- 자기회귀 특성 고려는 혁신적
- 실험 설정은 محدود (제한적)

**실무 적용 가능성**: ⭐⭐⭐☆☆ (3/5)
- 프로토타입 수준, 운영 시스템 아님
- 초기 배포 복잡성 높음

**장기적 영향력**: ⭐⭐⭐⭐☆ (4/5)
- 엣지 LLM 분산 추론의 새로운 패러다임
- 2025년 이후 관련 연구의 기초 마련

---

## 참고문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef0d119c-09ed-4c2f-b098-03adb11e5a9f/2505.18164v1.pdf)
[2](https://www.mdpi.com/2077-0472/15/21/2220)
[3](https://dl.acm.org/doi/10.1145/3711896.3737858)
[4](https://arxiv.org/abs/2502.07503)
[5](https://arxiv.org/abs/2510.16374)
[6](https://dl.acm.org/doi/10.1145/3711896.3737848)
[7](https://arxiv.org/abs/2503.22732)
[8](https://ieeexplore.ieee.org/document/11179972/)
[9](https://arxiv.org/abs/2509.12645)
[10](https://ieeexplore.ieee.org/document/11182097/)
[11](https://www.jmir.org/2025/1/e74177)
[12](http://arxiv.org/pdf/2406.15758.pdf)
[13](https://arxiv.org/pdf/2410.11845.pdf)
[14](https://arxiv.org/pdf/2503.09114.pdf)
[15](http://arxiv.org/pdf/2411.02829.pdf)
[16](http://arxiv.org/pdf/2410.00531.pdf)
[17](https://arxiv.org/pdf/2504.03360.pdf)
[18](http://arxiv.org/pdf/2405.14636.pdf)
[19](http://arxiv.org/pdf/2504.03668.pdf)
[20](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2025.1538277/full)
[21](https://date24.date-conference.com/proceedings-archive/2024/DATA/223_pdf_upload.pdf)
[22](https://par.nsf.gov/servlets/purl/10345258)
[23](https://ceur-ws.org/Vol-3943/paper28.pdf)
[24](https://iqua.ece.toronto.edu/papers/chenghao-icdcs24.pdf)
[25](https://davisjam.github.io/files/publications/GoelTungHuThiruvathukalDavisLu-ASPDAC2022.pdf)
[26](https://ina.kaist.ac.kr/assets/bibliography/SpecEdge.pdf)
[27](https://www.infracloud.io/blogs/inference-parallelism/)
[28](https://ieeexplore.ieee.org/document/9996638/)
[29](https://arxiv.org/html/2505.16508v1)
[30](https://ieeexplore.ieee.org/document/10546617/)
[31](https://arxiv.org/abs/2110.14895)
[32](https://dl.acm.org/doi/10.1145/3731806.3731859)
[33](https://huggingface.co/docs/transformers/perf_infer_gpu_multi)
[34](https://dl.acm.org/doi/10.1145/3676536.3676779)
[35](https://www.sciencedirect.com/science/article/abs/pii/S1574013725000310)
[36](https://arxiv.org/html/2403.03699v1)
[37](https://www.themoonlight.io/ko/review/asteroid-resource-efficient-hybrid-pipeline-parallelism-for-collaborative-dnn-training-on-heterogeneous-edge-devices)
[38](https://neurips.cc/virtual/2025/poster/115088)
[39](https://scholarx.skku.edu/handle/2021.sw.skku/111645)
[40](https://arxiv.org/html/2512.21835v1)
[41](https://arxiv.org/html/2507.12145v1)
[42](https://arxiv.org/pdf/2510.11331.pdf)
[43](https://ar5iv.labs.arxiv.org/html/2110.14895)
[44](https://arxiv.org/abs/2501.06589)
[45](https://arxiv.org/abs/2311.05827)
[46](https://arxiv.org/html/2506.09397v3)
[47](https://arxiv.org/abs/2507.14392)
[48](https://arxiv.org/html/2502.19864v1)
[49](https://arxiv.org/html/2410.11845v2)
[50](https://arxiv.org/html/2408.07802v1)
[51](https://arxiv.org/abs/2408.08015)
[52](https://arxiv.org/html/2510.18544v3)
[53](https://arxiv.org/abs/2411.01738)
[54](https://arxiv.org/abs/2109.13356)
[55](https://arxiv.org/html/2505.18164v1)
[56](https://arxiv.org/abs/2405.17245)

