# PETALS: Collaborative Inference and Fine-tuning of Large Models

### 1. 핵심 주장 및 기여도 요약

**PETALS**(Petrov Ensemble Learning with Transformer Adaptation via Large-Scale collaboration)는 지리적으로 분산된 여러 참여자의 GPU 자원을 활용하여 176B 매개변수 규모의 대규모 언어 모델(LLM)을 효율적으로 추론하고 미세조정할 수 있는 시스템이다.[1]

**핵심 주장**은 다음과 같다:

1. **접근성 제약 해결**: BLOOM-176B와 OPT-175B 같은 모델은 350GB 이상의 GPU 메모리를 필요로 하여 대부분의 연구자에게 접근 불가능했으나, PETALS는 여러 소비자급 GPU를 네트워크로 연결하여 이를 해결한다.

2. **기존 방법의 한계 극복**: RAM 오프로딩(5.5초 이상의 지연시간)과 API 기반 접근(내부 상태 접근 불가) 방식의 문제를 해결하면서도 각각의 이점을 유지한다.

3. **협업 인프라 구축**: 인터넷을 통한 협업 추론으로 초당 1 스텝의 추론 속도를 달성하며, 이는 많은 인터랙티브 LLM 애플리케이션에 충분하다.[1]

**주요 기여**:
- 파이프라인 병렬화를 통한 계층별 분산 배치 아키텍처
- 8비트 동적 양자화를 활용한 대역폭 50% 감소
- 적응형 로드 밸런싱 및 클라이언트 라우팅 알고리즘
- 파라미터 효율적 미세조정(Adapter, PromptTuning) 지원
- Hugging Face Hub를 통한 미세조정 모듈 공유 인프라

***

### 2. 문제 정의, 제안 방법론, 모델 구조

#### 2.1 문제 정의

PETALS가 해결하는 근본적인 문제는 **계산 비용과 메모리 제약**으로 요약된다.[1]

- **메모리 병목**: BLOOM-176B의 16비트 표현은 352GB 필요 → 44개의 8GB GPU 필요
- **지연시간 문제**: 토큰당 네트워크 전달 시간이 병렬 처리의 이점을 상쇄
- **신뢰성 도전**: 자원봉사자 네트워크에서 임의의 노드가 언제든 이탈 가능
- **유연성 부족**: API는 내부 상태 접근 불가, 오프로딩은 너무 느림

#### 2.2 제안 방법론

PETALS는 세 가지 주요 설계 원칙을 기반으로 한다:

##### (1) 추론 구조

클라이언트-서버 모델로 작동한다:[1]

$$\text{클라이언트} \xrightarrow{\text{토큰 벡터}} \text{서버}_{1} \xrightarrow{\text{활성화}} \text{서버}_{2} \rightarrow \cdots \xrightarrow{\text{출력}} \text{클라이언트}$$

각 참여자는 일부 Transformer 블록을 호스트하는 서버, 또는 모델을 사용하는 클라이언트, 또는 둘 다의 역할을 수행한다.

**추론 과정**:
1. 클라이언트가 토큰 임베딩을 로컬에서 계산: $h_0 = E(x_t)$ (E: 임베딩 함수)
2. $h_0$를 서버 체인에 전송
3. 각 서버가 자신의 Transformer 블록 계산: $h_i = B_i(h_{i-1})$
4. 마지막 서버에서 로그잇 반환: $\hat{y} = \text{softmax}(W_L h_L)$

##### (2) 분산 미세조정

핵심 원리: **클라이언트가 학습 가능한 파라미터를 소유하고, 서버는 원래 사전학습된 계층만 유지한다.**[1]

소프트 프롬프트 튜닝 예시로 설명하면:

$$\mathcal{L} = \text{CrossEntropy}(\text{head}(h_L), y)$$

여기서:
- $h_L$: 서버 체인의 최종 출력
- $\text{head}(\cdot)$: 클라이언트의 분류 헤드 (학습 가능)

역전파 계산:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{p}} = \frac{\partial \mathcal{L}}{\partial h_L} \cdot \frac{\partial h_L}{\partial \mathbf{p}}$$

서버는 $\frac{\partial h_L}{\partial h_{L-1}}$만 계산하여 클라이언트에 반환하고, 자신의 파라미터는 업데이트하지 않는다.

#### 2.3 모델 구조

##### (1) 메모리 최적화 기법

**8비트 양자화 (LLM.int8())**:

기본 문제: 176B 파라미터 모델이 44개 노드 필요 → 높은 레이턴시

해결책: 8비트 혼합 정밀도 분해[2]

$$\text{가중치} = \underbrace{0.1\%}_{\text{16비트 이상치}} + \underbrace{99.9\%}_{\text{8비트 값}}$$

이를 통해 노드 수를 22개로 감소:

| 정밀도 | 메모리 요구 (GB) | 노드 수 | 모델 성능 |
|--------|---|---|---|
| 16비트 | 352 | 44 | 100% |
| 8비트 | 176 | 22 | 100% (BLOOM) |

표 1의 실험 결과:[1]

| 모델 | 비트 | HellaSwag | LAMBADA | WinoGrande | 평균 |
|------|-----|----------|---------|-----------|------|
| OPT-175B | 16 | 78.5 | 74.7 | 72.6 | 75.3 |
| OPT-175B | 8 | 78.5 | 74.6 | 71.7 | 74.9 |
| BLOOM | 16 | 73.0 | 67.2 | 70.1 | 70.1 |
| BLOOM | 8 | 72.8 | 68.1 | 70.1 | 70.3 |

**통신 최적화**:

동적 블록별 양자화를 사용하여 숨겨진 상태 교환 시 대역폭 50% 감소:[1]

$$\text{대역폭}_{압축} = 0.5 \times \text{대역폭}_{원본}$$

##### (2) 파이프라인 병렬화 구조

모델은 연속적인 Transformer 블록 그룹으로 분할:

$$\text{모델} = [B_1, B_2, \ldots, B_{96}] \text{ (BLOOM의 경우)}$$

각 서버는 contiguous 블록 집합을 호스트. 이 설계는 sequential dependencies를 고려하되, 동시에 여러 클라이언트가 독립적으로 처리할 수 있게 한다.

##### (3) 로드 밸런싱 알고리즘

서버들은 분산 해시 테이블에 자신의 처리량을 공시:[1]

$$\text{새로운 서버의 블록 선택} = \arg\max_{I} \sum_{i \in I} \text{처리량}_{i}$$

이를 통해 가장 느린 블록들이 개선된다.

***

### 3. 성능 향상 및 한계

#### 3.1 성능 벤치마크

표 3에 따른 추론 성능:[1]

| 환경 | 단일 배치 (steps/s) | 병렬 포워드 (tokens/s) |
|------|----------|----------|
| PETALS 3서버, 1Gbps, RTT 5ms | 1.71 | 253.6 |
| PETALS 12가상서버, 100Mbps, RTT 5ms | 1.24 | 66.6 |
| PETALS 14실제서버(글로벌) | 0.83 | 179.4 |
| RAM 오프로딩 (1×A100, 최대 256Gbps) | 0.18 | 170.3 |

**핵심 발견**:
- 단일 배치 추론에서 PETALS는 RAM 오프로딩 대비 4-9배 빠름
- 병렬 포워드 패스(배치 학습)에서는 조건에 따라 경쟁력 있는 성능

**레이턴시 민감도**:[1]

인터넷 환경에서 네트워크 레이턴시가 가장 중요한 요소. 동시 클라이언트 8개일 때 각각 20배 느려짐 (12서버, 100Mbps, 100ms RTT).

#### 3.2 양자화 오버헤드

표 2에 따른 생성 처리량:[1]

| 정밀도 | 배치 1 | 배치 8 | 배치 32 |
|--------|--------|--------|---------|
| 16비트 | 4.18 | 31.3 | 100.6 |
| 8비트 | 3.95 | 29.4 | 95.8 |
| 오버헤드 | 5% | 6% | 5% |

배치 크기가 증가할수록 양자화 오버헤드는 무시할 수 있는 수준.

#### 3.3 한계 및 미해결 문제

**프라이버시 취약점**:[1]

첫 번째 계층을 호스팅하는 서버는 입력 토큰을 복원 가능:

$$\text{토큰} = D(\text{임베딩 벡터})$$ (D: 임베딩 역함수)

**해결책**: 신뢰할 수 있는 서버만 사용하거나, 향후 안전한 다자간 계산 활용

**보안 도전**:[1]

악의적인 서버가 잘못된 계산 결과 반환 가능. 논문은 경제적 인센티브 기반 검증 메커니즘 제안:

$$\text{보증금} \rightarrow \text{입출력 암호 해시} \rightarrow \text{검증 실패 시 몰수}$$

**공급-수요 불균형**:[1]

서버 역할의 참여자가 부족할 경우, 지점 기반 인센티브 시스템 필요.

***

### 4. 모델의 일반화 성능 향상 가능성

#### 4.1 파라미터 효율적 미세조정의 일반화

PETALS가 지원하는 Adapter와 LoRA 기반 미세조정의 일반화 특성:

**LoRA의 일반화**:[3]

저랭크 분해로 인한 정규화 효과:

$$\Delta W = BA, \quad A \in \mathbb{R}^{d \times r}, B \in \mathbb{R}^{r \times k}, r \ll \min(d,k)$$

매개변수 수: $r(d+k)$ vs 전체 $d \times k$ (보통 100-1000배 감소)

이러한 파라미터 제약이 자동적으로 정규화로 작용하여:
- 저데이터 레짐에서 전체 미세조정 능가[3]
- 도메인 외 강건성 향상

**Adapter의 전이 학습**:[4]

Adapter 모듈을 특정 계층에 삽입하면, 기본 모델의 표현은 유지되고 작업별 특화만 이루어짐:

$$h_i' = h_i + \text{Adapter}(h_i)$$

이를 통해 여러 작업에 빠르게 적응 가능 (학습 매개변수 0.02% 이하에서 98% 성능 달성)[4]

#### 4.2 협업 개선의 일반화 효과

PETALS의 핵심 이점: **더 많은 사용자가 기여할수록 모델이 개선됨**[1]

$$\text{모델 성능} = f(\text{기본 모델}, \text{누적 적응 모듈})$$

다양한 도메인과 작업의 적응 모듈이 축적되면:
- 기본 모델이 암묵적으로 여러 작업의 공통 표현 학습
- 새로운 작업에 대한 전이 학습 성능 향상

#### 4.3 양자화의 일반화 영향

**8비트 양자화의 안정성**:[2]

LLM.int8() 방식은 이상치(outliers)를 16비트로 유지하므로:
- 중요한 특성 손실 최소화
- 대규모 모델일수록 양자화 영향 감소

**4비트의 한계와 가능성**:[5]

최근 연구에 따르면 4비트 정밀도도 가능:
- Well-tuned INT8: 1-3% 정확도 손실
- FP8: 거의 손실 없음 (최신 하드웨어 지원)[6]

***

### 5. 2020년 이후 관련 최신 연구 비교 분석

#### 5.1 분산 추론 기술의 진화

**Phase 1: 오프로딩 기반 (2020-2021)**

- **문제**: 단일 기기의 메모리 제약
- **해결책**: RAM/SSD 오프로딩
- **한계**: 매우 높은 레이턴시 (5.5초 이상)

**Phase 2: 모델 병렬화 표준화 (2021-2022)**

1. **Tensor Parallelism**:[7]
   - 각 계층을 수평으로 분할
   - 어텐션 헤드나 MLP를 여러 장치에 분산
   - 장점: 메모리 효율적
   - 단점: 높은 통신 오버헤드

2. **Pipeline Parallelism**,:[8][7]
   - 모델 계층을 수직으로 분할
   - 각 장치가 순차 처리
   - **파이프라인 거품 문제**: 일부 장치가 유휴 상태

3. **Sequence Parallelism**:[8]
   - LayerNorm, Dropout 등 sequence-independent 연산을 분할
   - Tensor parallelism의 한계 보완

**Phase 3: 양자화 기술 발전 (2022-2023)**

| 기법 | 발표 | 특징 | 성능 |
|------|------|------|------|
| LLM.int8() | 2022 [2] | 혼합 정밀도 | 175B 무손실 |
| 8-bit Optimizer | 2022 [9] | 블록별 양자화 | 32비트 유지 |
| QLoRA | 2023 [10] | 4비트 + LoRA | 65B/48GB GPU |
| SmoothQuant | 2023 [11] | 채널별 스케일링 | W8A8 무손실 |

**Phase 4: 파라미터 효율적 미세조정 (2021-2023)**

- **LoRA** [Hu et al., 2021]: 저랭크 적응, 0.01-3% 매개변수
- **Adapter** [Houlsby et al., 2019]: 작은 모듈 삽입, 0.02% 매개변수로 98% 성능
- **Prefix Tuning** [Li & Liang, 2021]: 학습 가능한 프리픽스
- **Soft Prompt Tuning** [Lester et al., 2021]: 학습 가능한 토큰

**Phase 5: 분산 학습 패러다임 (2020-2023)**

1. **Federated Learning** [McMahan et al., 2017]:
   - 중앙 서버와 클라이언트 간 협업
   - 데이터 프라이버시 유지

2. **Split Learning**:[12]
   - 모델을 클라이언트-서버 간 분할
   - 의료 데이터 같은 민감한 데이터 처리에 적합
   - 성능: 연합 학습과 유사하나 메모리 효율성 우수 (77% 감소)

3. **Hivemind** [Learning@Home, 2020]:
   - 자원봉사자 네트워크에서 분산 학습
   - PETALS의 이론적 기초

#### 5.2 최신 엣지 컴퓨팅 최적화 (2024-2025)

**MDI-LLM** (Model-Distributed Inference):[13]
- 타겟: 엣지 장치
- 기법: 반복 파이프라인 병렬화
- 성능: 74% 처리량 향상 (4노드 기준)

**Distributed Inference Optimization (DIO-LLMs)**:[14]
- 엣지-클라우드 협업
- 2단계 모델 분할 (인터레이어 + 인트라레이어)
- Greedy PPO 기반 최적 오프로드 전략

**TD-Pipe** (Temporally-Disaggregated Pipeline):[15]
- 프리필-디코드 페이즈 분리로 파이프라인 거품 제거
- 처리량: 기존 파이프라인 대비 2.73배 향상
- 통신 제약 환경 (PCIe) 최적화

**ServerlessPD**:[16]
- 서버리스 환경 최적화
- RDMA 기반 원격 포크로 콜드스타트 제거
- 동적 프리필-디코드 스케줄링

#### 5.3 PETALS와 최신 연구의 위상

| 차원 | PETALS (2023) | 최신 연구 (2024-2025) |
|------|---|---|
| **배포 환경** | 인터넷 규모 분산 | 엣지, 클라우드, 하이브리드 |
| **주요 최적화** | 양자화 + 파이프라인 | 시간 분산, 동적 스케줄링 |
| **레이턴시 특성** | 높은 레이턴시 환경 | 저 레이턴시 환경 |
| **동적 참여** | **지원** | 제한적 또는 미지원 |
| **처리량** | 중간-높음 | 매우 높음 |
| **학습 가능성** | **지원** | 대부분 추론만 |
| **모듈 공유** | **Hugging Face Hub** | 미지원 |

**핵심 차별점**:
- PETALS는 여전히 유일하게 **인터넷 규모의 협업 추론 + 미세조정**을 동시에 지원
- 최신 연구들은 **처리량과 레이턴시 최적화**에 초점
- PETALS의 **모듈 공유 및 협업 개선** 개념이 향후 방향 제시

***

### 6. 앞으로의 연구에 미치는 영향과 고려사항

#### 6.1 기술적 영향

**1. 민주화 효과**

PETALS는 AI 연구의 접근성을 근본적으로 변화시킴:
- 개별 연구자: GPU 24GB 이내로 176B 모델 활용 가능
- 소규모 기관: 고가의 데이터센터 투자 없이 대규모 모델 연구 가능
- 개발도상국: 지역 컴퓨팅 자원으로 최신 모델 활용 가능

**2. 시스템 설계 패러다임의 변화**

기존: 중앙집중식 → 미래: 분산형 협업 모델
- 자원봉사 인프라 구축 가능성
- 오픈소스 커뮤니티의 주도적 역할 강화

**3. 하드웨어 효율성 표준 제시**

8비트 양자화 + 파이프라인 병렬화 조합의 효율성:
- 메모리 50% 감소, 대역폭 50% 감소
- 성능 손실 거의 없음
- 이후 연구들의 기본 패러다임으로 채택

#### 6.2 앞으로의 연구 과제

**A. 보안 및 신뢰성**

현재 한계: 첫 계층의 입력 공개, 악의적 서버 가능

필요 연구:
1. **동형 암호 (Homomorphic Encryption)**: 암호화 상태에서 계산 수행
2. **영지식 증명 (Zero-Knowledge Proof)**: 계산 정확성 증명
3. **신뢰도 추적**: 블록체인 기반 서버 평판 시스템

**B. 프라이버시 강화**

제안 방법:
1. **차등 프라이버시 (Differential Privacy)**: 그래디언트에 노이즈 추가
2. **안전한 다자간 계산 (Secure MPC)**: 입력 데이터 보호
3. **프라이버시 보존 하드웨어**: NVIDIA Confidential Computing

**C. 모델 버전 관리**

현재 미지원: 기본 모델 업데이트 시 적응 모듈의 호환성 문제

제안:
1. **의미론적 버전 관리**: 변경 사항의 영향도 추적
2. **적응형 벤치마크**: 새 버전의 성능 자동 평가
3. **호환성 레이어**: 이전 버전 적응 모듈 지원

**D. 인센티브 메커니즘**

현재: 제안만 있음, 구현 필요

설계 원칙:
$$\text{인센티브} = f(\text{서버 제공량}, \text{클라이언트 사용량}, \text{신뢰도})$$

구체적 메커니즘:
1. **포인트 시스템**: 기여도에 따른 우선순위 큐
2. **평판 시스템**: 작업 성공률에 따른 신뢰도 증가
3. **암호자산**: 블록체인 기반 토큰화

**E. 하드웨어 이질성 최적화**

현재: 동일 성능 가정

향후:
1. **동적 계층 할당**: 장치 성능에 따른 최적 계층 수
2. **연결 인식 라우팅**: 네트워크 토폴로지 학습
3. **적응형 양자화**: 장치 성능에 따른 정밀도 조정

**F. 멀티모달 확장**

현재: 텍스트 LLM만 지원

필요 연구:
1. **Vision Transformer 지원**: 이미지 입력 처리
2. **음성 처리 통합**: 음성 토큰 추론
3. **크로스모달 적응**: 멀티모달 미세조정

#### 6.3 실무적 고려사항

**1. 규제 준수**

| 지역 | 규제 | 영향 | 해결책 |
|------|------|------|------|
| EU | GDPR | 데이터 처리 위치 제한 | 지역 내 서버 우선 |
| US | CCPA | 개인정보 수집 동의 | 명확한 프라이버시 정책 |
| China | 데이터 주권 | 국경 외 데이터 이동 금지 | 로컬 배포 필수 |

**2. 경제성 분석**

API 가격 (OpenAI): $0.0005-0.002 per 1K token
PETALS 경우: 
- 서버 호스팅 비용 (GPU) vs. API 호출 비용
- 초대규모 배포에서 경제성 우수

**3. 확장성 한계**

현재 PETALS 평가:
- 최대 14개 실제 서버 (2023년 논문 발표 시점)
- 동시 클라이언트 8개에서 20배 성능 저하

향후 도전:
- 1,000+ 노드 환경에서의 안정성
- 하이브리드 클라우드-엣지 통합

***

### 7. 결론: PETALS의 위상과 미래

**현재 (2023-2024)**:
- 분산 AI 시스템 연구의 선도적 사례
- 실제 작동하는 프로토타입과 공개 서비스 제공 (chat.petals.ml)
- 학술 커뮤니티의 표준 참조점

**근기 (2024-2025)**:
- 엣지 컴퓨팅 통합으로 레이턴시 개선
- 양자화 및 파이프라인 기술의 표준화 가속
- 보안 메커니즘 강화 필수

**장기 (2026+)**:
- 자동화된 모델 버전 관리 시스템 확립
- 협업 모델 개선이 기본 모델 개선으로 직결되는 선순환 구조 구축
- AI 인프라의 진정한 민주화 실현

**최종 평가**:

PETALS는 단순히 기술적 혁신을 넘어 **AI 접근성의 패러다임 전환**을 제시한다. 대규모 모델이 더 이상 대기업의 독점이 아닌 커뮤니티 자산이 될 수 있음을 보여주었으며, 이는 AI 연구와 개발의 생태계를 근본적으로 변화시킬 잠재력을 지닌다.

***

**참고문헌**

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e8b7b5b0-6a05-401f-952f-35c1a3e074eb/2209.01188v2.pdf)
[2](https://arxiv.org/abs/2208.07339)
[3](https://www.emergentmind.com/topics/low-rank-adaptation-lora-adapters)
[4](https://aclanthology.org/2023.sustainlp-1.8.pdf)
[5](https://arxiv.org/abs/2212.09720)
[6](https://arxiv.org/pdf/2411.02355.pdf)
[7](https://nlpcloud.com/llm-inference-optimization-techniques.html)
[8](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)
[9](https://www.semanticscholar.org/paper/11fe37ab6faf6bf85ad2f5746c154dec5412bd04)
[10](https://arxiv.org/abs/2305.14314)
[11](https://arxiv.org/pdf/2211.10438.pdf)
[12](https://pmc.ncbi.nlm.nih.gov/articles/PMC10785879/)
[13](https://ieeexplore.ieee.org/document/11154542/)
[14](https://ieeexplore.ieee.org/document/11160773/)
[15](https://arxiv.org/html/2506.10470v1)
[16](https://ieeexplore.ieee.org/document/11169657/)
[17](https://dl.acm.org/doi/10.1145/3731806.3731859)
[18](https://arxiv.org/abs/2506.10426)
[19](https://arxiv.org/abs/2510.11211)
[20](https://ieeexplore.ieee.org/document/11124961/)
[21](https://linkinghub.elsevier.com/retrieve/pii/S0166531625000616)
[22](https://ieeexplore.ieee.org/document/10759588/)
[23](https://www.semanticscholar.org/paper/62811cc1ca22bc9ec66f053d8615f79c08a18380)
[24](https://arxiv.org/html/2503.16585v1)
[25](https://arxiv.org/pdf/2407.14645.pdf)
[26](http://arxiv.org/pdf/2405.14105.pdf)
[27](https://arxiv.org/pdf/2407.12391.pdf)
[28](https://arxiv.org/pdf/2311.11514.pdf)
[29](https://arxiv.org/pdf/2312.03140.pdf)
[30](http://arxiv.org/pdf/2402.15758.pdf)
[31](http://arxiv.org/pdf/2503.19050.pdf)
[32](https://www.youtube.com/watch?v=_xAXb70d4-0)
[33](https://www.linkedin.com/pulse/distributed-large-language-model-inference-ml-engineers-jawad-md-shskc)
[34](https://arxiv.org/abs/2510.00206)
[35](https://openreview.net/pdf?id=FYHktcK-7v)
[36](https://developers.redhat.com/articles/2025/11/21/introduction-distributed-inference-llm-d)
[37](https://aclanthology.org/2024.findings-acl.933.pdf)
[38](https://www.vldb.org/pvldb/vol15/p1581-wolfe.pdf)
[39](https://arxiv.org/html/2505.18164v1)
[40](https://arxiv.org/html/2505.18906v1)
[41](https://arxiv.org/pdf/2508.08382.pdf)
[42](https://arxiv.org/abs/2507.20424)
[43](https://www.arxiv.org/pdf/2509.10843.pdf)
[44](https://arxiv.org/html/2501.10326v2)
[45](https://arxiv.org/abs/2403.07585)
[46](https://arxiv.org/html/2509.24877v1)
[47](https://arxiv.org/pdf/2509.19628.pdf)
[48](https://arxiv.org/html/2309.16584v3)
[49](https://arxiv.org/html/2503.06072v3)
[50](https://ieeexplore.ieee.org/document/10247258/)
[51](https://openreview.net/forum?id=XmN7ZNbUAe)
[52](https://www.emergentmind.com/topics/lora-based-parameter-efficient-fine-tuning)
[53](https://arxiv.org/abs/2007.03970)
[54](http://papers.neurips.cc/paper/7454-collaborative-learning-for-deep-neural-networks.pdf)
[55](https://github.com/tao-shen/Distributed-LLM-Edges)
[56](https://www.semanticscholar.org/paper/d4bfb688a5a4644a1d187f345edaa27cc5710f6c)
[57](https://www.semanticscholar.org/paper/b3fc0ffc6d784973f2d5b34b06de323270392980)
[58](https://ieeexplore.ieee.org/document/9980120/)
[59](https://arxiv.org/abs/2312.05725)
[60](https://aclanthology.org/2023.emnlp-main.102)
[61](https://www.semanticscholar.org/paper/eca51d9b64855ee9d814be2eb0cafd03ebc78ef3)
[62](https://arxiv.org/pdf/2208.07339.pdf)
[63](http://arxiv.org/pdf/2405.14597.pdf)
[64](https://aclanthology.org/2023.emnlp-main.910.pdf)
[65](https://arxiv.org/pdf/2206.01861.pdf)
[66](https://aclanthology.org/2023.emnlp-main.617.pdf)
[67](https://arxiv.org/html/2410.07505)
[68](https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/)
[69](https://smashinggradient.com/2023/04/11/summary-of-adapter-based-performance-efficient-fine-tuning-peft-techniques-for-large-language-models/)
[70](https://www.reddit.com/r/MachineLearning/comments/wrpg59/r_llmint8_8bit_matrix_multiplication_for/)
[71](https://developer.nvidia.com/ko-kr/blog/mastering-llm-techniques-inference-optimization/)
[72](https://arxiv.org/html/2405.05493v1)
[73](https://arxiv.org/html/2402.16775v1)
[74](https://arxiv.org/html/2502.12478v1)
[75](https://www.semanticscholar.org/paper/LLM.int8():-8-bit-Matrix-Multiplication-for-at-Dettmers-Lewis/4be7d1524edb0137599a5cc95f72844b85a52fe1)
[76](https://arxiv.org/pdf/2207.00032.pdf)
[77](https://arxiv.org/html/2503.05683v1)
[78](https://www.arxiv.org/pdf/2509.19368.pdf)
[79](https://arxiv.org/pdf/2401.04679.pdf)
[80](https://www.infracloud.io/blogs/inference-parallelism/)
[81](https://www.emergentmind.com/topics/adapter-based-fine-tuning)
