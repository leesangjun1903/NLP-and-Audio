# End-to-End Anti-Spoofing with RawNet2

## 1. 핵심 주장과 주요 기여

이 연구는 **RawNet2를 최초로 음성 스푸핑 탐지에 적용한 연구**로, 다음과 같은 핵심 주장을 제시합니다:[1]

**핵심 주장:**
- 기존 hand-crafted 특징 기반 접근법은 알려진 공격에 특화된 아티팩트에 의존하므로, 미지의 공격 탐지에 한계가 있음
- Raw waveform에서 직접 동작하는 end-to-end 학습 접근법이 보다 일반화 가능한 표현을 학습할 수 있음
- 특히 A17 공격(neural network voice conversion)과 같은 최악의 시나리오에서 성능 향상 가능

**주요 기여:**
1. **RawNet2의 최초 안티스푸핑 적용**: 원래 화자 인식용으로 설계된 RawNet2를 스푸핑 탐지에 성공적으로 적용[1]
2. **A17 공격에 대한 우수한 성능**: A17 공격에 대해 두 번째로 우수한 성능 달성 (min t-DCF: 0.181 vs 기존 0.3524)[1]
3. **상호 보완적 융합 시스템**: LFCC 기반 시스템과 융합하여 전체 ASVspoof 2019 LA 조건에서 두 번째 우수 성능 달성[1]

## 2. 문제 정의 및 제안 방법

### 해결하고자 하는 문제

**핵심 문제:**
ASVspoof 2019 평가에서 대부분의 공격은 탐지 가능했으나, A17 공격(neural network voice conversion with generalised direct waveform modification)은 여전히 탐지를 회피하는 문제[1]

**구체적 한계:**
- High-spectral-resolution front-end도 A17 아티팩트를 포착하지만 탐지하지 못함
- 대표적 학습 데이터가 있을 때만 탐지 가능하고, 표준 학습 데이터만으로는 성능 저하
- 미지의 공격에 대한 사전 예방적 보호 전략 필요[1]

### 제안 방법

**RawNet2 아키텍처 수정사항:**

1. **입력 처리**:
   - Raw waveform 입력: 64,000 샘플 (≈4초)
   - Layer normalization 미적용 (성능 저하로 인해)
   - Fixed duration으로 정규화 (긴 발화는 자르고, 짧은 발화는 연결)[1]

2. **첫 번째 레이어 (Sinc 필터)**:
   - 필터 길이: 251 → 129 샘플로 단축
   - 고정 필터 사용 (학습하지 않음) - 과적합 방지
   - 3가지 초기화 방식 실험: Mel-distributed, linearly-distributed, inverse-Mel scaled[1]

3. **잔차 블록 구성**:
   - 첫 번째 잔차 블록: 128 필터 (변경 없음)
   - 두 번째 잔차 블록: 512 필터 (증가)
   - Filter-wise Feature Map Scaling (FMS) 적용[1]

4. **후단 처리**:
   - GRU layer: 1024 hidden nodes
   - 추가 fully connected layer
   - Softmax 활성화 함수 (2-class 분류: bona fide vs spoof)[1]

**학습 설정:**
- 최적화: ADAM optimizer
- 학습률: 0.0001
- 에포크: 100
- 배치 크기: 32[1]

## 3. 모델 구조

**전체 아키텍처 (Table 1 기준):**

```
Input: 64,000 samples
↓
Conv(129,1,128) - Fixed Sinc filters
↓
Maxpooling(3) → (21,290, 128)
↓
Residual Block × 2 (128 filters)
- BN & LeakyReLU
- Conv(3,1,128)
- FMS
↓
Maxpooling(3) → (2,365, 128)
↓
Residual Block × 4 (512 filters)  
- BN & LeakyReLU
- Conv(3,1,512)
- FMS
↓
Maxpooling(3) → (29, 512)
↓
GRU(1024) → (1024)
↓
FC(1024) → (1024)
↓
Output(2) - Softmax
```

**핵심 구조적 특징:**
- **Sinc 필터**: 제약된 첫 번째 레이어로 의미있는 필터뱅크 구조 학습
- **잔차 연결**: 더 깊은 네트워크 학습 가능
- **FMS (Filter-wise Feature Map Scaling)**: 주의 메커니즘으로 판별적 표현 강화
- **GRU**: 프레임 레벨 표현을 발화 레벨 표현으로 집약[1]

## 4. 성능 향상 및 한계

### 성능 향상

**전체 성능:**
- 기존 LFCC 기준선: pooled min t-DCF 0.09
- 최고 RawNet2 (S2): pooled min t-DCF 0.1175
- 전체적으로는 기준선보다 성능 저하[1]

**A17 공격 특화 성능:**
- 기존 LFCC: min t-DCF 0.3524  
- RawNet2 (S3): min t-DCF 0.181 (**48% 향상**)
- 현재까지 발표된 결과 중 두 번째 우수 성능[1]

**융합 시스템 성능:**
- LFCC + RawNet2 융합: pooled min t-DCF 0.0330
- ASVspoof 2019 LA 전체 조건에서 두 번째 우수 성능
- A17 공격에 대해서는 min t-DCF 0.1161로 크게 개선[1]

### 한계점

1. **전체 성능 저하**: 개별 RawNet2 시스템은 모든 공격에 대한 pooled 성능이 기준선보다 낮음
2. **여전히 높은 A17 오류율**: A17에 대한 min t-DCF가 pooled 결과보다 여전히 250% 이상 높음
3. **학습 데이터 제약**: ASVspoof 2019 LA 데이터셋의 제한된 공격 유형(6개)으로 인한 과적합 위험[1]

## 5. 일반화 성능 향상 가능성

### 핵심 메커니즘

**1. Raw Waveform 기반 학습:**
- Hand-crafted 특징에 의존하지 않아 알려진 공격의 특정 아티팩트에 제약되지 않음
- 자동적으로 최적화된 표현 학습으로 미지 공격에 대한 적응 가능성 증대[1]

**2. 시간적 주의(Temporal Attention) 메커니즘:**
- GRU와 FMS를 통한 시간적 패턴 학습
- A17 공격의 특징인 간헐적 클릭 노이즈와 같은 시간적 아티팩트 탐지 가능[1]

**3. 위상 정보 보존:**
- Linear-phase 필터를 통한 위상 정렬된 waveform 생성
- A17 공격의 위상 관련 아티팩트 포착 가능성 (가설)[1]

### 상호 보완성

**융합 시스템의 효과:**
- LFCC 시스템과 서로 다른 단서(cue) 학습
- 융합 결과가 개별 시스템보다 우수한 성능 → 상호 보완적 특성 입증[1]

## 6. 연구 영향과 향후 고려사항

### 연구에 미치는 영향

**1. 패러다임 전환:**
- Hand-crafted 특징에서 end-to-end 학습으로의 전환점
- Raw audio 기반 anti-spoofing 연구 활성화

**2. 최악 시나리오 대응:**
- A17과 같은 어려운 공격에 대한 새로운 접근법 제시
- 미지 공격에 대한 사전 예방적 전략 가능성 제시[1]

**3. 융합 시스템 효과성:**
- 서로 다른 접근법의 융합을 통한 성능 향상 전략 검증

### 향후 연구 고려사항

**1. 기술적 개선 방향:**
- 임베딩 기반 접근법 탐구
- 대안적 back-end 분류 방법 연구
- 위상 관련 아티팩트 탐지 메커니즘 검증 필요[1]

**2. 데이터 및 평가:**
- 더 다양한 공격 유형을 포함한 대규모 데이터셋 필요
- Cross-dataset 일반화 성능 평가
- Real-world 배포 환경에서의 성능 검증

**3. 해석 가능성:**
- 학습된 표현의 해석 가능성 연구
- 어떤 음향적 단서가 탐지에 기여하는지 분석
- 공격별 특화된 아티팩트 분석[1]

**4. 계산 효율성:**
- Real-time 처리를 위한 모델 경량화
- 에지 디바이스 배포를 위한 최적화

이 연구는 anti-spoofing 분야에서 end-to-end 학습의 가능성을 처음으로 입증했으며, 특히 어려운 공격에 대한 새로운 해결 방향을 제시했다는 점에서 중요한 기여를 했습니다.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/6e7693b3-0ca4-4171-ba45-98aee62dbafe/2011.01108v3.pdf)
