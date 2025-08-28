# UL2: Unifying Language Learning Paradigms

## 1. 핵심 주장 및 주요 기여  
**UL2**(Unifying Language Learning Paradigms)는 *다양한 NLP 과제*에서 **단일 사전학습 모델**이 우수한 성능을 내도록 하는 범용적 학습 프레임워크를 제안한다.[1]
- **Mixture-of-Denoisers(MoD)**: 서로 다른 스팬(sparse span corruption), 순차(prefix LM), 극단(extreme span corruption) 복원 과제를 혼합해 사전학습함으로써 모델이 폭넓은 문맥 조건화 능력을 획득  
- **Mode Switching**: 훈련·미세조정 시 `<R>`, `<S>`, `<X>` 토큰으로 사전학습 모드를 지정해 과제 유형에 최적화된 동작 전환  

이로써 UL2는 T5·GPT 계열 모델 대비 대부분의 텍스트 분류·생성·추론 과제에서 일관되게 성능 향상을 달성한다.[1]

## 2. 문제 설정  
기존 사전학습 모델은  
- *Encoder-Decoder(T5)*: 분류·생성 과제에 강하나, in-context 학습 성능이 제한적  
- *Decoder-only(GPT)*: 생성·추론에 강하나, 판별 과제 성능이 낮음  

과제별 최적 아키텍처·목적 함수를 선택해야 하는 번거로움과 자원 분산 문제를 해결하고자, **단일 모델**이 다양한 과제 패러다임을 동시에 수용하도록 **통합적 사전학습** 기법을 제안한다.[1]

## 3. 제안 방법  
### 3.1 Mixture-of-Denoisers (MoD)  
사전학습 시 서로 다른 denoising objective를 혼합  
- **R-Denoiser** (Regular): 짧은 span(μ=3‒8), 마스킹율 r=15%  
- **S-Denoiser** (Sequential): prefix LM (입력 시퀀스 앞부분만 컨텍스트로 사용)  
- **X-Denoiser** (Extreme): 긴 span(μ=64) 또는 높은 마스킹율(r=30–50%)  

MoD는 총 7개 조합으로 구성되며, 각 과제를 균등 비율로 혼합해 학습.[1]

### 3.2 Mode Switching  
사전학습·미세조정 입력 앞에 **모드 토큰** `<R>`, `<S>`, `<X>`를 붙여 해당 denoiser 모드를 활성화.  
모드 전환을 통해 downstream 과제에 적합한 학습 경로를 선택하게 함.[1]

### 3.3 수식적 통합 관점  
모든 denoising 과제를 **Input–Target** 형식으로 통일.  
- SpanCorrupt(μ, r, n) 함수로 corrupted span 생성  
- Causal LM, Prefix LM, Span Corruption을 span 길이 μ와 마스킹율 r 값을 조정해 동일한 포맷으로 표현 가능.[1]

## 4. 모델 구조  
- **아키텍처 무관**: Encoder-Decoder(T5)·Decoder-only(GPT-like) 모두 지원  
- Transformer 기본 구조에 SwiGLU, T5식 상대적 위치 인코딩 적용  
- 사전학습 대상은 C4 코퍼스(32B 토큰), 시퀀스 길이 512/512, Adafactor 옵티마이저 사용.[1]

## 5. 성능 향상 및 한계  
### 5.1 성능 향상  
- **슈퍼글루·생성 과제** 9종 모두에서 T5 대비 상대적 평균 **+43.6%** 향상, GPT 대비 **+76.1%** 향상.[1]
- **20B 규모**로 확장 시, 50여 개 벤치마크 과제에서 SOTA 달성 또는 근접(예: SuperGLUE 90.7점, XSum 8.6 rouge-2).[1]
- *Chain-of-Thought* prompting, self-consistency 등 추론 과제에서도 20B 모델이 emergent reasoning 능력 시연.[1]

### 5.2 한계 및 고려사항  
- **학습 복잡도**: MoD 혼합 과제 수 증가로 데이터 준비·하이퍼파라미터 튜닝 비용 상승  
- **스케일 안정성**: 20B 학습 중 간헐적 손실 스파이크 관찰, 장기간 모니터링·재시작 전략 필요  
- **모드 토큰 최적화**: downstream 과제별 최적 모드 토큰 선택이 성능에 민감(예: XSum의 경우)[1]

## 6. 일반화 성능 향상 관점  
MoD의 다양한 denoiser 노출은 모델이  
1. **짧은 범위 완성**(fact completion)  
2. **장문 생성**(long-span extrapolation)  
3. **순차 예측**(prefix LM)  

을 모두 학습하게 해 **범용성**을 확보한다. Mode Switching으로 downstream 과제에 맞춘 미세조정 없이도 적절한 전이 학습이 가능해 **제로/몇 샷 일반화** 성능이 크게 향상된다.[1]

## 7. 영향 및 향후 연구 고려사항  
- **범용 사전학습**: 단일 모델로 다양한 NLP 과제 대응하는 추세 가속  
- **하이브리드 목적 함수 설계**: MoD 개념을 이미지, 음성 등 멀티모달 학습에도 적용 가능  
- **스케일링 법칙**: 사전학습 데이터·모델 크기 확장에 따른 MoD 구성 최적화 연구 필요  
- **모드 토큰 자동화**: downstream 과제별 최적 모드 선택을 위한 자동화·메타 학습 기법 개발  

이 논문은 **사전학습 목적 함수 설계**와 **모드 전환** 개념을 통해 범용 자연어 처리 모델의 가능성을 열었으며, 후속 연구에서 다양한 과제 통합·절충(point-of-need) 학습 전략이 활발히 탐구될 것으로 기대된다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/f440f6d6-df89-44cf-8567-6f018d4ab056/2205.05131v3.pdf)
