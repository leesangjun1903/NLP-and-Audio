# Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality

## 1. 핵심 주장과 주요 기여  
**핵심 주장**  
이 논문은 기존의 *Transformers*와 최근 주목받는 *Structured State Space Models(SSMs)* 사이의 상호보완적 관계를 규명하고, 이를 기반으로 새로운 아키텍처 *Mamba-2* 및 효율적 연산 알고리즘 *SSD(State-Space Duality)*를 제안한다.  

**주요 기여**  
- **Structured State Space Duality(SSD) 프레임워크** 제안  
  -  SSM과 주의(attention) 메커니즘을 *반구조적 행렬*(semiseparable matrix) 관점으로 통합  
  -  이중(dual) 형태:  
    – 선형 모드: 시간 복잡도 $$O(NT)$$의 재귀 스캔  
    – 이차 모드: 행렬 곱으로 구현되는 유사 주의 연산  
- **효율적 SSD 알고리즘**  
  -  블록 분해를 통해 선형 성능과 행렬곱 최적화를 결합  
  -  훈련 시 FLOPs $$O(N^2T)$$, 추론 시 FLOPs/메모리 $$O(N^2)$$  
- **Mamba-2 아키텍처**  
  -  SSD 레이어, 병렬 파라미터 투영, 추가 정규화, 멀티-입력(다중 값) 헤드 구조 적용  
  -  Transformer 대비 긴 시퀀스에서 속도 및 확장성 우수  

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계  

### 2.1 해결 과제  
- Transformer의 $$O(T^2)$$ 비용(주의 행렬 계산)  
- SSM의 하드웨어 비효율성(재귀 스캔 기반) 및 복잡한 구현  

### 2.2 제안 방법  
1. **SSM ⇄ Attention의 행렬 관점 통합**  
   - SSM을 *N-순차적 세미분리(SSS)* 행렬로 표현  
   - 마스킹된 주의 연산을 *구조화된 마스크* 행렬(semiseparable)로 일반화  
2. **Dual Form 구현**  
   - 선형 모드(재귀 스캔):  

$$
       h_t = a_t h_{t-1} + b_t
     $$
   
   - 이차 모드(나이브 행렬곱):

$$
       Y = (L \circ (QK^\top))V,\quad L_{ij}=a_i\cdots a_{j+1}
     $$

3. **효율적 SSD 알고리즘**  
   - 시퀀스를 블록으로 분할해 대각/저(低)랭크 블록 연산 결합  
   - GPU 매트릭스 곱 최적화 활용  

### 2.3 모델 구조: Mamba-2 블록  
```text
Input u → 병렬 선형 투영 → {X, A, B, C} 
   → Conv1D(X) → SSD(A,B,C;Δ)(Conv1D(X)) 
   → 게이팅(·) 및 GroupNorm → 출력 투영 → u + 출력
```
- **병렬 투영**: A,B,C,X 생성 순서 병렬화 → 텐서-병렬 친화적  
- **추가 정규화**: 게이팅 후 안정성 개선  
- **헤드 구조**: Multi-Input SSM(=Multi-Value Attention)  

### 2.4 성능 향상  
- **효율성**:  
  – SSD, Mamba-2, FlashAttention-2 대비 2–8× 빠른 재귀 스캔  
  – 2K 시퀀스 이상에서 Transformer 주의보다 빠름  
- **언어 모델링**(Pile, 300B 토큰):  
  – Mamba 대비 Perplexity 1–2% 감소  
  – 동등 파라미터 규모 Transformer++ 대비 동급 또는 우수  
- **합성 과제**(MQAR):  
  – 상태 크기 $$N$$ 확장 시 복합 연상 과제 성능 대폭 향상  

### 2.5 한계  
- **표현력 제약**: SSD는 A 행렬을 스칼라·단위행렬로 제한 → 일부 복잡 동역학 부진  
- **짧은 시퀀스의 효율**: Transformer+MLP 대비 Mamba-2 블록 수 증가로 학습/추론 시 초기 오버헤드  
- **정규화 및 커널 추정 약함**: 다양한 커널 근사 실험에서 단순 Swish가 최적 → 소프트맥스 근사 과제  

## 3. 모델의 일반화 성능 향상 가능성  
- **상태 확장(State Expansion)**: $$\;N$$ 증가 시 메모리 용량 제어하에 장기 의존성 캡처 강화  
- **하이브리드 조합**: SSD+Attention+MLP 구성 시 Recall-Throughput 균형 최적화  
- **구조화된 행렬 프레임워크**: 새로운 행렬 구조(Toeplitz, Butterfly 등) 도입 가능 → 다양한 inductive bias 부여  
- **커널 피처 맵**: Swish 외 다른 양의 특성$$\phi$$ 연구로 비선형성 강화 및 generalization 개선  

## 4. 미래 연구 방향 및 고려 사항  
- **확장된 $$A$$ 구조**: 복소수 대각(diagonal complex) 또는 하이브리드 block-diagonal로 표현력·현실 효율성 균형  
- **비 인과적 변형**: Bidirectional SSD 설계로 요약·정보검색 등 다양한 과제 적용  
- **정규화·안정화 기법**: Residual·LayerNorm 위치 최적화, 게이팅 다변화 연구  
- **대화형·비교형 학습**: SSD 기반 대규모 파운데이션 모델에서 Few-Shot 일반화 및 In-Context 학습 성능  
- **하드웨어 통합**: 대규모 TPU/GPU 클러스터에 맞춘 Tensor/Sequence Parallelism 강화  

**결론**: 본 논문은 SSM과 Transformer의 경계를 허물어, 차세대 효율적·확장 가능한 시퀀스 모델 설계에 핵심 이정표를 제시했다. SSD와 Mamba-2는 긴 시퀀스 처리 및 차원 확장 면에서 뛰어난 가능성을 보이며, 후속 연구를 위한 다양한 영역(행렬 구조·하드웨어·아키텍처 설계)에서 풍부한 영감을 제공한다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/889a7773-8a98-4b46-a806-dd9b29f800e2/2405.21060v1.pdf)
