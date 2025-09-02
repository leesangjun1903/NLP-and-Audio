# One Model To Learn Them All 

**핵심 주장 및 주요 기여**  
“One Model To Learn Them All” 논문은 *단일 모델*이 **다양한 도메인(task)** 에서 동시에 학습할 수 있음을 보여준다.  
- **모두 공유되는 ‘Unified Representation’**: 텍스트, 이미지, 음성, 파싱 등 서로 다른 입력·출력 모달리티를  
  변환하는 *modality nets*를 통해 하나의 공통된 표현 공간으로 매핑.  
- **도메인-특화 연산 블록의 결합**:  
  - Depthwise-separable convolution  
  - Multi-head attention  
  - Sparsely-gated mixture-of-experts  
  세 가지 주요 블록을 결합하여, 각 도메인에 특화된 성능을 유지하면서도 서로 다른 태스크 간 **성능 저하 없이** 상호 전이(transfer)가 가능함을 입증.  
- **다중 테스크 Joint Training 이점**: 데이터가 적은 태스크(예: 파싱)는 다른 대규모 태스크(예: 번역, 이미지 분류)와 공동 학습 시  
  성능이 크게 향상되었고, 대규모 태스크 성능은 거의 손실되지 않음.  

***

## 1. 해결하고자 하는 문제  
- **단일 도메인-특화 모델의 한계**: 전통적 딥러닝 모델은 각 태스크별로 아키텍처 설계·튜닝이 필요.  
- **모델 일반화 부족**: 서로 다른 도메인 간 전이가 어려워, 도메인 간 *transfer learning* 활용도가 낮음.

***

## 2. 제안 방법

### 2.1. Unified Representation via Modality Nets  
- **입력 변환**:  

$$
    \text{ModalityIn}(x) = f_{\text{in}}(x) \quad\text{(learnable embedding 또는 ConvRes)}
  $$

- **출력 변환**:  

$$
    \text{ModalityOut}(h) = \begin{cases}
      \text{Softmax}(W_s h), & \text{(텍스트)}\\
      \text{PointwiseConv}(W_c, \text{GlobalAvgPool}(h)), & \text{(카테고리)}
    \end{cases}
  $$

### 2.2. 주요 연산 블록  
1. **Depthwise-Separable Convolution**:  

$$
     \text{ConvStep}\_{d,s,f}(W,x)=\mathrm{LayerNorm}(\mathrm{SepConv}_{d,s,f}(W,\mathrm{ReLU}(x)))
   $$  
   
   -  국소 패턴 감지에 효율적  
2. **Multi-Head Attention** ([Vaswani et al., 2017]):  
  

$$
     \mathrm{Attn}(Q,K,V)=\mathrm{softmax}\bigl(\tfrac{QK^\top}{\sqrt{d_k}}\bigr)V
   $$  
   
   -  위치 인식을 위한 *timing signal* 사용  
3. **Sparsely-Gated Mixture-of-Experts** ([Shazeer et al., 2017]):  
   -  k=4 전문가 선택, 240 experts 풀(pool)  
   -  대규모 모델 용량 확보  

### 2.3. 모델 구조  
- **Encoder**: 6× ConvBlock + MoE  
- **Mixer**: Attention + 2× ConvBlock  
- **Decoder**: 4× (ConvBlock+Attention) + MoE  
- **Autoregressive**: 왼쪽 패딩된 convolutions 사용  

***

## 3. 성능 향상 및 한계

| 문제        | Joint 8-task | Single-task | State-of-the-Art |
|-----------|------------:|-----------:|-----------------:|
| ImageNet  | 86% top-5   | —          | 95%            |
| WMT EN→DE | BLEU 21.2   | —          | BLEU 26.0      |
| Parsing   | 98% accuracy| 97%        | —              |

- **다중 테스크 도움이 큰 저자원 태스크**: 파싱 accuracy +1.8%p [Parsing alone→joint]  
- **대규모 태스크 안정적 성능**: ImageNet 성능 1%p 이내 손실  
- **한계**:  
  - 아직 태스크별 SOTA에는 미치지 못함  
  - 하이퍼파라미터 튜닝 부족  
  - 향후 더 많은 도메인·태스크로 확장 필요  

***

## 4. 일반화 성능 향상 관점  
- *모달리티 간 transfer* 입증: 파싱↔이미지 간 전이 효과[Parsing w/ ImageNet]  
- **컴퓨테이션 블록 공유**: 도메인 특화 블록을 모두 도메인에 상관없이 적용해도 성능 손실이 없고,  
  오히려 소폭 향상(예: ImageNet에 MoE·Attention 추가 시)  
- **데이터 균형**: 저자원 태스크에 풍부한 태스크 데이터가 이득  

***

## 5. 향후 연구 영향 및 고려사항  
- **범용 AI 모델**: 여러 도메인·태스크에 적용 가능한 *‘올인원’* 모델 설계 가능성 제시  
- **하이퍼파라미터·아키텍처 자동화**: 적응적 블록 조합, 태스크별 자동 튜닝  
- **확장성**: 더 많은 모달리티(예: 그래프, 시간 시계열) 적용  
- **효율성**: 연산량·메모리 절감 위한 전문가 수·attention head 조정  

*이 논문은 하나의 통일된 모델이 도메인 간 전이를 자연스럽게 달성할 수 있음을 보여주며, 앞으로 범용 딥러닝 아키텍처 연구에 중요한 이정표가 될 것이다.*

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/30b6b126-c0b4-4a00-8861-b55d5aade40a/1706.05137v1.pdf)
