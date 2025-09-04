# AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks

# 핵심 요약 및 주요 기여

**주요 주장:**  
AASIST는 스펙트로-템포럴(주파수 및 시간) 정보를 통합적으로 모델링하여 단일 경량 시스템으로 광범위한 오디오 스푸핑 공격을 효과적으로 탐지하는 새로운 오디오 안티-스푸핑 모델이다.

**주요 기여:**  
1. **이종 스태킹 그래프 어텐션 레이어(HS-GAL):** 서로 다른 노드 수와 차원을 가지는 스펙트럴 및 템포럴 서브그래프를 공통 잠재공간으로 매핑하고, 이들을 이종 어텐션 메커니즘과 추가 스택 노드를 통해 통합적으로 모델링.  
2. **맥스 그래프 연산(MGO):** 두 개의 병렬 브랜치에서 HS-GAL과 그래프 풀링을 거쳐 얻은 노드 표현에 대해 원소별 최대 연산을 수행, 서로 다른 종류의 스푸핑 아티팩트를 경쟁적으로 선택.  
3. **개선된 리드아웃(readout) 스킴:** 마지막 히든 레이어에서 노드별 평균, 최대, 스택 노드를 연결하여 최종 이진 분류를 수행.  
4. **경량화 버전(AASIST-L):** 85K 파라미터만으로도 최첨단 성능을 유지하는 모델을 제안하여 임베디드 환경 적용 가능성을 제시.

***

## 1. 문제 정의  
자동 스피커 검증 시스템에 대한 스푸핑 공격(음성 변환 및 텍스트-투-스피치 기반)의 탐지는 실제 응용에서 신뢰성을 결정짓는 핵심 요소이다. 기존 연구들은 스펙트럴 또는 템포럴 도메인의 아티팩트에 특화된 앙상블 시스템에 의존했으나, 계산 비용이 크고 다양한 공격을 단일 모델로 처리하기 어려웠다.

***

## 2. 제안 방법 상세

### 2.1. HS-GAL 구성  
HS-GAL은 두 종류의 그래프 $$G_s \in \mathbb{R}^{N_s \times D_s}$$, $$G_t \in \mathbb{R}^{N_t \times D_t}$$를 다음과 같이 통합하여 입력한다.  
1. **잠재공간 투영:**  

```math
     \widetilde{G}_s = W_s G_s + b_s,\quad
     \widetilde{G}_t = W_t G_t + b_t,\quad
     \widetilde{G}_s,\widetilde{G}_t \in \mathbb{R}^{N\times D_{st}}
```

2. **이종 어텐션:**  
   - **동일 서브그래프 간 어텐션:** $$\alpha_{ii} = \mathrm{softmax}\bigl(\mathrm{LeakyReLU}(a_i^\top(\widetilde{h}_i\odot\widetilde{h}_j))\bigr)$$  
   - **서브그래프 간 어텐션:** $$a_{st}^\top(\widetilde{h}\_s\odot\widetilde{h}\_t)$$  
   각기 다른 프로젝션 벡터 $$a_s, a_{st}, a_t \in \mathbb{R}^{D_{st}}$$ 사용.  
3. **스택 노드:**  
   $$\text{stack}$$ 노드를 도입하여 모든 노드로부터 유니디렉셔널 연결만을 수신, 서브그래프 간의 관계 정보를 누적.

### 2.2. 맥스 그래프 연산(MGO)  
두 개의 병렬 브랜치 각기 HS-GAL → 그래프 풀링 → HS-GAL → 그래프 풀링을 거친 후, 각 노드 표현과 스택 노드에 대해  

$$
  H_{\text{out}} = \max\bigl(H_{\text{branch1}},\,H_{\text{branch2}}\bigr)
$$  

를 수행하여 경쟁적으로 중요한 아티팩트를 선택.

### 2.3. 리드아웃 및 분류  
최종 그래프에서 노드별 평균 $$\mathrm{MeanPool}(H_{\text{out}})$$, 최대 $$\mathrm{MaxPool}(H_{\text{out}})$$, 스택 노드를 연결(concatenate)한 후, 완전연결층으로 이진(보나파이드/스푸핑) 출력.

***

## 3. 모델 구조 개요  
```
Raw Waveform → RawNet2-based Encoder → 
    ├─ Spectral Graph Module → G_s
    ├─ Temporal Graph Module → G_t
    ↓
  HS-GAL Projection & 이종 어텐션 → Max Graph Operation (MGO) → Readout → Output
```
- **Encoder:** 64,600 샘플 입력, 6개의 ResBlock, sinc-conv(70필터), SeLU, BatchNorm, MaxPool  
- **그래프 모듈:** HS-GAL(64→32 채널), Graph Pool(스펙트럴 50%, 템포럴 30% 노드 축소)  
- **분류기:** Node mean, max, stack 노드의 3가지 표현을 FC로 결합

***

## 4. 성능 및 한계

### 4.1. 성능 향상
- **ASVspoof 2019 LA** 데이터셋 평가:  
  - **AASIST:** pooled EER 0.83%, min t-DCF 0.0275 (기존 대비 약 20% 상대 개선)  
  - **AASIST-L:** 85K 파라미터로도 EER 0.99%, min t-DCF 0.0309, 대부분 모델 성능 능가  
- **어블레이션:** HS-GAL의 이종 어텐션, 스택 노드, MGO 모두 제거 시 성능 저하 확인

### 4.2. 한계
- **데이터 의존성:** LA 시나리오(음성 변환·TTS)에 최적화되어 있어 리얼월드 녹음 환경 및 물리 접근 공격(리플레이) 일반화 불확실  
- **계산 복잡성:** AASIST-L 경량화는 성공했으나, AASIST 본 모델은 300K 파라미터로 하드웨어 제약이 있는 환경에서 적용 제한

***

## 5. 일반화 성능 향상 가능성  
- **스택 노드와 이종 어텐션** 구조는 다른 멀티도메인(예: 스펙트로그램+피치, 음성+비디오) 융합 작업에도 확장 가능  
- **MGO 메커니즘**을 통해 다양한 도메인별 특징 경쟁 모델에 적용하여 과적합 방지 및 로버스트니스 증진 전망  
- **경량화 기법**(파라미터 프루닝, 양자화) 결합 시 모바일·엣지 환경으로 일반화 가능

***

## 6. 향후 연구 방향 및 고려사항  
- **물리 접근 스푸핑(리플레이) 확장:** LA 외 PA 시나리오로 HS-GAL·MGO 검증  
- **다중 언어·다중 환경 테스트:** 노이즈·방향성 마이크환경 일반화 성능 평가  
- **도메인 어댑테이션:** 도메인 불일치 환경에서의 스푸핑 탐지 성능 안정화 기법 개발  
- **경량화 최적화:** 하프 프리시전 인퍼런스·네트워크 프루닝·NAS 결합으로 더욱 작은 모델 탐색

AASIST는 통합적 그래프 어텐션과 경쟁적 맥스 연산을 통해 오디오 스푸핑 탐지 분야의 새로운 방향을 제시하며, 다중 도메인 특징 융합 및 경량화에 기반한 향후 연구에 큰 영향을 미칠 것으로 기대된다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/f800fad4-a7a5-4b33-a1eb-543e32d538d0/2110.01200v1.pdf)
