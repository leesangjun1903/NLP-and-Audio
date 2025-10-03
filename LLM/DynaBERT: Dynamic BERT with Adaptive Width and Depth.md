# DynaBERT: Dynamic BERT with Adaptive Width and Depth

**주요 주장**  
DynaBERT은 BERT 모델을 **가변적 너비(width)**와 **가변적 깊이(depth)**로 조정하여, 단일 모델에서 다양한 하드웨어 성능 및 동적 환경 제약에 적응 가능한 서브네트워크(sub-networks)를 제공한다. 이는 고정 크기로 압축된 기존 BERT 압축 기법과 달리, 한 번의 훈련으로 폭넓은 효율성-정확도 트레이드오프를 실현할 수 있다는 점을 핵심 기여로 제시한다.[1]

**주요 기여**  
1. **Adaptive Width & Depth**: Transformer의 각 레이어에서 주의(attention) 헤드 수 $$N_H$$와 FFN 중간층 뉴런 수 $$d_{ff}$$를 조절하여 너비를, 레이어 수 $$L$$를 조절하여 깊이를 동적으로 변경 가능하도록 설계.[1]
2. **Two-Stage Distillation**:  
   a. 너비 적응 전용 모델 $$DynaBERT_\text{W}$$를 먼저 훈련하고, 이로부터 가변 너비·깊이 모델 DynaBERT로 지식 증류(distillation)를 수행.[1]
   b. 중요도가 높은 헤드·뉴런을 더 많은 서브네트워크가 공유하도록 **network rewiring** 기법을 적용.[1]
3. **No Per-Subnetwork Fine-Tuning**: 최종 훈련 이후 서브네트워크별 추가 미세조정 없이 바로 배포 가능.[1]
4. **Regularization 효과**: 훈련 난이도 증가가 오히려 일반화 성능 향상에 기여함을 실험적으로 확인.[1]

***

# 문제 정의 및 제안 방법

## 해결하고자 하는 문제  
- **다양한 엣지 디바이스 환경**: 장치별 연산·메모리 성능 차이로 단일 고정 크기 BERT로 대응 불가.  
- **동적 자원 제약**: 실행 시점에 따라 사용 가능한 자원 변동으로, 모델 크기 및 지연시간을 실시간 조정할 필요.

## 제안 방법  
### 1단계: 너비 적응 $DynaBERT_\text{W}$ 훈련  
- 입력 행렬 $$X\in\mathbb{R}^{n\times d}$$일 때,  
  - **Multi-Head Attention (MHA)**:  

$$ \text{MHAttn}(X)=\sum_{h=1}^{N_H}\text{Attn}_h(X)\,,\quad \text{Attn}_h(X)=\text{Softmax}\!\bigl(\tfrac{XW^Q_h (XW^K_h)^\top}{\sqrt{d_h}}\bigr)\,XW^V_hW^O_h. $$  
  
  - **Feed-Forward Network (FFN)**:  

$$ \text{FFN}(A)=\sum_{i=1}^{d_{ff}}\mathrm{GeLU}(A W_{1,i}+b_{1,i})\,W_{2,i}+b_{2,i}. $$  

- 너비 배율 $$mw$$에 따라 상위 $$mwN_H$$개의 헤드와 $$mw\,d_{ff}$$개의 뉴런만 남김.[1]
- **Network Rewiring**: 제거 시 손실 변화량에 기반해 헤드·뉴런 중요도 점수 계산 후 정렬하여 중요한 부분을 서브네트워크에 우선 배치.[1]
- **지식 증류 손실**:  

$$ \mathcal{L}=\lambda_1\!\,\mathrm{SCE}(y_{mw},y)+\lambda_2\,\bigl\|\!E_{mw}-E\bigr\|_2^2+\sum_{l=1}^L\bigl\|\!H^l_{mw}-H^l\bigr\|_2^2, $$  
  
  여기서 $$\mathrm{SCE}$$는 소프트 크로스 엔트로피, $$E$$, $$H^l$$는 교사(teacher) 임베딩과 레이어 출력.[1]

### 2단계: 너비·깊이 적응 DynaBERT 훈련  
- 학습된 $DynaBERT_\text{W}$ 를 고정 교사로 사용하여, 깊이 배율 $$md$$에 따른 레이어 드롭(drop) 및 **Every-Other** 전략으로 서브네트워크 생성.[1]
- 남은 레이어 쌍을 교사 모델의 대응 계층과 매칭하여 은닉 상태(distillation) 수행.[1]

***

# 모델 구조 및 성능 향상

## 모델 구조  
- 기본 백본: $BERT_\text{BASE}$ (12층, 히든 768, 헤드 12, FFN 3072) 또는 $$RoBERTa_\text{BASE}$$.  
- 너비 배율: $$\{1.0,0.75,0.5,0.25\}$$, 깊이 배율: $$\{1.0,0.75,0.5\}$$ → 총 12개 구성 서브네트워크.

## 성능 비교  
- GLUE: 최대 크기에서는 $BERT_\text{BASE}$ 대비 동등 성능, 소형 서브네트워크에서 DistilBERT·TinyBERT 초과.[1]
- SQuAD v1.1: 파라미터·FLOPs 동일 시 EM/F1 모두 TinyBERT·DistilBERT 상회.[1]
- **일관된 우위**: 다양한 파라미터, FLOPs, GPU/CPU 지연 측면의 효율성 제약 하에서도 종합 성능이 타 기법 대비 우수.[1]

***

# 일반화 성능 향상 및 한계

## 일반화 성능  
- **Regularization 효과**: Adaptive width·depth 훈련이 과잉적합 방지 역할, 최고 크기 서브네트워크가 종종 원본 $$BERT_\text{BASE}$$를 소폭 능가.[1]
- **Attention 패턴 융합**: 중간층에서 언어적 기능(위치·구문·의미) 융합 현상이 확인되어, **다양한 서브모델 실행에도 핵심 표현력 유지**.[1]

## 한계  
1. **훈련 비용**: Two-stage distillation 및 rewiring 단계로 인해 학습 시간이 증가.  
2. **구현 복잡도**: 다양한 서브네트워크 관리·배포를 위한 프레임워크 지원 필요.  
3. **적용 범위**: 제안은 인코더 전용 BERT 구조 대상이며, 생성 모델(GPT 등) 확장 시 추가 고려사항 존재.

***

# 향후 연구 방향 및 고려사항

- **생성 모델로의 확장**: GPT 계열에 adaptive depth·width 기법 적용 및 출력 품질·정책 준수 검증.  
- **자동 하드웨어 배치 최적화**: 런타임 환경에 따른 서브모델 선택 정책 자동화 및 오케스트레이션 연구.  
- **훈련 효율 개선**: 한 번의 학습 단계에서 가변 구조를 효율적으로 지원하는 경량화 알고리즘 개발.  
- **다중 모달·다국어 모델**: 비전-언어 및 다국어 사전학습 모델로 확대하여 범용성 검증.

DynaBERT은 **유연한 모델 규모 조정**과 **일관된 성능 유지**라는 두 마리 토끼를 잡은 혁신적 접근으로, 엣지 환경에서의 대규모 언어 모델 활용 가능성을 크게 확장한다.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/54e2bc94-165c-4ae6-b8dd-43ab91bc7c8c/2004.04037v2.pdf)
