# Outlier Suppression: Pushing the Limit of Low-bit Transformer Language Models

## 1. 핵심 주장 및 주요 기여 (간결 요약)
이 논문은 Transformer 기반 NLP 모델의 저비트(6-bit 이하) 양자화(Post-Training Quantization, PTQ) 시 성능 저하의 주원인인 **구조적 아웃라이어(outliers)** 문제를 근본적으로 해결하기 위해, LayerNorm의 스케일 파라미터 γ가 아웃라이어를 증폭시키는 원인임을 규명하고 이를 제거·억제하는 두 단계의 **아웃라이어 억제(outlier suppression) 프레임워크**를 제안한다.  
주요 기여:  
- γ를 분리해 후속 모듈로 이동시키는 **Gamma Migration**  
- 토큰별 동적 범위에 따라 클리핑 범위를 효율적으로 찾는 **Token-Wise Clipping**  
이를 통해 BERT, RoBERTa, BART 등 다양한 모델에서 **6-bit PTQ** 정확도를 FP32 수준으로 회복하며, 4-bit 양자화 학습(QAT)에서도 FP32에 근접한 성능을 달성한다.

## 2. 해결 문제 및 제안 방법

### 2.1 해결하고자 하는 문제
- **저비트 양자화 시 큰 성능 저하**: 표준 6-8-bit PTQ 또는 4-bit QAT 적용 시, LayerNorm·GELU 출력의 극단적 아웃라이어로 인해 정량화 오차가 크게 증가, GLUE 및 SQuAD 성능이 10% 이상 하락.

### 2.2 Gamma Migration
- **문제 원인 분석**: LayerNorm 출력 $$\tilde X_{t,j} = \gamma_j \frac{X_{t,j}-\mu_t}{\sqrt{\sigma_t^2+\epsilon}} + \beta_j$$에서 γ가 특정 임베딩 차원에 걸쳐 아웃라이어를 공통 증폭함.
- **변환**: γ를 분리해 비스케일링 LayerNorm으로 교체  

```math
    X'_{t,j} = \frac{X_{t,j}-\mu_t}{\sqrt{\sigma_t^2+\epsilon}} + \frac{\beta_j}{\gamma_j},\quad
    \tilde X_{t,j} = \gamma_j \,X'_{t,j}
```

- **Equivalent Migration**: 잔차 연결 및 FFN/Attention의 가중치에 γ를 흡수하여 FP 연산 동일성 유지.  
- **효과**: LayerNorm 출력 범위가 $$|\max(\gamma)|$$배 축소되어 양자화 친화적 분포 생성.  

### 2.3 Token-Wise Clipping
- **아웃라이어 클리핑의 중요도 편차**: 일부 극단 아웃라이어(10–100)는 전체 정확도에 미치는 영향이 미미하며, 이들은 소수 토큰에만 존재.  
- **수식적 목표**:  

$$
    L(s) = \|\hat f(s) - f\|_F^2,\quad
    \hat f(s)=\mathrm{DeQuant}(\mathrm{clip}(\mathrm{Quant}(f, s),\,s))
  $$

- **Coarse Stage**: 각 토큰별 최대·최소값 집합 $$\{o_u, o_l\}$$에서 α-quantile로 클리핑 범위 결정 → 초기 step size $$s_0$$ 탐색 (그리드 서치).  
- **Fine Stage**: $$s \leftarrow s - \eta\,\partial L/\partial s$$로 미세 조정.  
- **효율성**: 전체 값이 아닌 축소된 토큰 대표값으로 빠른 탐색, 불필요한 영역 신속 건너뜀.

## 3. 모델 구조 및 성능 향상
- **적용 모델**: BERT-Base, RoBERTa-Base, BART-Base  
- **기존 최첨단 대비**:  
  - 6-bit PTQ에서 GLUE 평균 정확도 FP32 대비 2.6% 이내 근접 (종전 12% 이상 하락)  
  - 4-bit QAT에서 GLUE 평균 정확도 FP32 대비 ≈2.7% 차이 (LSQ+ 대비 3–4% 개선)  
  - SQuAD-v1.1 F1: 6-bit PTQ로 BERT 88.28→84.48, RoBERTa 92.25→80.79→Ours 91.57[84.86] (↑1.09)  
  - Summarization ROUGE-L: BART 42.88→38.51→Ours 43.45 (8-bit), 34.92→38.51 (6-bit)  
- **한계**:  
  - γ 이동 과정에서 모델 구조에 따른 세부 마이그레이션 필요  
  - 토큰별 통계 기반이므로 비표준 토크나 언어 특이 토큰에 대한 민감도 검증 추가 필요  

## 4. 일반화 성능 향상 가능성
- **LayerNorm 아웃라이어 억제**는 NLP 외 **비전, 음성 분야의 Transformer**에도 적용 가능  
- **토큰별 클리핑** 개념은 BPE, SentencePiece 등 다양한 토크나이저 및 희소 토큰에도 확장 가능  
- **QAT 초기화**로서 좋은 step size를 제공, 수렴 가속 및 안정성 개선  
- **사전학습(Pre-training) 단계**에도 γ 분리·클리핑 적용 시, 대규모 파라미터의 일반화 바이어스 조절 기대  

## 5. 향후 연구 방향 및 고려 사항
1. **다른 도메인 적용 검증**: Vision Transformer, 음성 인식 모델에서 아웃라이어 특성 분석 및 동일 프레임워크 적용  
2. **동적 γ 적응**: 학습 중 γ 재학습(scoping)으로 토크 빈도 편향 완화  
3. **토큰 임베딩 리프리젠테이션 개선**: 빈도 편향 제거한 임베딩과 클리핑 결합  
4. **양자화 친화적 사전학습**: 아예 사전훈련 단계에서 Non-scaling LayerNorm 활용  
5. **하드웨어 최적화**: γ 이동 및 토큰별 클리핑이 하드웨어 ReQuant 지원 환경에서 효율적 구현 여부 검증

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/d5b1a561-c5ba-4b70-8d67-33c28f1cabab/2209.13325v3.pdf
