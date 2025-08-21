# Understanding and Overcoming the Challenges of Efficient Transformer Quantization

## 주요 주장 및 기여
이 논문은 대규모 Transformer 기반 언어 모델(BERT 등)의 추론 효율화를 위해 **저비트 양자화(quantization)** 기법을 탐구하며, 다음의 핵심 기여를 제시한다.  
1. **양자화의 한계 규명**: 표준 8-bit 후처리 양자화(Post-Training Quantization, PTQ)가 Transformer 모델의 성능을 크게 저하시킨다는 사실을 실험적으로 입증.  
2. **동적 범위 문제 조사**: 잔차 연결(residual connection) 내부 활성화의 **구조화된 이상치(outlier)** 가 낮은 비트 정밀도로는 표현이 어렵고, 이로 인해 특정 토큰([SEP])에 과도하게 주목(attention)하는 양상을 보인다는 분석.  
3. **세 가지 해결책 제안**  
   - **혼합 정밀도 PTQ (Mixed-Precision PTQ)**: 민감한 연산(예: FFN의 입력·출력·합산)만 16-bit로 유지하고 나머지는 8-bit로 양자화  
   - **Embedding-그룹별 양자화 (Per-Embedding-Group Quantization)**: 활성화 텐서를 d차원 embedding 축을 따라 K개 그룹으로 나누어 각 그룹별로 별도 scale·zero-point를 적용  
   - **양자화 인지 학습(QAT)**: 훈련 과정에서 양자화 오차를 시뮬레이션하여 모델이 정량화 잡음에 적응하도록 fine-tuning  
4. **초저 비트 실현**: BERT-base에서 2-4 bit로 가중치 및 임베딩을 양자화해 메모리 8× 절감하면서 GLUE 벤치마크 성능 저하를 1% 미만으로 최소화  

***

## 문제 정의 및 제안 방법

### 1. 해결하고자 하는 문제
- **추론 비용과 메모리 제약**: BERT-like 모델은 수백만~수십억 개의 파라미터로 구성되어, 모바일·임베디드 장치에 실용적이지 않음.  
- **양자화 시 성능 급락**: 8-bit PTQ 적용 시, GLUE 과제 중 MNLI, QNLI, RTE 등에서 정확도가 절반 이하로 떨어짐.  

### 2. 동적 범위의 불일치와 이상치(outlier)
- FFN의 입력과 출력 활성화 사이에 **수십 배에 이르는 dynamic range 차이** 발생  
- 출력값 중 일부 embedding 차원이 극단값을 보여, 평균 ± 6σ를 초과하는 이상치(patterned outlier) 다수 존재  
- 이 이상치가 다음 어텐션 레이어에서 특정 헤드가 [SEP] 토큰으로 집중(attention collapse)되도록 유도  

### 3. 제안 기법

#### 3.1 혼합 정밀도 PTQ (Mixed-Precision PTQ)
- FFN 입력·출력·합산 연산만 16-bit 고정, 나머지 연산은 8-bit로 실행  
- 수학적 표현:  

$$ \tilde{x} = s_w x^{(Z)} \odot s_a a^{(Z)}, \quad s_w, s_a \in \mathbb{R}^+, x^{(Z)}, a^{(Z)} \in \{0,\dots,2^b-1\} $$  

- 실험 결과, 전체 활성화 중 22%만 16-bit로 유지해도 FP32 대비 GLUE 점수를 1% 이내로 회복  

#### 3.2 Per-Embedding-Group Quantization (PEG-PTQ)
- d차원 활성화 텐서 x∈ℝ^(B×T×d)를 K개 그룹으로 분할하여 각 그룹 g마다 별도 scale s_g, zero-point z_g 설정  
- 그룹별 재스케일링 비용은 d→K로 획기적 절감  
- 이상치 집중 차원끼리 같은 그룹에 묶기 위해 **range-based permutation** 적용  
- 실제 HW에서 per-tensor 연산만 지원할 때는, 그룹별 Linear 계층 분할·재조합으로 기능적 동등성 보장  

#### 3.3 Quantization-Aware Training (QAT)
- 훈련 중에 양자화 연산을 모사하여 모델이 정량화 잡음에 적응  
- scale·zero-point를 학습 가능하도록 설정, straight-through estimator(STE) 활용  

***

## 모델 구조 및 성능 개선

| 기법                    | GLUE 점수     | 메모리 절감 | 주요 특징                                          |
|-----------------------|-------------|----------|-------------------------------------------------|
| Baseline FP32         | 83.06       | ×1.0     | —                                               |
| PTQ W8A8              | 71.03       | ×4       | 활성화 이상치 문제 미해결                            |
| MP-PTQ (W8A{8,16})     | 82.43       | ×3.9     | FFN 연산 16-bit로 개선                              |
| PEG-PTQ (W8A8)         | 82.45       | ×4       | K=6 그룹, range-based permutation 적용              |
| QAT (W8A8)            | 83.26       | ×4       | 양자화 잡음에 적응, PTQ 대비 1%p 이상 호전                |
| Ultra-Low (W4A8+2bit embd.) | 82.29       | ×8.85    | 가중치 4-bit, 임베딩 2-bit 양자화                      |

- **혼합 정밀도**와 **PEG-PTQ**는 PTQ임에도 불구하고 GLUE 성능을 FP32 대비 1%p 이내로 유지  
- **QAT**는 추가 미세조정으로 FP32 수준까지 회복  
- **초저 비트** 가중치·임베딩 양자화 시에도 평균 0.8%p 이내 감소  

***

## 한계 및 일반화 성능 고려

- **하드웨어 지원 제약**: PEG-PTQ는 일부 타깃에서 추가 연산 필요  
- **비트 선택 트레이드오프**: 16-bit 연산 비율 증가 시 속도·에너지 이득 축소  
- **일부 다운스트림 태스크 취약**: CoLA 등 문법적 추론 과제에서는 여전히 2-3%p 성능 저하  
- **모델 일반화**:  
  - **양자화 잡음 적응**: QAT 과정이 다양한 입력 분포에 대한 내부 표현을 견고하게 만들어 일반화 성능 향상에 기여할 가능성  
  - **그룹화 기반 정밀도 제어**: PEG-PTQ는 이상치 기반 차원 선택이므로, 도메인 적응 시 특이 토큰 분포에도 유연하게 대처할 여지  

***

## 향후 연구 방향 및 고려사항

1. **하드웨어 친화적 구현**: PEG-PTQ를 지원하는 경량 라이브러리 및 accelerator 명세 개발  
2. **자동 비트폭 결정**: Mixed-Precision 설정을 태스크·레이어별로 자동 검색하는 NAS 연계 기법  
3. **Activation Regularization**: 사전 학습 단계에서 이상치 억제를 위한 정규화 추가로 PTQ 민감도 완화  
4. **도메인·태스크 일반화**: QAT 기반 양자화가 드문 데이터셋이나 OOD 상황에서도 표현 학습 견고성에 미치는 효과 검증  
5. **비트별 에너지·지연 분석**: 실제 장치에서 측정한 전력·지연 데이터를 활용하여 최적의 정밀도-효율 트레이드오프 설정  

이 논문은 대규모 Transformer 모델을 낮은 비트 정밀도로 안정적으로 양자화하는 새로운 패러다임을 제안함으로써, **효율적 NLP 추론** 연구의 중요한 이정표를 세웠다. 향후 하드웨어-소프트웨어 공동 최적화, 자동화된 정밀도 탐색, 그리고 정규화 기반 사전 학습 확장이 활발히 이루어질 것으로 전망된다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/a91c653c-af6a-481e-b0ec-19f819380dbe/2109.12948v1.pdf
