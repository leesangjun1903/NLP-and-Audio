# TernaryBERT: Distillation-aware Ultra-low Bit BERT

**핵심 주장 및 주요 기여**  
TernaryBERT는 BERT 모델의 무게를 {–1, 0, +1} 세 값으로 **초저비트(2-bit)로 양자화**하면서, 지식 증류(knowledge distillation)를 통합해 **성능 손실을 최소화**한 점이 핵심이다.  
1. **양자화 기법**: 근사 기반(TWN)과 손실 인지 기반(LAT) ternarization을 도입하고, 단어 임베딩에는 row-wise, Transformer 가중치에는 layer-wise granular quantization을 적용.  
2. **지식 증류 통합**: Transformer 계층의 은닉 표현과 어텐션 스코어, 최종 로짓을 교사 모델(BERT)과의 MSE 및 soft cross-entropy로 맞추어, 14.9× 모델 크기 축소에도 Full-precision BERT와 유사한 성능을 달성.  
3. **활성화 양자화**: 8-bit min–max quantization을 사용해, 비대칭 분포인 은닉 표현에 대응.  

***

## 1. 해결 문제  
표준 BERT-base(109M 매개변수, 400MB 이상)는 모바일·임베디드 장치에 적용하기 어려울 만큼 **메모리와 연산 비용**이 크다.  
- 1–2비트 급의 초저비트 양자화 시 **모델 용량은 대폭 줄지만**, 성능 저하가 매우 심각함.  
- Mixed-precision(3–8bit) quantization은 하드웨어 호환성 문제 존재.  

TernaryBERT는 ultra-low bit 양자화(2-bit)와 지식 증류를 결합해, **경량화와 성능 유지**를 동시에 달성하고자 한다.

***

## 2. 제안 방법  
### 2.1 가중치 ternarization  
양자화된 가중치 $$\hat w\in\{-1,0,+1\}^n$$를 스케일 파라미터 $$\alpha>0$$와 이진 텐서 $$b$$의 곱으로 표현:  

$$
\hat w = \alpha\,b,\quad b_i\in\{-1,0,+1\}.
$$  

- **TWN (Approximation-based)**:  

```math
  \min_{\alpha,b} \|w - \alpha b\|_2^2,\quad b = I_\Delta(w),\ \alpha = \frac{\|b\odot w\|_1}{\|b\|_1}.
```

- **LAT (Loss-aware)**:  

$$
  \min_{\alpha,b} L(\alpha b)\quad\text{s.t. }b\in\{-1,0,1\}^n
  $$  
  
Adam의 2차 모멘트 $$v_t$$를 이용해 가중치 업데이트.  

- **Granularity**  
  - 단어 임베딩: row-wise ternarization → 각 행별 스케일링   
  - Transformer 가중치: layer-wise ternarization → 행렬별 스케일링  

### 2.2 활성화 양자화  
은닉 표현의 분포가 비대칭(-쪽으로 치우침)이므로, 8-bit **min–max quantization** 적용:  

$$
Q_a(x) = \mathrm{round}\bigl((x - x_\mathrm{min})/s\bigr)\times s + x_\mathrm{min},\quad s = \tfrac{x_\mathrm{max}-x_\mathrm{min}}{255}.
$$

### 2.3 지식 증류(Distillation-aware Ternarization)  
교사 모델(full-precision BERT)로부터 은닉층, 어텐션 스코어, 최종 로짓을 전이:  

$$
L_{\mathrm{trm}} = \sum_{l=1}^{L+1}\mathrm{MSE}(H^S_l, H^T_l) + \sum_{l=1}^{L}\mathrm{MSE}(A^S_l, A^T_l),\quad
L_{\mathrm{pred}} = \mathrm{SCE}(P^S, P^T),
$$  

$$
L = L_{\mathrm{trm}} + L_{\mathrm{pred}}.
$$  

은닉·어텐션 증류로 Transformer 내부 표현을, 로짓 증류로 최종 예측을 보강하여 양자화 성능을 극대화.

***

## 3. 모델 구조  
기본적으로 BERT-base(12-layer Transformer) 구조를 따르며,  
- 모든 선형 계층 가중치와 단어 임베딩을 ternarization  
- 어텐션·FFN 내부 활성화는 8-bit quantization  
- LayerNorm, Softmax, Task 헤드는 양자화 제외  

학습 절차는 Algorithm 1로 요약된다.

***

## 4. 성능 평가  
주요 벤치마크: GLUE, SQuAD v1.1/v2.0  

| 모델                     | W-E-A(bits) | 크기(MB) | MNLI-m/mm | SST-2  | CoLA  | RTE   | SQuAD v1.1 EM/F1 |
|-------------------------|-------------|----------|-----------|-------|-------|-------|------------------|
| BERT (32-32-32)         | 32-32-32    | 418      | 84.5/84.9 | 93.1  | 58.1  | 71.1  | 81.5/88.7        |
| TernaryBERT (TWN)       | 2-2-8       | 28 (×14.9) | 83.3/83.3 | 92.8  | 55.7  | 72.9  | 79.9/87.4        |
| TernaryBERT (LAT)       | 2-2-8       | 28       | 83.5/83.4 | 92.5  | 54.3  | 72.2  | 80.1/87.5        |

- **GLUE**: 14.9× 축소에도 MNLI-m 83.5%, SST-2 92.5% 달성.  
- **SQuAD**: EM/F1 80.1/87.5로 Full-precision 대비 1.4pt/1.2pt 하락에 그침.  
- Ablation: 지식 증류 제거 시 CoLA·RTE 3–5%p 하락, row-wise 임베딩이 layer-wise 대비 성능 우세.

***

## 5. 한계 및 일반화 성능 관점  
- **한계**:  
  - 2-bit 표현의 제약으로 복잡한 문장 구조나 희소 표현 처리에서 성능 손실 가능  
  - Layer-wise ternarization이 일부 민감 계층에서 표현력 부족  
- **일반화 성능**:  
  - 지식 증류가 은닉 표현 전반을 보강하므로, **언어 이해 태스크 전반에 걸쳐 강건성** 확보  
  - TinyBERT와 결합 시, 더 깊은 경량화 모델에도 양호한 전이 성능 확인  

***

## 6. 향후 연구 영향 및 고려 사항  
- **혼합 정밀도 확장**: 3-bit 이상과 결합해 하드웨어 특성에 최적화된 mixed-precision quantization 연구  
- **동적 양자화**: 입력 문장 길이나 복잡도에 따라 양자화 비트 수를 적응적으로 조정하는 방법  
- **증류 대상 다양화**: 어텐션 패턴 외에 파라미터 중요도, 언어적 구조 정보까지 증류하는 **멀티뷰 증류(Multi-view Distillation)**  
- **하드웨어 통합**: 실제 모바일/엣지 디바이스에서 latency·전력 효율을 측정하고, 양자화 연산을 가속화하는 전용 커널 설계  

TernaryBERT는 경량화 NLP 모델 설계에 있어 **양자화와 증류 결합**의 새로운 가능성을 제시하며, 향후 초저비트 모델의 실용적 적용과 추가적인 일반화 성능 강화 연구에 핵심적 토대를 제공할 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/e3ffb876-c87d-4500-9a1d-334eb31ff45d/2009.12812v3.pdf
