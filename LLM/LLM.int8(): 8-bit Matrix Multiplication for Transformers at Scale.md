# LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale

## 1. 핵심 주장 및 주요 기여  
LLM.int8() 논문은 **175억~1,750억 규모의 대형 트랜스포머 모델**을 8-bit 정밀도로 양(量) 또는 성능 손실 없이 바로 추론 가능하게 하는 **최초의 방법**을 제안한다.  
- **벡터 단위 양자화(vector-wise quantization)**: 각 행(row)·열(column)별로 별도 정규화 상수를 사용해 8-bit 범위 내에서 행렬 곱셈을 수행.  
- **혼합 정밀도 분해(mixed-precision decomposition)**: 전체 차원의 0.1%를 차지하는 고크기(outlier) 피처만 16-bit로 분리 처리하고, 나머지 99.9%는 8-bit로 처리하여 메모리 사용량을 절반으로 줄이면서도 성능 저하를 방지.  
- **결과**: OPT-175B, BLOOM-176B 등 대형 모델이 단일 서버·소비자용 GPU(A40, RTX3090 등) 상에서도 원본 정확도를 유지하며 추론 가능.

## 2. 해결하고자 하는 문제  
대형 언어 모델은 수십억~수백억 매개변수의 피드포워드 및 어텐션 프로젝션 레이어가 전체 파라미터의 95%·계산의 65–85%를 차지하여, 메모리·연산 부담이 매우 크다.  
- 기존 8-bit 양자화(절대값 최대 기준, row-/block-단위, zeropoint)는 6.7B 이상 규모에서 **고크기 특징값(outlier)** 등장으로 양자화 정밀도가 급격히 나빠져 성능 저하를 피할 수 없었다.

## 3. 제안 방법  
### 3.1 벡터 단위 양자화  
행렬곱 $$X^{f16}W^{f16}$$을 각 행 $$i$$와 열 $$j$$별로  

```math
X_{i}^{i8} = \left\lfloor \frac{127}{\max_k |X^{f16}_{ik}|}\,X^{f16}_{i} \right\rceil,\quad
W_{j}^{i8} = \left\lfloor \frac{127}{\max_k |W^{f16}_{kj}|}\,W^{f16}_{j} \right\rceil
```

로 각각 8-bit 정규화한 뒤, Int32 연산 후  

$$
\text{Out}^{f16} = \frac{1}{c^X_i\,c^W_j}\,\bigl(X^{i8}_i \cdot W^{i8}_j\bigr)
$$  

로 복원.

### 3.2 혼합 정밀도 분해  
전체 특성 차원 중 크기가 임계값 $$\alpha=6.0$$ 이상인 **0.1% 미만**의 outlier 차원 $$O$$만 16-bit로 분리 처리:  

```math
XW \approx \sum_{h\in O} X^{f16}_hW^{f16}_h \;+\;
S\!\!\sum_{h\notin O} X^{i8}_hW^{i8}_h
```

– outlier는 최대 7차원 이하로 희소·체계적으로 나타나므로 추가 메모리 오버헤드는 미미.

## 4. 모델 구조 및 성능 향상  
- **실험 모델**: OPT 시리즈(125M–175B), BLOOM-176B, GPT-2~GPT-J 등.  
- **평가 지표**: C4 언어모델링 검증 퍼플렉서티, EleutherAI 핫슈팅 zero-/few-shot 정확도.  
- **결과**: 모든 규모에서 16-bit FP 성능을 완벽 재현. 특히 6.7B 이상부터 기존 8-bit 방법은 퍼플렉시티·정확도 급락, LLM.int8()만 안정 유지.  

| 모델 규모 | 16-bit 퍼플렉시티 | 기존 8-bit 최선치 | LLM.int8() 퍼플렉시티 |
|:--------:|:----------------:|:----------------:|:--------------------:|
| 13B      | 12.45            | 16.48            | **12.45**            |
| 175B     | —                | 랜덤 수준        | **16-bit 동등**      |

- **메모리 절감**: 16-bit 대비 약 절반.  
- **추론 속도**: 6.7B 이상 대행렬 곱에서 1.6–2.3배 가속.  
- **한계**: attention 함수 자체는 8-bit 처리 미지원, 학습·미세조정(fine-tuning) 대상은 추후 연구 과제.

## 5. 일반화 성능 향상 가능성  
- **outlier 특성 보존**: 전체 차원의 극히 일부이지만 모델 성능 결정에 중요한 대규모 특징을 16-bit로 보전함으로써, 다양한 도메인·업스트림(task)에서도 **양자화 손실 없는 일관된 일반화**가 기대된다.  
- **zeropoint vs. symmetric quantization**: outlier는 한쪽 방향으로 치우친(zero-crossing 없는) 분포를 보이므로, asymmetric zeropoint 정밀도가 향상된다는 분석이 일반화에도 유효.

## 6. 향후 연구 영향 및 고려 사항  
- **FP8 등 차세대 저비트 유형**: 현재 GPU 미지원이지만, 본 연구 통찰은 exponent/fraction 조합 기반 FP8에도 적용 가능.  
- **양자화 훈련(quantized training)**: Int8 기반 학습·미세조정 시 outlier 처리가 핵심 변수로 작용하므로, mixed-precision 기법 확장 필요.  
- **더 대규모 모델**: 1,750억 이상에서 추가 emergent 현상이 발생할 수 있으므로, 동적 outlier 탐지 및 적응형 분해 전략 연구가 요구됨.  
- **Attention 양자화**: 파라미터 없는 attention 연산의 low-bit 처리 최적화는 남은 과제.  

이상의 기여는 대형 언어 모델의 접근성 확대와 저자원 환경에서 고성능 NLP 시스템 구축에 결정적 전환점을 제공할 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/c4437518-b531-4797-8b53-b6295e42193c/2208.07339v2.pdf
