# Hungry Hungry Hippos: Towards Language Modeling with State Space Models

**주요 주장 및 기여**  
– **SSM과 어텐션 간 성능 격차 해소**: 기존 State Space Model(SSM)은 언어 모델링에서 어텐션 대비 표현력과 하드웨어 효율성 면에서 뒤처졌으나, H3 계층과 FlashConv 연산으로 이를 크게 좁혔다.  
– **H3 계층 제안**: 두 개의 SSM(shift, diagonal)과 쿼리·키·값 간 곱셈 상호작용을 결합해 토큰 **기억(recall)** 및 **비교(comparison)** 능력을 갖추도록 설계.  
– **FlashConv 알고리즘**: 블록 기반 FFT 융합 연산과 상태 전달(state-passing)을 통해 시퀀스 길이 8K 이하에서는 2×, 그 이상에서는 5–6× 가속.  

***

## 1. 해결하고자 하는 문제  
1. SSM은 시퀀스 길이에 거의 선형적으로 확장되지만, 하드웨어 활용도가 낮아 어텐션보다 느리다.  
2. 언어 모델링에서 SSM이 어텐션 대비 (i) 과거 토큰 **기억**, (ii) 시퀀스 내 토큰 **비교** 능력을 충분히 갖추지 못해 퍼플렉시티가 크게 낮다.  

***

## 2. 제안 방법  
### 2.1 H3 계층  
– 입력 $$u\in\mathbb{R}^{N\times d}$$를 세 가지 투영으로 분리:  

$$
Q = uW_Q,\quad K = uW_K,\quad V = uW_V.
$$  

– **Shift SSM**: $$K$$를 discrete shift 행렬 $$A_{\text{shift}}$$로 순환 저장해, 과거 이벤트 이후 토큰을 로그로 보존.  
– **Diagonal SSM**: $$K\odot V$$의 외적을 대각 행렬 $$A_{\text{diag}}$$로 누적해, 전체 시퀀스에 걸친 비교 정보를 유지.  
– **출력**:  

```math
\mathrm{H3}(u) = \bigl(Q \,\odot\, \mathrm{SSM}_{\mathrm{diag}}\bigl(\mathrm{SSM}_{\mathrm{shift}}(K)\odot V\bigr)\bigr)W_O.
```

– 곱셈 상호작용($$\odot$$)이 인덕션(head)·연관기억(recall) 과제를 해결하는 핵심.  

### 2.2 FlashConv 연산  
1. **블록 FFT 융합**: cuFFT의 입출력 병목을 제거하고, 텐서코어 친화적 블록단위 행렬곱으로 FFTConv를 구현.  
2. **상태 전달(state-passing)**: 전체 시퀀스를 GPU SRAM 한계를 넘는 청크로 나누고, 각 청크 간에 SSM 상태를 전달해 무한 확장 가능.  

***

## 3. 모델 구조  
– **Pure H3 모델**: 12~32개 H3 계층 + MLP + LayerNorm.  
– **Hybrid H3-Attention**: 총 레이어 중 2개를 어텐션으로 유지(초기·중간 위치), 나머지 H3.  
– 파라미터 규모: 125M, 355M, 1.3B, 2.7B  

***

## 4. 성능 향상  
| 모델                        | OpenWebText PPL | Pile PPL  | SuperGLUE Zero-shot (%) |
|-----------------------------|----------------|-----------|------------------------|
| Transformer (125M)          | 20.6           | 19.0\*   | 50.1                   |
| Pure S4D / GSS              | 24.9 / 24.0    | –         | –                      |
| H3 (125M)                   | 21.0           | 8.8       | –                      |
| Hybrid H3-Attn (125M)       | **19.6**       | **8.8**   | **53.9**               |
| Hybrid H3-Attn (355M, 1.3B) | 15.9 / 12.4    | 7.1 / 6.0 | 54.7 / 56.5            |

– Hybrid H3-Attn(125M)은 Transformer(125M) 대비 PPL 1.0 포인트 개선, SuperGLUE 평균 3.8%↑.  
– **추론 속도**: H3-Attn(1.3B) 최대 2.4× 빠른 토큰 생성.  
– **LRA 벤치마크**: S4에 FlashConv 적용 시 Transformer 대비 5.8× 가속.  

***

## 5. 한계 및 일반화 성능 향상 가능성  
– **복잡한 언어 구조**: 코어 기계작업(Induction, Recall)은 해결했으나, 문법·추론·장기 의존성의 까다로운 패턴까지 입증 필요.  
– **하이퍼파라미터 민감도**: SSM 상태 차원(m)·청크 크기(N′)·투영 차원(d) 조정에 따른 안정성·성능 연구 과제.  
– **일반화**: SSM의 연속-순환 혼합 구조는 적은 데이터 상황에서 장기 의존 학습에 유리. 그러나 소규모 학습 데이터셋 및 도메인 편차에 대한 실험 확대가 필요.  

***

## 6. 향후 연구에 미치는 영향과 고려 사항  
– **하이브리드 아키텍처의 잠재력**: SSM과 어텐션의 보완성 강조, 다양한 배치(Interleaving) 전략 탐색 가능성.  
– **하드웨어 친화적 알고리즘**: FlashConv와 같은 메모리·연산 효율화 기법이 SSM 기반 대형 모델 상용화 촉진.  
– **확장성**: 수십억 파라미터급 H3-Attn 모델에 대한 자원 효율적 훈련·배포 전략 수립이 관건.  
– **응용 분야**: 시간-공간적 데이터(EEG, 오디오, fMRI) 모델링으로 확장 시, SSM 고유의 선형 복잡도 장점 극대화 기대.  

---  

이 논문은 SSM의 언어 모델링 한계를 체계적으로 분석하고, 아키텍처·알고리즘 차원에서 해결책을 제시하여, 향후 **표현력-효율성** 트레이드오프를 재정의할 중요한 이정표가 될 것이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/daa307c6-8ca1-402e-af3b-086f51fc0376/2212.14052v3.pdf)
