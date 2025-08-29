# Switch Transformers: Scaling to Trillion Parameter Models with Simple and Eﬃcient Sparsity 

## 1. 핵심 주장 및 주요 기여  
Switch Transformer는 **Mixture-of-Experts(MoE)** 기반의 희소 활성화(sparse activation) 모델로, 기존 밀집(dense) 트랜스포머 대비  
- **파라미터 수는 수백억~수조 개**까지 확장하면서  
- **FLOPs는 일정**하게 유지하여  
- **7배 이상의 학습 속도**를 달성하고,  
- **메모리·통신 비용과 훈련 불안정성** 문제를 대폭 경감  

한다. 주요 기여는 다음과 같다:[1]
- 단일(expert) 라우팅(k=1)으로 라우터·통신 복잡도 획기적 감소  
- bfloat16 환경에서도 안정적 훈련을 위한 국소적(float32) 정밀도 전환  
- 소규모 자원에서도 2개 이상의 expert만으로도 품질 개선  
- 사전학습(pre-training), 파인튜닝(fine-tuning), 다국어·다작업 학습에서 일관된 성능 향상  
- Sparse→Dense 지식증류로 90% 이상 압축 시 30% 이상의 품질 이득 보존

***

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제  
- **파라미터 확장**과 **연산 효율성** 간 상충  
- MoE 모델의 복잡한 라우팅, 통신 오버헤드, 훈련 불안정성  
- 대규모 모델의 메모리 한계  

### 2.2 제안 방법  
1. **Switch Routing (k=1)**  
   - 각 토큰을 확률 최대인 단일 expert에만 라우팅  
   - $$y = p_i(x)\,E_i(x)$$, $$\displaystyle p_i(x)=\frac{e^{W_r x}_i}{\sum_j e^{W_r x}_j}$$  
2. **부가적 로드밸런싱 손실**  
   - 균등 라우팅 유도를 위한 손실  

```math
     \mathcal{L}_{\text{aux}} = \alpha\,N \sum_{i=1}^N f_i\,P_i,\quad
     f_i=\frac1T\sum_{x\in B}\mathbb{1}\{\arg\max p(x)=i\},\quad
     P_i=\frac1T\sum_{x\in B}p_i(x)
```

3. **선택적 정밀도(Selective Precision)**  
   - 라우터 내부 연산만 float32, 나머지는 bfloat16 사용  
4. **파인튜닝 시 Expert Dropout**  
   - 일반 레이어 드롭아웃 0.1, expert 레이어 드롭아웃 0.4 적용

### 2.3 모델 구조  
- 기존 Transformer FFN 대신 **Switch-FFN** 삽입  
- 128~2048개 소규모 expert 병렬  
- 데이터·모델·expert 병렬 조합으로 TPU/분산 학습 최적화  

### 2.4 성능 향상  
| 모델             | 사전학습 속도↑ | Fine-tuning 성능↑            | 압축율 | 지식증류 보존↑ |
|------------------|---------------|------------------------------|--------|----------------|
| Switch-Base(64e) | 7.5×[1]       | SuperGLUE +4.4%p[1]          | 95%    | 30%[1]         |
| Switch-Large     | 2.5×[1]       | SQuAD  +0.5%p[1]             | 97%    | 30%[1]         |
| Switch-XXL       | 4× over T5-XXL[1] | ANLI +9.3%p[1]           | —      | —              |

### 2.5 한계  
- **대규모(수조 파라미터) 모델 훈련 불안정성** 여전히 잔존  
- FLOPs 증가 시 모델·데이터 병렬 통신 병목 우려  
- 일부 파인튜닝 과제에서 과적합·하이퍼파라미터 민감도  

***

## 3. 일반화 성능 향상 관점  
- **샘플 효율성**: 동일 연산량 대비 더 빠르게 수렴[1]
- **파인튜닝 전이**: 지식·추론 과제 모두에서 일관된 성능 향상  
- **다국어·다작업**: 101개 언어 전역에서 평균 5× 속도↑, 품질↑[1]
- **지식증류**: 압축된 밀집 모델에도 sparse 모델 이득 일부 전이 가능  

이로써 모델 확장 시 과적합 우려를 완화하고, 저자원 환경에서도 일반화 성능을 확보할 수 있다.

***

## 4. 향후 연구 영향 및 고려 사항  
- **훈련 안정화 기법 개발**: 대규모 sparse 모델의 수렴 안정성 보완  
- **전달학습(transfer learning) 이해**: upstream perplexity→downstream 성능 관계 정량 분석  
- **하드웨어·소프트웨어 co-design**: expert·model·data 병렬 최적화 자동화  
- **이기종 전문가(heterogeneous experts)**: 토큰 난이도 따라 계산량 적응적 분배  
- **다양한 모달리티 확장**: 비언어·멀티모달 sparse 모델 적용 가능성 탐색  

Switch Transformer는 **매우 큰 모델을 효율적·안정적으로 학습**시키는 새로운 패러다임을 제시하며, 앞으로 sparse 아키텍처 연구 전반에 큰 영향을 미칠 전망이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/3b2acc4b-8e99-48dc-a4a1-246b932ca2ab/2101.03961v3.pdf)
