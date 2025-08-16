# Noise Contrastive Alignment of Language Models with Explicit Rewards

## 1. 핵심 주장과 주요 기여  
**핵심 주장:**  
기존의 Direct Preference Optimization(DPO)는 페어와이즈 선호 데이터만 다룰 수 있는 반면, 본 논문에서 제안하는 InfoNCA(Information Noise Contrastive Alignment)와 NCA(Noise Contrastive Alignment)는 명시적 스칼라 보상 데이터와 선호 데이터를 모두 활용하여 언어 모델을 직접 정렬(alignment)할 수 있다.  

**주요 기여:**  
- InfoNCA: 보상 데이터 $$(x \to \{y_i, r_i\}_{i=1}^K)$$를 멀티클래스 대조학습 문제로 전환하여, 응답별 보상으로부터 언어 모델 정책을 직접 추출  
- NCA: InfoNCA/DPO의 상대적 우위 최적화가 야기하는 ‘선호 응답의 우도 감소’ 현상을 해결하고, 절대적 우도 증가를 보장  
- DPO를 페어와이즈 선호 설정에서 InfoNCA의 특수 사례로 보임으로써, 페어와이즈 선호 정렬 이론과 대조학습(InfoNCE) 이론의 통합  

***

## 2. 문제 정의, 제안 방법 및 모델 구조

### 2.1 해결하고자 하는 문제  
- **제약:** DPO 등 기존 방법은 페어와이즈($x→{y_w ≻ y_l}$) 선호 라벨만 처리할 수 있어, 스칼라 보상 $$(x→\{y_i,r_i\})$$이 주어졌을 때 비효율적(서브샘플링·정보 손실)  
- **목표:** 다수의 응답에 부여된 연속 스칼라 보상을 모두 활용하여, 언어 모델 정책  
$$\displaystyle \pi^*(y|x)\propto \mu(y|x)e^{r(x,y)/\alpha}$$을 직접 학습  

### 2.2 제안 방법  
#### 2.2.1 InfoNCA (Multi-Class InfoNCE)  
K개의 후보 응답 중 보상이 큰 응답을 선택하는 멀티클래스 확률:  

$$
p(\nu=i\mid x,\{y_j\})=\frac{e^{r(x,y_i)/\alpha}}{\sum_{j=1}^K e^{r(x,y_j)/\alpha}}
$$  

모델 예측 확률은 보상 모델 $$r_\theta$$로부터:  

$$
p_\theta(\nu=i\mid x,\{y_j\})
=\frac{e^{r_\theta(x,y_i)}}{\sum_{j=1}^K e^{r_\theta(x,y_j)}}
$$  

Cross-entropy 형태의 InfoNCA 손실:  

```math
\mathcal{L}\_{\rm InfoNCA}
=-\sum_{i=1}^K\underbrace{\frac{e^{r_i/\alpha}}{\sum_{j}e^{r_j/\alpha}}}_{\text{soft label}}
\log\frac{e^{r_\theta(x,y_i)}}{\sum_{j}e^{r_\theta(x,y_j)}}
```

#### 2.2.2 NCA (Binary NCE)  
각 응답 $$y$$가 최적 정책 $$\pi^*$$ 혹은 사전학습 정책 $$\mu$$로부터 샘플되었는지를 이진 분류:  

$$
p(\nu=1\mid x,y)=\sigma\bigl(r(x,y)/\alpha -\log Z(x)\bigr)
\approx\sigma\bigl(r_\theta(x,y)\bigr)
$$  

절대 우도 증가를 유도하는 NCE 손실:  

```math
\mathcal{L}_{\rm NCA}
=-\sum_{i=1}^K w_i\,\log\sigma\bigl(r_\theta(x,y_i)\bigr)
-\tfrac1K\sum_{i=1}^K\log\sigma\bigl(-r_\theta(x,y_i)\bigr),
``` 

where $$w_i=e^{r_i/\alpha}/\sum_j e^{r_j/\alpha}$$.

### 2.3 모델 구조 및 구현  
- 언어 모델 정책 $$\pi_\theta(y|x)$$는 사전학습 모델 $$\mu(y|x)$$의 로짓 위에 학습된 보상 $$r_\theta(x,y)$$를 덧셈  
- 보상 파라미터화: $$r_\theta(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\mu(y|x)}$$  
- QLoRA 기법으로 효율적 파인튜닝  

***

## 3. 성능 향상 및 한계  

### 3.1 성능 향상  
- **스칼라 보상 활용:** UltraFeedback(4개 응답×GPT-4 보상)에서 InfoNCA/NCA가 DPO 기반 페어와이즈 정렬 대비 MT-bench 점수 +0.3–1.2 향상  
- **추가 서브옵티멀 응답 활용:** K(응답 수) 증가 시 MT-bench·AlpacaEval 모두 성능 상승  
- **추론·코딩 과제:** UltraInteract(pairwise)에서 NCA는 DPO 대비 복잡 수학·코딩 벤치마크에서 큰 폭 우위  
- **우도 감소 회피:** DPO/InfoNCA가 선호 응답 우도 감소 현상을 보이나, NCA는 절대 우도 증가 보장  

### 3.2 한계  
- **계산 비용:** K-응답 대비 멀티클래스 대조학습 비용 증가  
- **하이퍼파라미터 민감도:** α(보상 온도), β(파라미터화 계수) 조정 필요  
- **이론적 가정:** 충분한 샘플·무제한 모델 용량 가정 하에서 최적 보장  

***

## 4. 일반화 성능 향상 가능성  
- **절대 우도 보존:** NCA는 데이터 우도를 유지·증가시켜, 추론·수학 과제의 ‘결과 정답’ 패턴을 더 충실히 학습  
- **하이퍼파라미터 견고성:** NCA는 DPO 대비 정책 발산·KL 발산에 더욱 내성  
- **서브옵티멀 사례 활용:** 다양한 난이도 응답을 동시에 학습함으로써, 실제 분포 일반화 능력 제고  

***

## 5. 향후 연구에 미치는 영향 및 고려 사항  
- **통합 Alignment 프레임워크:** 선호·스칼라 보상 모두 처리 가능한 대조학습 기반 방법론 확장  
- **효율적 샘플링 전략:** K-응답 비용 절감을 위한 하드·소프트 네거티브 샘플링 연구  
- **하이퍼파라미터 자동 최적화:** α, β의 자동 조정 혹은 적응형 스케줄링  
- **대규모·실세계 적용:** 대용량 웹 규모 보상 데이터 및 인간 라벨링 병행 실험  

이 논문은 언어 모델 정렬 연구에서 선호 데이터와 명시적 스칼라 보상을 통합하는 새로운 관점을 제시하며, 특히 추론·수학·코딩 분야에서의 일반화 성능 개선 가능성을 열어두었다. 앞으로 계산 효율성과 안정성을 높이는 연구가 중요할 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/aa6f56c7-948a-4b79-804b-cef8efb29776/2402.05369v3.pdf
