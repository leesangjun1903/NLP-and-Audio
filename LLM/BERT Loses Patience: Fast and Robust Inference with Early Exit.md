# BERT Loses Patience: Fast and Robust Inference with Early Exit

**주요 주장 및 기여**  
“BERT Loses Patience” 논문은 **Patience-based Early Exit (PABEE)**라는 간단하면서도 강력한 추론 기법을 제안한다. 각 Transformer 레이어 뒤에 내부 분류기를 장착하고, 연속된 내부 예측이 t회 연속으로 동일할 때 추론을 조기 종료함으로써,  
- 추론 속도를 크게 높이고,  
- 과잉 추론(overthinking)을 방지하여 정확도와 적대적 강건성을 동시에 향상시킨다.[1]

# 1. 해결하고자 하는 문제  
기존의 대규모 사전학습 언어모델(PLM)은 수십 개의 레이어와 수억~수십억 파라미터를 가진다.  
- **고비용·저속도**: 메모리와 레이턴시 측면에서 비효율적이다.  
- **과잉 추론(Overthinking)**: 얕은 레이어에서 충분한 표현을 얻었음에도 불구하고, 마지막 레이어까지 지나치게 계산을 계속하여 잘못된 예측을 내릴 수 있으며, 이로 인해 일반화 성능이 저하되고 적대적 공격에 취약해진다.[1]

# 2. 제안 방법  
## 2.1 PABEE 메커니즘  
- 각 레이어 $$L_i$$ 뒤에 내부 분류기 $$C_i$$를 추가  
- 입력 $$x$$가 레이어를 통과할 때마다  
  1. 은닉 상태 $$h_i = L_i(h_{i-1})$$ 계산  
  2. 예측 $$y_i = C_i(h_i)$$ 생성  
  3. 연속 예측 일치 횟수 $$\mathrm{cnt}_i $$ 업데이트

$$
       \mathrm{cnt}_i = 
       \begin{cases}
         \mathrm{cnt}_{i-1} + 1, & \underset{j}{\arg\max}\,y_i = \underset{j}{\arg\max}\,y_{i-1}\\
         0, & \text{otherwise}
       \end{cases}
     $$

- $$\mathrm{cnt}_j \ge t$$ 이면 레이어 $$j$$에서 추론 종료  
- 종료 조건을 충족하지 않으면 마지막 분류기 $$C_n$$ 사용  

## 2.2 손실 함수  
내부 분류기를 함께 학습하여, 각 분기(branch)에 대한 예측을 보장  
- **분류**: 교차 엔트로피  

$$
    \mathcal{L}_i = -\sum_{z\in Z} \mathbf{1}(z=\bar{y}) \log P(y_i=z\mid h_i)
  $$

- **회귀**: 평균 제곱 오차  

$$
    \mathcal{L}_i = (y_i - \bar{y})^2
  $$

- 최종 손실:  

$$
    \mathcal{L} = \frac{1}{n}\sum_{i=1}^n \lambda_i \mathcal{L}_i
  $$  
  
  ($$\lambda_i$$: 레이어별 가중치)[1]

## 2.3 이론적 분석  
이진 분류에서 내부 분류기의 오차율이 $$q$$, 최종 분류기의 오차율이 $$p$$, 분기 수가 $$n$$, 인내도(patience)가 $$t$$일 때,  

$$
  n - t + 1 \ge 2q^t \,p - q - p
$$  

조건에서 만족하면 정확도가 향상됨을 보였다.[1]

# 3. 모델 구조  
- Backbone: BERT, ALBERT 등 Transformer 계열 PLM  
- 각 레이어 뒤에 **경량 선형 분류기** 추가  
- 인내도 $$t$$ 하이퍼파라미터로 속도-정확도 트레이드오프 조절  

# 4. 성능 향상 및 한계  
## 4.1 GLUE 벤치마크  
- ALBERT-base에 PABEE 적용 시, 평균 1.57배 속도 향상과 함께 정확도 0.7–2.3%p 개선.[1]
- BERT-base에도 1.62배 속도 향상 및 다양한 지식 증류(knowledge distillation) 기법을 능가하는 정확도 달성.[1]

## 4.2 적대적 강건성  
- TextFooler 공격에서 원본 ALBERT 대비 더 많은 쿼리가 필요하고, 공격 후 정확도 저하가 크게 감소하여 강건성 향상.[1]

## 4.3 일반화 성능  
- NLP 외에 **이미지 분류**(ResNet-56, CIFAR-10/100)에 적용하여 속도 1.22–1.26배, 정확도 0.2–0.5%p 향상, 내부 분기 기법의 일반화 가능성 확인.[1]

## 4.4 한계  
- 단일 브랜치 구조(ResNet, Transformer)에 최적화되어 있으며, NASNet 같은 다중 분기 구조에는 추가 설계 필요.

# 5. 향후 영향 및 고려 사항  
- **모바일·엣지 컴퓨팅**에서 PLM 활용 촉진: 에너지 효율, 레이턴시 절감에 기여할 전망.  
- **다른 동적 추론 기법과의 결합**: 분포 기반(exit on confidence) 방법과의 하이브리드 적용 가능성 탐색.  
- **윤리적·편향 이슈 검증**: 기존 PLM에 스크립팅된 편향에 PABEE가 미치는 영향 분석 필요.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b1613fcc-0d17-4ae4-a663-9abcf784fa0e/2006.04152v3.pdf)
