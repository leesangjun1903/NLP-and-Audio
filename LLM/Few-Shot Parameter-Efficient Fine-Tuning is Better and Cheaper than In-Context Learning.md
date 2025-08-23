# Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning

## 1. 핵심 주장 및 주요 기여 (간결 요약)
“Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning” 논문은 소수의 학습 예시만으로도 모델 전체를 미세조정하지 않고 일부 파라미터(Activation-scaling 벡터)만 학습하는 **적은 파라미터 효율적 미세조정(Parameter-Efficient Fine-Tuning, PEFT)**이, 예시를 매번 모델 입력에 포함하는 **Few-Shot In-Context Learning(ICL)**보다 더 **높은 정확도**와 **현저히 낮은 계산 비용**을 동시에 달성할 수 있음을 보인다.

주요 기여:
- (IA)³: 키·밸류 어텐션 및 피드포워드 중간 활성화에 학습 가능한 벡터로 요소별 스케일링을 적용하는 새로운 PEFT 기법 제안
- T-Few 레시피: T0 모델 위에 (IA)³, Unlikelihood 손실, 길이 정규화 손실을 결합하여 단일 하이퍼파라미터 설정만으로 다양한 신규 과제에서 강력한 성능 달성
- RAFT 벤치마크에서 인간 성능 상회 및 기존 최첨단 대비 절대 6%p 우위 입증
- 계산·메모리·저장 비용 분석을 통해 ICL 대비 1,000배 이상 효율성 개선 제시

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 향상 및 한계

### 2.1 해결하고자 하는 문제
Few-shot 설정에서  
- ICL은 매 예측마다 모든 예시를 입력에 포함해 처리하므로 연산·메모리 비용이 선형으로 증가  
- 프롬프트 형식, 예시 순서 등 민감하여 성능 불안정  
- PEFT는 효율적이나, 극소량 데이터(few-shot)에서 충분한 성능을 보이는지 미흡  

이를 위해 “극소량 학습” 상황에서도 낮은 추가 파라미터만으로 안정적이고 강력한 일반화 성능을 확보하는 기법 개발이 목표.

### 2.2 제안하는 방법
#### (IA)³: Infused Adapter by Inhibiting and Amplifying Inner Activations
Transformer 블록의  
- 어텐션 키·밸류 행렬 출력  
- 위치별 Feed-Forward 네트워크 중간 활성화  
에 대해 크기 $$d_k, d_v, d_{ff}$$인 학습 벡터 $$\ell_k, \ell_v, \ell_{ff}$$를 도입하고 요소별 곱셈 수행:

$$
\text{Attention}(Q, K, V) = \text{softmax}\bigl(Q(\ell_k \odot K)^T / \sqrt{d_k}\bigr)\,(\ell_v \odot V)
$$

$$
\text{FFN}(x) = (\ell_{ff} \odot \gamma(W_1 x))W_2
$$

벡터 수는 레이어당 $$d_k + d_v + d_{ff}$$로 극소 파라미터만 추가. 초기값은 1로 설정해 학습 전 원 동작 유지.

#### T-Few 레시피
- 백본: T0 (11B 파라미터)
- PEFT: (IA)³ + 사전학습(멀티태스크 미리 학습)
- 손실 함수:
  - 크로스엔트로피 $$\mathcal{L}_{\mathrm{LM}}$$
  - Unlikelihood 손실:
  
```math
      \mathcal{L}_{\mathrm{UL}} = -\frac{1}{\sum_n T^{(n)}} \sum_{n=1}^N \sum_{t=1}^{T^{(n)}} \log\bigl(1 - p(\hat y^{(n)}_t \mid x, \hat y^{(n)}_{ < t})\bigr)
```
  
- 길이 정규화 손실:
  
```math
      \mathcal{L}_{\mathrm{LN}} = -\log\frac{\exp\bigl(\beta(x,y)\bigr)}{\exp\bigl(\beta(x,y)\bigr) + \sum_n \exp\bigl(\beta(x,\hat y^{(n)})\bigr)}, 
      \quad \beta(x,y)=\tfrac1T\sum_t\log p(y_t\mid x,y_{ < t})
```

- 하이퍼파라미터: 1,000 스텝, 배치 8, 학습률 $$3\times10^{-3}$$, Adafactor, 선형 감쇠+워밍업

### 2.3 성능 향상
- **Held-out T0 과제**: GPT-3 175B few-shot ICL 대비 +6.0%p 우위, FLOPs 1,000× 절감  
- **RAFT 벤치마크**: 인간 기준점(73.5%) 상회(75.8%), 다음 최선 기법 대비 +6%p  
- **계산 비용**: ICL 대비 예측당 3자릿수 FLOPs 절감, fine-tuning 30분·\$2

### 2.4 한계
- **분류 과제 중심**: 추후 요약·QA 등 생성 태스크로 확장 필요  
- **추가 PEFT 비교**: (IA)³ 외 신속 부상하는 P-Tuning v2 등과 직접 비교 미흡  
- **언어·도메인 일반화**: 비영어·특수 도메인 과제에서 검증 추가 필요

## 3. 일반화 성능 향상 관점
- (IA)³ 사전학습된 활성화 스케일링 벡터는 **멀티태스크 사전학습**을 통해 광범위 표현을 내재화  
- Unlikelihood와 길이 정규화 손실이 **잘못된 후보 확률 억제** 및 **길이 편향 완화**로 테스트이상 일반화 안정성 강화  
- 하이퍼파라미터 고정식으로 **과제별 튜닝 불필요**, 소규모 검증셋만으로도 **일관된 성능** 달성  

## 4. 향후 연구 영향 및 고려사항
- (IA)³의 활성화 스케일링 개념은 **다양한 아키텍처·모달리티**(비전·멀티모달)로 확장 가능  
- 생성(tasks) 및 구조화된 출력(task-specific head) 등으로 PEFT 범위 확대 연구  
- 도메인·언어 일반화를 위한 **사전학습 데이터 다양성**과 **프롬프트 설계** 상호작용 추가 분석 필요  
- PEFT와 ICL의 **혼합 패러다임**(하이브리드 인퍼런스) 개발로 효율-성 균형 최적화 기대

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/7a6f3f76-7a61-4c14-9730-b1cee2bc2b4c/2205.05638v2.pdf)
