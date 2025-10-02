# Balanced Multimodal Learning via On-the-fly Gradient Modulation

# 핵심 요약 및 기여

**핵심 주장:**  
“Balanced Multimodal Learning via On-the-fly Gradient Modulation” 논문은 **멀티모달 모델 학습에서 특정 모달리티가 최적화 과정을 지배하여 다른 모달리티의 표현 학습이 부족한** 최적화 불균형(opt imization imbalance) 문제를 지적하고, 이를 해소하기 위한 **On-the-fly Gradient Modulation with Generalization Enhancement**(OGM-GE) 전략을 제안한다.

**주요 기여:**  
- **최적화 불균형 현상 규명**: 멀티모달 모델이 단일 모달리티보다 성능이 높아도, 지배적인 모달리티가 학습 손실을 낮추며 다른 모달리티의 그래디언트 기여를 억제해 표현 학습이 미흡해짐을 이론·실험적으로 분석.  
- **OGM(Gradient Modulation)**: 각 모달리티의 손실 기여도를 실시간 모니터링하여 불균형 비율 $$\alpha_t$$을 계산하고,  

$$
    k^u_t = 
      \begin{cases}
        1 - \alpha_t^\beta, & \text{지배 모달리티}\\
        1, & \text{그 외}
      \end{cases}
  $$  
  
로 그래디언트를 조절함으로써 **저평가 모달리티**에 더 많은 최적화 기회를 부여.  
- **GE(Generalization Enhancement)**: OGM 적용 시 감소하는 SGD 노이즈 강도를 보완하기 위해, 각 스텝마다 **동적 분산** $$\Sigma_t$$를 가진 **가우시안 노이즈** $$\mathbf{h}_t\sim\mathcal{N}(0,\Sigma_t)$$를 추가하여 일반화 성능 유지·향상.  
- **플러그인 유연성 및 검증**: 단순 병합(fusion) 기법(Concatenation, Summation)뿐 아니라 FiLM, TSN-AV, TSM-AV, PSP, AGVA 등 다양한 멀티모달 구조에 적용 가능하며, CREMA-D, KS(Kinetics-Sounds), VGGSound, AVE 등 **4개 데이터셋**에서 일관된 성능 향상을 입증.

***

# 문제 정의 및 제안 방법

## 1. 해결하고자 하는 문제  
멀티모달 분류 모델은 서로 다른 모달리티(예: 오디오, 비주얼)를 **균일한 학습 목표**로 joint training하지만, 실제로는 데이터셋 편향(dataset preference) 등으로 특정 모달리티가 **지배(dominant)**하여 다른 모달리티의 파라미터가 충분히 갱신되지 않는 **최적화 불균형**이 발생한다.

## 2. OGM-GE 방법 상세

### 2.1. 기호 정의  
- $$x_i=(x_i^a, x_i^v)$$: 오디오, 비주얼 입력  
- $$\theta^a,\theta^v$$: 모달리티별 인코더 파라미터  
- $$\mathcal{L}=\frac1N\sum_i\ell(f(x_i),y_i)$$: 교차엔트로피 손실  

### 2.2. 기여도 불균형 측정  
각 모달리티 $$u\in\{a,v\}$$의 **불균형 비율** $$\alpha_t$$를,  

$$
  \alpha^a_t = \frac{\sum_{i\in B_t}s^a_i}{\sum_{i\in B_t}s^v_i},  
  \quad
  s^u_i=\mathrm{softmax}_c(W^u\phi^u(x_i^u)+b^u)\big|_{c=y_i}
$$  

로 정의해 “오디오 vs 비주얼”의 기여도 비율을 동적으로 계산.

### 2.3. On-the-fly Gradient Modulation (OGM)  
불균형 비율 $$\alpha_t$$에 따라 **그래디언트 스케일** 계수 $$k^u_t$$를:  

$$
  k^u_t =
    \begin{cases}
      1 - \alpha_t^\beta, & u\text{가 지배 모달리티인 경우}\\
      1, & \text{그 외}
    \end{cases}
$$  

로 설정해, 지배 모달리티일수록 그래디언트 크기를 축소함으로써, 저평가 모달리티의 최적화 기회를 증대.

### 2.4. Generalization Enhancement (GE)  
OGM 적용 시 SGD 노이즈 공분산이 $$k_t^2\Sigma_t$$로 줄어드는 문제를 해결하기 위해, 매 스텝마다 **동일 분산** $$\Sigma_t$$의 가우시안 노이즈 $$\mathbf{h}_t\sim\mathcal{N}(0,\Sigma_t)$$를 추가하여 **일반화 능력**을 보전·강화.

***

# 모델 구조 및 실험 결과

## 1. 모델 구성  
- **백본 인코더**: ResNet-18 기반 비주얼/오디오 인코더  
- **퓨전**: Concatenation(기본), Summation, FiLM, TSN-AV, TSM-AV, PSP, AGVA 등  
- **최적화**: SGD + OGM-GE (Adam에도 적용 가능)

## 2. 주요 성능 개선  
| 데이터셋      | 단일 모달 최고(Acc) | 기본 Multimodal | +OGM-GE  |
|-------------|-------------------|---------------|---------|
| CREMA-D     | 54.4 (Audio)      | 51.7          | **61.9** |
| Kinetics-Sounds | 59.8 (Audio)  | 59.8          | **63.1** |
| VGGSound    | 44.3 (Audio)      | 49.1          | **50.6** |
| AVE (로컬)  | –                 | 72.0          | **76.9** |

**OGM-GE** 적용 시 모든 데이터셋에서 **5–10%p** 이상의 일관된 성능 향상 확인.

## 3. 한계  
- 멀티모달 모델 내 단일 모달 성능이 여전히 **단독 최적** uni-modal 모델에는 미치지 못함.  
- 최적화 관점만으로는 불균형 완전 해결 어려워, **퓨전 전략**·**네트워크 구조** 개선 병행 필요.

***

# 일반화 성능 및 향후 연구 시 고려사항

**일반화 강화 효과:**  
- OGM 적용 시 SGD 노이즈 강도 감소 → GE를 통해 원래 노이즈 강도 복원·증가  
- 실험적으로 batch size·learning rate 변화에 따른 일반화 성능 민

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/91932cc3-aec9-4292-ab07-efb16b862544/2203.15332v1.pdf)
