# Non-Autoregressive Neural Machine Translation

## 1. 핵심 주장 및 주요 기여  
**핵심 주장:**  
비순차적(non-autoregressive)으로 전체 번역문을 병렬 생성함으로써 추론(inference) 속도를 기존 Transformer 대비 최대 15배까지 가속하면서도 BLEU 성능 손실을 최소화할 수 있다.

**주요 기여:**  
- 입력 단어별 ‘생산량(fertility)’을 잠재 변수로 도입하여 다중 모달리티(multimodality) 문제 해소  
- 지식 증류(sequence-level knowledge distillation)와 강화학습 기반 미세조정(fine-tuning) 결합  
- **NAT** 모델 설계 및 **Noisy Parallel Decoding** 알고리즘 제안으로 속도·성능 균형 달성  

## 2. 문제 정의 및 제안 기법

### 2.1 해결하고자 하는 문제  
전통적 NMT 모델은 토큰을 순차적으로 생성(autoregressive)하므로 병목 발생. Transformer의 병렬 처리 장점을 추론 시에는 활용 불가능.

### 2.2 제안 방법  
#### 2.2.1 비순차적 출력 분포  
$$ p_{NA}(Y|X;\theta) = p_L(T|X;\theta)\prod_{t=1}^T p(y_t|X;\theta) $$  
- 모든 출력 토큰을 독립적으로 병렬 생성  
- 길이 분포 $$p_L$$는 잠재 변수인 fertilities로 대체  

#### 2.2.2 fertilities 잠재 변수 도입  
- 각 입력 토큰 $$x_{t'}$$에 정수 fertilities $$f_{t'}$$ 부여  
- 전체 출력 길이 $$T=\sum_{t'} f_{t'}$$  
- 모델화:  

$$ p_{NA}(Y|X;\theta)=\sum_{f\in F}\Bigl[\prod_{t'}p_F(f_{t'}|X;\theta)\cdot\prod_{t}p(y_t|X,f;\theta)\Bigr] $$  

- fertilities 예측: 인코더 최종 층 위 softmax (클래스 수 $$L=50$$)  

#### 2.2.3 디코딩 알고리즘  
- **Argmax**: 각 $$f_{t'}=\arg\max p_F$$  
- **Average**: $$f_{t'}=\mathrm{Round}\bigl(\sum p_F\cdot f\bigr)$$  
- **Noisy Parallel Decoding (NPD)**: 샘플링 후 교사(autoregressive) 모델로 평가  

### 2.3 모델 구조  
- **인코더:** 기존 Transformer와 동일  
- **디코더:**  
  - autoregressive causal mask 제거  
  - self-attention에서 자기 자신(attend-to-self)만 마스킹  
  - 위치 정보 강화 positional attention 추가  
  - 디코더 입력으로 복사된 소스 임베딩(ferility 기반)  

## 3. 성능 향상 및 한계

| 모델                             | BLEU (En→De WMT14) | 추론 속도       |
|----------------------------------|--------------------:|---------------:|
| Autoregressive (beam=4)          |              23.45  | 607 ms (1×)   |
| NAT (argmax)                     |              17.35  | 39 ms (15.6×) |
| NAT + Fine-tuning + NPD (s=100)  |              19.17  | 257 ms (2.36×)|

- 지식 증류로 BLEU +5점, fertilities 도입으로 +2–4점, fine-tuning +1.5점 개선  
- NPD (s=100)로 gap 4 → 2.5점까지 축소  
- **한계:** 다중 모달리티 완전 해소 불가, 장문서 번역 시 반복 오류(repetition) 여전히 발생, NPD 병렬 자원 필요  

## 4. 일반화 성능 향상 가능성  
- **Fertility**는 입력 문장 구조 전반에 대한 글로벌 플랜을 제공하여 다양한 길이·어순에 대응 가능  
- **Positional attention** 모듈로 국지적 재배열(local reordering) 능력 강화  
- **Fine-tuning** 시 REINFORCE 기반 K-L 손실로 출력 분포 첨예화(peaking) 유도  
- 소규모 데이터셋(IWSLT)에서도 distillation+fertility를 결합하면 overfitting 억제 및 성능 안정화 관찰  

## 5. 향후 연구에의 영향 및 고려 사항  
- **영향:**  
  - 실시간 번역, 모바일·엣지 디바이스 MT에 병렬 NMT 적용 가능성 확대  
  - 잠재 변수 기반 다중 모달리티 모델링 연구 활성화  
- **고려점:**  
  - fertilities 외에 재배열(reordering) latent 변수 통합 연구  
  - NPD 연산 자원 절감 및 더 효율적 샘플링 기법 개발  
  - 대형·다언어 코퍼스에서 일반화·확장성 검증 및 개선

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/2ea8702c-81de-44b1-ba52-f7fad0ae280d/1711.02281v2.pdf)
