# OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models

**핵심 주장 및 주요 기여**  
OmniQuant는 기존의 후처리 정량화(Post-Training Quantization, PTQ)가 극저비트(예: W4A4, W2A16)에서 성능이 크게 떨어지는 한계를 극복하고, 정량화 인지 학습(Quantization-Aware Training, QAT) 수준의 정확도를 유지하면서도 PTQ와 유사한 소량의 데이터(128개 샘플)·짧은 시간(1–16시간) 내에 대형 언어 모델(LLM)을 변환할 수 있음을 보였다.  
1. **범방향성 미세 조정**: 학습 가능한 클리핑 임계치(Learnable Weight Clipping, LWC)와 동치 변환(Learnable Equivalent Transformation, LET)을 도입해, 가중치 및 활성화 양쪽 모두를 최적화 가능한 파라미터로 설계.  
2. **블록 단위 오류 최소화**: Transformer의 각 블록을 순차적으로 정량화하며, 오차를 블록 단위로 미분 가능하게 최소화하는 프레임워크를 제안.  
3. **다양한 비트 폭 적용**: W6A6, W4A4, W3A16, W2A16 등 극저비트 설정에서 기존 PTQ 기법 대비 최대 11.8% 제로샷 정확도 향상 및 평균 퍼플렉서티 대폭 감소를 달성.  

***

## 1. 해결하고자 하는 문제  
- **자원 제약**: GPT-3급 모델들은 수백 GB 메모리와 수십 시간의 연산을 요하지만, 실환경 배포 시에는 메모리·연산 모두 크게 제한됨.  
- **극저비트 정량화 성능 저하**: 기존 PTQ 기법들은 W4A4 이하 비트 폭에서 퍼플렉서티가 수천 배 증가하며 성능이 크게 떨어짐.  
- **QAT의 고비용**: QAT는 정확도를 유지하지만 100k개 이상의 데이터, 수백 GPU시간을 요구해 실사용에 부적합.  

***

## 2. 제안하는 방법  
### 2.1 블록 단위 오류 최소화 프레임워크  
Transformer 블록 $$F(W, X) $$를 정량화된 함수  

$$
F\bigl(Q_w(W;\Theta_1),\,Q_a(X;\Theta_2)\bigr)
$$  

로 근사하며,  

$$
\min_{\Theta_1, \Theta_2} \bigl\| F(W,X) - F(Q_w(W;\Theta_1),Q_a(X;\Theta_2)) \bigr\|^2
$$  

을 블록 단위로 최적화.  

### 2.2 Learnable Weight Clipping (LWC)  
가중치 분포의 상·하위 극단값을 학습 가능한 스케일 $$\gamma,\beta\in$$로 조절:[1]

$$
h = \frac{\gamma\,\max(W) - \beta\,\min(W)}{2^N - 1},\quad
z = -\bigl\lfloor\tfrac{\beta\,\min(W)}{h}\bigr\rceil
$$  

$$
W_q = \mathrm{clamp}\bigl(\lfloor W / h\rceil + z,\,0,\,2^N -1\bigr)
$$  

여기서 $$\lfloor \cdot \rceil$$는 반올림, $$N$$은 비트 수.  

### 2.3 Learnable Equivalent Transformation (LET)  
활성화의 채널별 이상치(outlier) 완화를 위해 스케일 $$s$$와 시프트 $$\delta$$를 학습:  

$$
Y = (X-\delta)\oslash s\;\bigl(s\odot W\bigr) + (\delta\,W + B)
$$  

또한 어텐션의 Q/K에 채널별 스케일 $$s_a$$를 학습 적용:  

$$
P = \mathrm{Softmax}\bigl((Q\oslash s_a)(s_a\odot K^T)\bigr)
$$  

이후 $$W$$와 변환된 $$X$$를 균일 정량화.  

***

## 3. 모델 구조 및 학습  
- **모델 범위**: OPT, LLaMA-1/2, Falcon, 챗용 LLaMA-2-chat (125M–180B)  
- **데이터**: WikiText2에서 랜덤으로 추출한 128×2048-토큰 샘플  
- **옵티마이저**: AdamW, LR(LWC)=5e-3, LR(LET)=1e-2, 배치 1, 20–40 epochs  
- **절차**: 각 블록별로 LWC/LET 파라미터 초기화 → 미분 가능 손실 최적화 → 최종 가중치에 파라미터 융합 → 균일 정량화  

***

## 4. 성능 향상 및 한계  
### 4.1 성능 개선  
- **극저비트(4비트) 정량화**: LLaMA-7B W4A4에서 제로샷 정확도 +4.99%∼+11.8%, 퍼플렉서티 38.41→52.65 (개선)  
- **일반화**: instruction-tuned LLaMA-2-chat Vicuna-벤치 GPT-4 평가에서 AWQ 대비 최대 80% 승률  
- **배포 효율**: A100-80G에서 W4A16g128, W2A16g128 양쪽 모두 메모리 약 2× 절감, 속도 2× 향상  

### 4.2 한계  
- **INT2/3 지원 미흡**: 현재 MLC-LLM 배포 시 2/3비트 처리 속도 저하  
- **SoftMax 양자화**: 4비트 이하 양자화시 성능 급락, 추가 연구 필요  
- **대규모 모델 시간**: GPTQ 대비 5× 느린 변환 시간  

***

## 5. 일반화 성능 향상 관점  
LET의 채널별 동치 변환은 모델 내 활성화 분포를 **다양한 도메인** 샘플에 걸쳐 균일화하여, 소량의 캘리브레이션 데이터에도 불구하고 다른 데이터셋(PTB, C4)에서도 일관된 성능 유지·향상을 확인했다. 이는 모델이 **도메인 편향 없이** 일반화할 수 있는 잠재력을 높인다.  

***

## 6. 향후 연구 방향 및 고려사항  
- **하드웨어 최적화**: INT2/3 가속을 위한 저비트 커널 및 아키텍처 연구  
- **비균일 양자화**: 4비트 미만 SoftMax 양자화 문제 해결을 위한 로그 스케일링 등 고급 기법 통합  
- **소량 데이터 일반화**: 128샘플 이상 활용 시 과적합 방지와 도메인 확장성 연구  
- **비영어·멀티모달 적용**: 다양한 언어 및 비전-언어 모델에서 LWC/LET의 일반화 가능성 평가  

---

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/1d3930d6-76a1-4b2e-8422-c0ef7cf0378e/2308.13137v3.pdf
