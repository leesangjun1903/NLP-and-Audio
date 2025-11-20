# SnapFusion: Text-to-Image Diffusion Model on Mobile Devices within Two Seconds

**핵심 주장:**  
SnapFusion은 텍스트-투-이미지 확산(diffusion) 모델을 모바일 기기에서 2초 이내에 실행할 수 있도록 최초로 구현한 모델이다. 이 모델은 기존 Stable Diffusion v1.5 대비 8단계(steps)로 획기적으로 줄인 디노이징 과정과 경량화된 네트워크 구조, 그리고 향상된 스텝-증류(step distillation) 기법을 결합하여, iPhone 14 Pro에서 1.84초 만에 512×512 이미지를 생성하면서 FID 및 CLIP 점수에서 오히려 더 우수한 성능을 달성한다.[1]

**주요 기여:**  
1. **효율적 UNet 설계:** 사전 학습된 Stable Diffusion UNet의 구조적 중복성을 분석하고, 교차-어텐션 및 ResNet 블록을 단계별로 제거·추가하는 *아키텍처 진화(evolving-training) 프레임워크*를 도입하여, 파라미터는 860M→848M, 레이턴시는 1.7s→0.23s로 대폭 단축.  
2. **이미지 디코더 압축:** VAE 디코더에 50% 채널 프루닝과 합성 데이터 기반 디스틸레이션을 적용하여, 디코더 파라미터를 14배 축소하고 실행 속도를 3.2배 향상.  
3. **CFG-인식 스텝 증류:** 기존 바닐라 증류 손실에 *분류기-프리 가이던스(classifier-free guidance)*를 결합한 손실 함수를 제안하여, CLIP 점수를 크게 향상시키는 동시에 FID 저하를 최소화.  
4. **훈련 전략 최적화:** 직접 증류(direct distillation)를 채택하고, 교차-프루닝 및 CFG 확률·범위 하이퍼파라미터를 조정하여 FID vs. CLIP trade-off를 효과적으로 제어.  
5. **모바일 배포 가능성:** iPhone 14 Pro에서 1.84초, iPhone 13 Pro Max에서 2.67초, iPhone 12 Pro Max에서 4.40초로 텍스트-투-이미지 생성이 가능함을 실험으로 입증.[1]

# 세부 설명

## 1. 해결하고자 하는 문제  
- **높은 계산 비용과 느린 속도:** 텍스트-투-이미지 확산 모델은 복잡한 네트워크와 수십 단계의 디노이징 반복 때문에 고성능 GPU 또는 클라우드를 필수로 하며, 이는 비용 및 개인정보 보호 문제를 야기.  
- **기존 경량화 한계:** 퀀타이제이션·GPU 최적화 기반 가속화는 일부 속도 향상을 제공하나 2초 이내 실시간 생성 목표에는 미치지 못하고, 품질 손실이 발생할 수 있음.

## 2. 제안하는 방법  
### 2.1 Efficient UNet via Architecture Evolving  
- **강건 훈련(Robust Training):** 교차-어텐션 및 ResNet 블록을 확률적으로 스킵하면서 훈련하여, 블록 제거 시 성능 저하를 완화한다(수식 5).  
- **아키텍처 진화(Architecture Evolving):**  
  - 각 블록의 기여도를 CLIP 점수와 레이턴시(lookup table) 기반으로 평가.  
  - 가치 점수 $$V = \frac{\Delta \text{CLIP}}{\Delta \text{Latency}}$$ 기준 블록 추가·제거를 반복.[1]
- **결과:** UNet 파라미터 860M→848M, 50단계→8단계 디노이징 시 1.7s→0.23s 단축.

### 2.2 Efficient Image Decoder  
- VAE 디코더에 50% 균등 채널 프루닝 후, 원본 SD-v1.5 디코더와 출력 차이를 MSE로 최소화하는 합성 데이터 기반 디스틸레이션 수행.[1]
- 디코더 파라미터 14배 축소, 레이턴시 369ms→116ms 달성.

### 2.3 CFG-Aware Step Distillation  
- **바닐라 증류 손실:**  

$$
    L_{\text{vanilla}} = \bigl\|\tilde x_t^s - \tilde x_t^T\bigr\|^2
  $$  

- **CFG-인식 손실:** 교사와 학생 예측에 classifier-free guidance를 적용($$w$$-scale), 이후 손실 산출.[1]
- **최종 손실:**  

$$
    L = \lambda L_{\text{denoise}} + 
    \begin{cases}
      L_{\text{CFG}} & \text{with prob }p,\\
      L_{\text{vanilla}} & \text{otherwise}.
    \end{cases}
  $$  
  
  하이퍼파라미터 $$p$$와 CFG 범위 $$[2, 14]$$로 FID/CLIP 트레이드오프 제어.

## 3. 모델 구조  
- **텍스트 인코더:** ViT-H, 77 토큰 → 임베딩.  
- **디노이징 UNet:** 64×64 해상도 입력, 다운·미드·업 스테이지별 교차-어텐션 + ResNet 블록.  
- **VAE 디코더:** 8×8→512×512 재구성.  
- 전체 구조는 Stable Diffusion v1.5를 토대로, UNet·디코더를 경량화.

## 4. 성능 향상  
- **속도:** iPhone14 Pro에서 총 1.84s (UNet:230ms, 디코더:116ms) vs. SD-v1.5 85s.[1]
- **화질:**  
  - MS-COCO 2014(30K) Zero-shot FID/CLIP: 8단계 SnapFusion이 SD-v1.5(50단계) 대비 FID 동등 수준, CLIP +0.004–0.010 우위.  
  - 주요 비교: SnapFusion 8단계 FID 24.2 vs. Meng et al. 8단계 FID 26.9.[1]
- **증류 기법 효과:** CFG-aware 증류 적용 시 CLIP 점수 +0.002–0.007 향상.

## 5. 한계  
- 여전히 수백만~억 단위 파라미터 보유, 모바일 기기별 최적화 추가 연구 필요.  
- iPhone 14 Pro 중심 분석, 기타 기기 호환성·레이턴시 검증 부족.

# 일반화 성능 향상 가능성  
SnapFusion의 **아키텍처 진화** 및 **CFG-aware 증류** 기법은 특정 하드웨어 제약을 벗어나 일반적인 경량 확산 모델 학습에도 적용될 수 있다.  
- 블록별 기여도 평가·제거 방식은 다른 비전·언어-비전 모델에도 확장 가능.  
- CFG 인식을 통한 증류 손실은 다양한 조건부 생성 모델에서 품질·다양성 조절에 활용될 수 있어, 도메인 적응 및 제너레이티브 튜닝에 기여할 전망.

# 향후 연구 영향 및 고려 사항  
- **엣지 디바이스 배포:** 추가 하드웨어 아키텍처(ARM, 다양한 SoC) 대응, 양자화·하드웨어 가속과 결합 연구.  
- **모델 크기 축소:** 네트워크 경량화·지식 증류를 결합한 더 강력한 하이브리드 압축 기법 개발.  
- **보안·윤리:** 모바일 생성 콘텐츠의 오남용 방지를 위해 자동 검열·위변조 탐지 모듈 통합.  
- **확장성:** 영상·3D·비디오·다중모달 생성 모델로의 적용 및 실시간 상호작용 연구.  

SnapFusion은 **모바일 실시간 텍스트-투-이미지 생성**의 새로운 가능성을 열었으며, 경량 확산 모델과 증류 기법의 결합 방향을 제시함으로써 차세대 엣지 AI 애플리케이션 연구에 중요한 기준점이 될 것이다.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e801fb62-c690-40ec-848f-26fc29f5f970/2306.00980v3.pdf)
