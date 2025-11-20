# InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation

**핵심 주장 및 주요 기여**  
InstaFlow는 Stable Diffusion 기반의 대규모 텍스트-투-이미지 확산 모델을 **단일 단계**(one step)로 고품질 이미지를 생성할 수 있도록 혁신적으로 개선한 기법이다. 기존에는 20~100단계 이상의 반복적 샘플링이 필요했으나, InstaFlow는 *Rectified Flow*의 *reflow* 절차를 도입하여 노이즈와 이미지 간의 궤적을 직선화함으로써 학생 모델에 대한 증류(distillation)를 용이하게 하고, MS COCO 2017-5k에서 FID를 23.3으로 기록하며 종전 최저치(37.2)를 크게 경신하였다.[1]

***

## 1. 문제 정의  
대규모 확산 모델(예: Stable Diffusion)은 텍스트-투-이미지 생성에서 탁월한 성능을 보이지만, 매 반복 단계마다의 연산 비용으로 인해 **추론 속도가 매우 느리다**. 특히 10단계 이하의 저단계 샘플링 구간에서는 FID 성능이 급격히 저하되어, 실시간 응용이나 대규모 서비스에 적용하기 어렵다.

***

## 2. 제안 방법  
### 2.1 Rectified Flow 및 Reflow  
Rectified Flow는 분포 간 최적 수송을 ODE 흐름으로 모델링하며,  

$$
\frac{dZ_t}{dt} = v(Z_t, t),\quad Z_0\sim\mathcal{N}(0,I),\;Z_1\sim p_{\text{data}},
$$  

를 학습한다. Reflow 단계는 이 흐름의 **궤적을 직선화**하여 1단계 Euler 법으로도 정확한 근사를 가능하게 한다. 텍스트 조건을 반영한 reflow 목표는  

$$
v_{k+1} = \arg\min_v \mathbb{E}_{X_0,T}\int_0^1 \bigl\|X_1 - X_0 - v(X_t, t\mid T)\bigr\|^2 dt,
$$  

여기서 $$X_1$$은 이전 ODE 흐름의 종점이다. Reflow는 분포의 말단 분포를 유지하면서 궤적의 곡률을 줄이고, 노이즈-이미지 매칭을 개선한다.[1]

### 2.2 텍스트-조건 증류(Text-Conditioned Distillation)  
Reflow를 통해 얻은 거의 직선궤적 ODE를 교사 모델로 사용하여, 학생 모델이 단일 Euler 단계로 매핑을 학습하도록 한다:  

$$
\min_{\tilde v}\mathbb{E}_{X_0,T}\;D\bigl(\mathrm{ODE}_{v_k}(X_0,T)\,,\,X_0 + \tilde v(X_0,T)\bigr),
$$  

여기서 $$D$$는 LPIPS 유사도 손실이며, 순수 지도학습만으로 고품질 단일 단계 모델을 학습한다.[1]

***

## 3. 모델 구조  
- **텍스트 인코더**: CLIP ViT-L/14, 파인튜닝 없이 고정  
- **잠재 공간 생성기**: Stable Diffusion U-Net 기반  
- **스태킹 U-Net**: 파라미터 공유 없이 두 개를 연속 배치하여 용량을 1.7B 파라미터로 확장  
- **디코더**: 사전학습된 Autoencoder, 다운샘플 8배  

Reflow 단계에서는 기존 U-Net을 그대로 사용하며, 증류 단계에서는 스태킹 U-Net을 일부 블록 제거하여 연산 비용을 최적화하였다.  

***

## 4. 성능 향상  
- MS COCO 2017-5k:  
  - 단일 단계 FID 23.3 → 기존 Progressive Distillation 37.2 대비 **37.2% 향상**  
  - 대안 모델인 StyleGAN-T(13.9@0.1s)와도 대등한 성능  
- MS COCO 2014-30k:  
  - 단일 단계 FID 13.1 → StyleGAN-T(13.9) 대비 **0.8pt 우위**  
- **추론 속도**: 단일 단계 0.09s(on A100)  
- **학습 자원**: 199 A100 GPU 일로 비용 효율적  

***

## 5. 한계 및 일반화 성능  
InstaFlow는 직선화된 ODE로 인해 **일반화 성능**이 향상될 가능성이 높다. reflow가 노이즈-이미지 간 결합을 개선함으로써, 학생 모델이 다양한 텍스트 프롬프트에 대해 더욱 안정적인 매핑을 학습한다. 다만 다음과 같은 한계가 존재한다:  
- **복잡 구성 실패**: 다중 객체 상호작용(prompt composition)에서 여전히 오류 발생 가능(예: 손잡는 동물)  
- **데이터 편향**: LAION-5B 기반 학습으로 특정 도메인에 편향될 수 있음  
- **단일 단계 제약**: 세밀한 디테일 회복에는 후처리(Refiner)가 필요  

***

## 6. 향후 연구 영향 및 고려 사항  
InstaFlow는 **초고속 T2I 모델**의 새로운 기준을 제시하며, 다음 연구에 영감을 줄 것이다:  
- **3D 및 동영상 생성**: 확산 모델의 빠른 시뮬레이션을 활용한 실시간 3D/영상 생성  
- **도메인 적응**: 의료·과학·산업용 특화 도메인에 대한 reflow 기반 증류  
- **공정한 모델링**: 초고속 생성의 부작용(허위정보·딥페이크) 방지를 위한 윤리적·기술적 제어 기법  
- **하이브리드 파이프라인**: 단일 단계 예비 생성 후, 세밀한 후처리 모델 통합  

향후 연구 시에는 **reflow 단계 수**, **더 큰 데이터/모델 스케일**, **다중 조건 학습**을 통해 한계를 극복하고 폭넓은 일반화를 달성하는 데 주목할 필요가 있다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/516c7845-a68b-43c7-936e-e1fa2ef67e49/2309.06380v2.pdf)
