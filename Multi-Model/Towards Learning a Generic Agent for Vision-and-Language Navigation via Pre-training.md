# Towards Learning a Generic Agent for Vision-and-Language Navigation via Pre-training

# 핵심 주장 및 주요 기여 요약

**핵심 주장**  
‘PREVALENT’이라 명명된 첫 번째 Vision-and-Language Navigation(VLN)용 사전학습(pre‐training)–미세조정(fine‐tuning) 패러다임을 제안하여, 대규모 이미지‐텍스트‐행동 삼중쌍(image-text-action triplets)의 자기지도 학습으로 얻은 **일반화 가능한 시각언어 표현**이 새로운 경로 탐색 과제에 효과적으로 전이(transfer)됨을 보인다.  

**주요 기여**  
1. 대규모 Matterport3D 시뮬레이터 기반 이미지-텍스트-행동 삼중쌍 658만 개를 활용한 VLN 첫 사전학습 프레임워크 설계  
2. Vision-language encoder의 **Image-attended Masked Language Modeling**과 **Action Prediction** 두 가지 자기지도 목표 제안  
3. PREVALENT를 기존 R2R, CVDN, HANNA VLN 과제에 드롭인(drop-in) 적용하여 모두에서 최첨단 성능 경신  
4. 사전학습이 보이지 않은 환경 및 새로운 과제로의 일반화 성능(generalization) 크게 향상  

***

# 문제 정의 및 제안 방법

## 해결하고자 하는 문제  
- VLN 과제는 자연어 지시문과 파노라마 이미지가 주어질 때 적절한 이동 행동 시퀀스를 생성하도록 학습하나,  
  -  지시문과 시각 입력의 **다양성**이 매우 크고  
  -  새로운 환경에서 학습 데이터가 **부족**하여 일반화 어려움  

## 모델 구조

1. 입력 임베딩  
   - **Visual Embedding**: 36-view 파노라마 각 이미지에 대해 ResNet-2048 특징 $$s_v$$와 위치 정보 sin/cos 벡터 $$s_p$$를 합쳐 $$2176$$차원→FC→LayerNorm→ $$d_h$$ -차원  
   - **Text Embedding**: Transformer 표준 토큰+위치 임베딩 후 LayerNorm  

2. 인코더  
   - **Single‐modal Encoders**  
     – 시각, 언어 각각 $$L_{\text{vision}}$$, $$L_{\text{text}}$$ 개의 Transformer 레이어  
   - **Cross‐modal Encoder**  
     – 양쪽 모달리티 간 크로스어텐션(Cross‐Attention) 레이어 $$L_{\text{cross}}$$개로 특징 융합  
   - 총 Transformer 레이어 수: $$L_{\text{text}}+L_{\text{vision}}+L_{\text{cross}}=9+1+3$$

3. 사전학습 목표  
   1) Image-attended Masked Language Modeling  

$$
      \mathcal{L}_{\mathrm{MLM}}
      = -\mathbb{E}_{s,x}\sum_{i\in\mathcal{M}}\log P\bigl(x_i\mid x_{\setminus \mathcal{M}},s\bigr)
      $$  
      
  ($$\mathcal{M}$$은 15\% 무작위 마스크 단어)  
   2) Action Prediction  

$$
      \mathcal{L}_{\mathrm{AP}}
      = -\mathbb{E}_{s,a,x}\log P\bigl(a\mid \texttt{[CLS]},s,x\bigr)
      $$  
   
   - 전체 손실: $$\mathcal{L}\_{\mathrm{pre}} = \mathcal{L}\_{\mathrm{MLM}} + \mathcal{L}_{\mathrm{AP}}$$

***

# 성능 및 한계

## 성능 향상  
- **Room-to-Room(R2R)**  
  – SPL 47→51 (SPL 단일 지시문)로 최고치 경신[1]
  – Seen/Unseen 간 성능 격차 감소  
- **Cooperative Vision-and-Dialogue Navigation(CVDN)**  
  – Goal Progress 기준 Random→2.10→3.15로 개선[1]
- **HANNA**  
  – Success Rate 35%→52.9%, SPL 35%→28.7% (Unseen)[1]

## 일반화 성능  
- 사전학습된 임베딩이 **초기 학습 단계**부터 빠르게 적응하며, 보이지 않은 환경에서도 더 높은 성능에 수렴[1]
- Ablation: $$\mathcal{L}_{\mathrm{AP}}$$ 포함 시 AP 제거 대비 모든 과제에서 일관된 향상[1]

## 한계  
1. **데이터 편중**: Matterport3D 기반 실내 환경에 국한  
2. **계산비용**: 대규모 사전학습에 GPU 8대×V100 사용  
3. **시각 입력 단순화**: 파노라마 이미지 전역 특징만 사용, 객체 수준(region-level) 특징 미반영  

***

# 일반화 성능 향상 관점 심층 분석

- **마스크 언어 모델링**에 시각 정보 결합으로 지시문-환경 정합성 강화  
- **행동 예측 손실**로 중간 행동 단계의 강화학습 보조 없이도 의사결정 정책 예비 학습  
- R2R 학습곡선에서 Seen/Unseen SPL 차이가 ENVDrop 대비 좁음[1]

***

# 향후 연구 및 고려 사항

1. **데이터 다양화**: 실내 외 다양한 환경(야외, 공공장소)으로 확장  
2. **객체 수준 특징 통합**: Faster R-CNN 등 region-level 시각피처 도입 검토  
3. **효율적 사전학습**: 모델 경량화, 지식 증류(distillation)로 실무 적용성 개선  
4. **멀티모달 강화학습**: 사전학습된 표현과 강화학습 결합해 무보상 학습 효율 향상  
5. **강인성 평가**: 노이즈, 조명 변화, 지시문 모호성 등 실제 환경 취약성 분석  

PREVALENT 프레임워크는 VLN뿐 아니라 향후 시각언어 제어, 로봇 내비게이션 등 폭넓은 멀티모달 순차 의사결정 분야 연구에 중요한 초석이 될 것이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/332659a6-1a05-4981-86f5-4a5f89c341f6/2002.10638v2.pdf)
