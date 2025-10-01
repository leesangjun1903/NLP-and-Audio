# UniVL: A Unified Video and Language Pre-Training Model for Multimodal Understanding and Generation

**핵심 주장 및 주요 기여**  
UniVL은 비디오와 언어 정보를 동시에 이해하고 생성할 수 있도록 설계된 **통합 프리트레이닝 모델**이다. 기존 모델들이 주로 이해(understanding) 과제에만 초점을 맞추어 프리트레이닝–파인튜닝 불일치(pretrain–finetune discrepancy)를 겪는 반면, UniVL은 인코더–디코더 구조에 기반한 **멍석(masked) 기반의 다섯 가지 학습 목표**와 **두 가지 프리트레이닝 전략**을 도입하여 이해 및 생성(generation) 과제 모두에서 최적화된다.[1]

# 1. 해결하고자 하는 문제  
기존 비디오–언어 프리트레이닝 연구는  
  * 비디오와 텍스트 간 상호작용을 모델링했으나,  
  * 주로 이해 과제에만 초점  
  * 생성 과제(예: 자막 생성)에서는 성능 저하와 불일치 현상 발생  
이를 해결하기 위해 UniVL은 **이해와 생성 모두를 통합**해 학습함으로써 downstream 과제에서 일관된 성능 향상을 달성한다.[1]

# 2. 제안 방법

## 2.1 모델 구조  
UniVL은 네 개의 주요 컴포넌트로 구성된다(그림 3 참조):[1]
  -  **Text Encoder**: BERT-Base를 활용해 토큰 시퀀스를 임베딩  
  -  **Video Encoder**: S3D로 추출한 3D 특징에 Transformer 인코더 적용  
  -  **Cross Encoder**: 텍스트와 비디오 인코딩을 결합한 뒤 Transformer로 상호 모달 상호작용(fusion)  
  -  **Decoder**: Transformer 기반 디코더로 언어 재구성 및 생성  

## 2.2 수식 및 학습 목표  
총 다섯 가지 손실 함수로 전체 모듈을 동시 최적화한다:[1]
1. **Video–Text Joint (L_Joint)**  
   MIL-NCE 기반 양성/음성 쌍 대비 학습:  

$$
     L_{\text{Joint}} = -\mathbb{E}_{t,v\in B}\log\frac{\exp(\langle v,t\rangle)}{Z}
   $$  

2. **Conditioned Masked Language Model (L_{CMLM})**  
   마스킹된 토큰 복원:  

$$
     L_{\text{CMLM}} = -\mathbb{E}\sum_{m}\log P(t_m\mid \tilde{t},v)
   $$  

3. **Conditioned Masked Frame Model (L_{CMFM})**  
   마스킹된 프레임 식별:  

$$
     L_{\text{CMFM}} = -\mathbb{E}\sum\log\frac{\exp(f(\tilde{v}_m,v))}{Z}
   $$  

4. **Video–Text Alignment (L_{Align})**  
   CLS 토큰에 대한 이진 분류 NCE 손실:  

$$
     L_{\text{Align}} = -\mathbb{E}\log\frac{\exp(s(t,v))}{Z}
   $$  

5. **Language Reconstruction (L_{Decoder})**  
   오토리그레시브 생성 학습:  

$$
     L_{\text{Decoder}} = -\mathbb{E}\sum_{i}\log P(t_i\mid t_{ < i},t_{\setminus m},v)
   $$  

최종 손실:  

$$
  L_{\text{UniVL}} = L_{\text{Joint}} + L_{\text{CMLM}} + L_{\text{CMFM}} + L_{\text{Align}} + L_{\text{Decoder}}
$$

## 2.3 프리트레이닝 전략  
* **StagedP (Stage-by-Stage)**:  
  - 1단계: 텍스트·비디오 인코더만 Video–Text Joint로 학습  
  - 2단계: 전체 모듈을 모든 목표로 학습 → 안정적 수렴 및 속도 향상  
* **EnhancedV**:  
  - 배치 내 15%의 텍스트를 전부 마스킹  
  - 비디오만으로 텍스트 생성하도록 강제 → 비디오 표현 강화  

# 3. 성능 향상 및 한계

## 3.1 성능 개선  
UniVL은 다섯 가지 downstream 과제에서 SOTA 달성:  
  -  **텍스트 기반 비디오 검색**: YouCook2 R@1 28.9%, MSR-VTT R@1 21.2%[1]
  -  **멀티모달 비디오 캡셔닝**: BLEU-4 17.35, CIDEr 1.81[1]
  -  **액션 분할**: COIN FAcc 70.02% (기존 대비 +9%↑)[1]
  -  **액션 단계 국소화**: CrossTask Avg. Recall 42.0% (비지도 학습 기준)[1]
  -  **멀티모달 감정 분석**: CMU-MOSI Corr 0.767 (오디오 제외 모델 중 최고)[1]

## 3.2 한계  
  -  **고정된 비디오 특징** 사용으로 end-to-end 학습 불가 → 추후 raw video 학습 필요  
  -  **대규모 연산·메모리** 요구 → 실시간 응용 제약  
  -  **과도한 Joint 강조** 시 생성 성능 저하 관찰[1]

# 4. 일반화 성능 향상 관점  
EnhancedV 전략을 통해 **비디오만으로 언어를 재구성**하도록 학습함으로써,  
  * 텍스트가 없는 영역에서도 의미론적 정보를 포착  
  * 다양한 downstream 도메인으로의 전이(transfer) 성능 강화  
실험 결과, EnhancedV 적용 시 R@1이 평균 +3% 상승, BLEU-4 +0.3점 개선됨을 확인.[1]

# 5. 향후 연구 영향 및 고려 사항  
**영향**  
  -  **통합 이해·생성** 프레임워크 제시 → 멀티모달 AI 연구 방향 확장  
  -  **대규모 교육 비디오 활용** 사례 증명 → HowTo100M 유형 데이터 중요성 부각  

**고려 사항**  
  1. **End-to-End 학습**: raw 영상 입력부터 fine-tuning 가능한 경량화 아키텍처  
  2. **효율화**: 모델 경량화 및 분산 학습 최적화로 실시간 적용  
  3. **다양한 모달리티**: 음성·오디오·센서 데이터 통합 확장  
  4. **윤리·편향**: 대규모 instructional 영상 데이터의 편향성 검토 및 보정  

이상으로 UniVL의 핵심 주장, 방법론, 성능 및 한계를 요약하였으며, 해당 연구가 멀티모달 AI 분야에서 차세대 통합 모델 개발에 기여할 전망과 후속 연구 시 고려해야 할 주요 사항을 제시하였다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/043f55f5-39e9-453c-bae4-6276870a687f/2002.06353v3.pdf)
