# EAT: Self-Supervised Pre-Training with Efficient Audio Transformer

**핵심 주장 및 주요 기여**  
Efficient Audio Transformer(EAT)는 오디오 자기지도 학습(Self-Supervised Learning)에서  
-  높은 마스크 비율(약 80%)의 **Inverse Block Multi-Mask** 전략과  
-  전역(발화) 수준과 지역(프레임) 수준을 결합한 **Utterance-Frame Objective (UFO)**  
를 도입함으로써, 오디오 표현 학습의 효율성과 성능을 동시에 크게 향상시킨다.  
주요 기여는 다음과 같다.  
- **UFO 손실함수**: 발화 전체 특성을 예측하는 utterance-level MSE와, 패치 단위 프레임 복원을 위한 frame-level MSE를 조합해 학습 효율과 표현력 동시 확보  
- **Inverse Block Multi-Mask**: 2D 블록 단위로 마스킹하여 시간·주파수 영역 간 상관관계 유지 및 학습 난이도 상승, 복수 마스크 병렬 처리로 데이터 활용 극대화  
- **비대칭 구조**: 복잡한 Transformer 인코더 + 경량 CNN 디코더로 복원 부담 최소화, 10 에폭만으로 기존 SOTA 대비 10배 이상 빠른 학습  

***

## 1. 해결 문제 및 한계
기존 오디오 SSL 모델들은 마스크 기반 재구성(MAE), 부트스트랩(BYOL-A) 등으로 표현 학습 성과를 개선했으나  
- **높은 계산 비용**: 대규모 Transformer 인코더·디코더 반복 사용  
- **마스크 전략 단순**: 1D 랜덤 마스킹으로 시간·주파수 정보 단절  
- **전역 정보 미반영**: 프레임 단위 복원만 집중해 발화 전체 맥락 학습 미흡  
이로 인해 대량 데이터 셋(Audioset-2M) 사전학습 시 높은 자원 소모와 긴 학습 시간이 발생했다.

***

## 2. 제안 방법

### 2.1 모델 구성  
1) **CNN 패치 임베딩**: 멜스펙트로그램을 (S×S) 컨볼루션으로 non-overlap 패치화  
2) **Transformer 기반 인코더**: 12-layer ViT-B 구조  
3) **경량 CNN 디코더**: 6-layer 2D CNN + LayerNorm, GELU  

### 2.2 Utterance-Frame Objective (UFO)  
$$
L_{UFO} = L_f + \lambda L_u
$$  
- Frame-level Loss $$L_f = \|X_o - Y_o\|_2^2$$ : 마스크된 패치 예측  
- Utterance-level Loss $$L_u = \|c' - \bar{Y}_a\|_2^2$$ : CLS 토큰 $$c'$$로 타겟 평균 $$\bar{Y}_a$$ 회귀  
- $$\lambda$$ = 1으로 설정 시 최적 성능  

### 2.3 Inverse Block Multi-Mask  
- 원본 패치 $$X_p\in\mathbb{R}^{T'\times F'\times E}$$를 2D 블록 단위로 마스킹  
- 블록 크기 $$5\times5$$에서 최고 성능(40.2%mAP)  
- 다중 마스크 클론(batch×16) 병렬 입력으로 데이터 활용 극대화  

***

## 3. 성능 및 일반화 향상

### 3.1 주요 성능 지표  
| 데이터셋         | EAT (mAP/Acc) | 기존 최고 Self-SSL (Audio-MAE) |
|------------------|---------------|---------------------------------|
| AudioSet-2M      | 48.6%         | 47.3%                           |
| AudioSet-20K     | 40.2%         | 37.1%                           |
| ESC-50           | 95.9%         | 94.1%                           |
| SPC-2            | 98.3%         | 98.3%                           |

- **학습 속도**: 10 에폭·58시간 (4×3090)으로 Audio-MAE(32에폭·36시간, 64×V100)에 비해 약 10배 빠름  
- **Ablation**: $$\lambda=1$$과 CLS 토큰 활용을 통해 전역 표현 학습이 1.3%p 향상, 블록 크기 확대로 지역 정보 도전 과제 강화  

### 3.2 일반화 성능 향상 요소  
- **Global-Local 이중 손실**: 전역 발화 표현 학습으로 다양한 음향 상황에 대한 견고한 특징 획득  
- **블록 마스킹**: 연속된 시간·주파수 영역 보존으로 음향 이벤트 간 관계 파악 능력 강화  
- **다중 마스크 병렬화**: 단일 클립을 여러 관점에서 학습시켜 노이즈나 변형에 대한 강건성 확보  

***

## 4. 향후 영향 및 고려사항
본 논문은 오디오 SSL의 효율성과 성능 양립 가능성을 제시하며,  
- **대규모 모델 확장**: 더 깊은 Transformer 또는 다양한 마스크 스케일링 연구  
- **오디오-음성 통합 학습**: 스피치와 일반 음향을 결합한 멀티태스크 확장  
- **비지도 파인튜닝**: 소량 레이블링 환경에서도 UFO 기반 전역-지역 손실 활용 방안  

추후 연구에서는 **마스크 형태 다양화**, **UFO 손실 가중치 자동 최적화**, **실시간 추론 경량화** 등을 검토하면 모델의 **범용성**과 **효율성**을 더욱 강화할 수 있을 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/8b82866b-0506-4bfa-a3cd-82c85e414b0f/2401.03497v1.pdf
