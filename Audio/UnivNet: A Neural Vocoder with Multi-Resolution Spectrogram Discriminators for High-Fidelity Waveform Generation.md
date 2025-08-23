# UnivNet: A Neural Vocoder with Multi-Resolution Spectrogram Discriminators for High-Fidelity Waveform Generation

## 1. 핵심 주장 및 주요 기여  
**핵심 주장**  
UnivNet은 다중 해상도 스펙트로그램 판별자(MRSD)를 도입하여, 풀밴드 멜-스펙트로그램 조건 하에서 발생하는 **과도한 스무딩(over-smoothing) 현상**을 완화하고, 실시간으로 고해상·고충실도의 음파형을 생성할 수 있음을 보인다.  

**주요 기여**  
1. 다중 해상도 스펙트로그램 판별자(MRSD) 설계: 서로 다른 STFT 파라미터(FFT 점수, 프레임 이동, 윈도우 길이)를 적용한 여러 선형 스펙트로그램을 동시에 판별자 입력으로 사용함으로써 고주파 대역의 세부 디테일을 유지.  
2. 멀티-피리어드 웨이브폼 판별자(MPWD)와 결합: 주기성 기반의 여러 주기로 신호를 재구성하여 시간 영역의 디테일도 강화.  
3. 효율적인 생성기 구조: 위치-가변 컨볼루션(LVC)과 게이트 활성화 유닛(GAU)을 활용하여 적은 파라미터로 높은 음질과 빠른 생성 속도 달성.  
4. 대규모 다중 화자 데이터셋(LibriTTS)에서 학습 및 평가하여, **학습 화자(seen)**와 **미학습 화자(unseen)** 모두에서 경쟁 모델 대비 우수한 객관·주관 평가 결과 확보.  

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계  

### 2.1 해결하고자 하는 문제  
- 기존 GAN 기반 풀밴드 멜-스펙트로그램 vocoder들은 고주파 정보 부족으로 인해 스펙트로그램이 과도하게 스무딩되어 음질이 저하되는 문제가 발생.[1]

### 2.2 제안 방법  
- **다중 해상도 스펙트로그램 판별자(MRSD)**  
  - 서로 다른 STFT 파라미터 $$(\text{FFT points}, \text{hop}, \text{win})$$로 계산한 $$M$$개의 선형 스펙트로그램 $$\{s_m, \hat s_m\}_{m=1}^M$$을 입력으로 사용.  
  - 스펙트로그램 판별 손실:  

$$ L_{\text{aux}} = \frac{1}{M} \sum_{m=1}^M \mathbb{E}\big[L_{\text{sc}}(s_m,\hat s_m) + L_{\text{mag}}(s_m,\hat s_m)\big], $$  
    
where $$L_{\text{sc}} = \frac{\|s-\hat s\|\_F}{\|s\|\_F}$$, $$L_{\text{mag}} = \frac{1}{S}\|\log s - \log\hat s\|_1$$[1].  

- **멀티-피리어드 웨이브폼 판별자(MPWD)**  
  - 소수(prime) 주기로 신호를 재구성하여 시간적 패턴을 포착.  
- **최종 손실**  
  - 생성기:  

$$ L_G = \lambda L_{\text{aux}} + \frac{1}{K}\sum_{k=1}^K \mathbb{E}_{z,c}[(D_k(G(z,c))-1)^2], $$  
  
  - 판별자:  
    
$$ L_D = \frac{1}{K}\sum_{k=1}^K \big(\mathbb{E}\_x[(D_k(x)-1)^2] + \mathbb{E}_{z,c}[D_k(G(z,c))^2]\big). $$  

### 2.3 모델 구조  
- **생성기(Generator)**  
  - 입력: 노이즈 $$z$$ (차원 64) + 풀밴드(0–12 kHz) 100-밴드 멜-스펙트로그램.  
  - 핵심 모듈: 위치-가변 컨볼루션(LVC) 레이어, 게이트 활성화 유닛(GAU), 잔차 스택.  
  - 채널 크기 $$c_G$$를 16/32로 설정한 두 버전(UnivNet-c16, c32).  
- **판별자(Discriminators)**  
  - MRSD: 3개 STFT 파라미터 세트 $$(1024,120,600), (2048,240,1200), (512,50,240)$$.[1]
  - MPWD: prime-period 재구성.  
  - 다중 판별자 출력 평균화.  

### 2.4 성능 향상  
- **객관 지표**  
  - PESQ, RMSE에서 UnivNet-c32가 최고 성능 달성(Seen: PESQ=3.70, RMSE=0.316; Unseen: PESQ=3.70, RMSE=0.294).  
- **주관 평가**  
  - MOS: UnivNet-c16(Seen=3.92±0.08, Unseen=3.79±0.09), UnivNet-c32(Seen=3.93±0.09, Unseen=3.90±0.09)로 HiFi-GAN 대비 동등하거나 우수.  
- **속도 및 파라미터**  
  - UnivNet-c16: 4.00M 파라미터, ×227 실시간 속도. UnivNet-c32: 14.86M 파라미터, ×204 속도.  

### 2.5 한계  
- MRSD 및 MPWD의 추가 연산으로 인한 학습 복잡도 증가.  
- LVC 및 GAU의 최적 배치 탐색이 수작업이며, 다른 언어·도메인으로의 확장성 검증 미흡.  
- 실제 환경 잡음 조건 및 저자원(모바일) 디바이스 성능 평가는 미제공.  

## 3. 일반화 성능 향상 가능성  
UnivNet은 풀밴드 스펙트로그램의 고주파 대역 정보를 보존하는 MRSD와 다양한 주기 정보를 포착하는 MPWD를 결합함으로써, **미학습 화자**에서도 음질 저하 없이 높은 MOS를 유지한다는 점에서 **강력한 일반화 능력**을 시연했다. 특히, 다중 해상도 스펙트로그램 특징이 배경 잡음 변화나 음색 차이에 유연하게 대응할 수 있으므로, 언어·화이트박스 환경 변화에도 적응할 가능성이 높다.

## 4. 향후 연구에의 영향 및 고려사항  
- **영향**:  
  - 풀밴드 조건에서 GAN 기반 vocoder의 과도한 스무딩 문제 해결을 위한 새로운 방향 제시.  
  - MRSD 개념을 타 음성·다중 모달 생성 모델에 확장 적용 가능성.  
- **고려사항**:  
  - MRSD 및 MPWD 구조의 경량화 및 자동화된 하이퍼파라미터 탐색 연구.  
  - 실제 노이즈 환경, 저전력 디바이스 배포 시 성능 평가 및 최적화.  
  - 언어별 음향 특성, 음색 다양성에 대한 일반화 실험 및 데이터 효율성 연구.

***

References  
 Won Jang et al., “UnivNet: A Neural Vocoder with Multi-Resolution Spectrogram Discriminators for High-Fidelity Waveform Generation,” arXiv:2106.07889v1.[1]
 Table 2 performance 결과.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/a42a827f-ffaa-48d7-b65b-2878c76acda2/2106.07889v1.pdf)
