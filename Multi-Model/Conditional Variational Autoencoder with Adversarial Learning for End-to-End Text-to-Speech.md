# Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech

## 1. 핵심 주장 및 주요 기여
이 논문은 **텍스트 입력에서 중간 스펙트로그램 없이 바로 파형을 생성**하는 엔드투엔드 TTS 시스템 VITS(Variational Inference with adversarial learning for Text-to-Speech)를 제안한다.  
주요 기여는 다음과 같다.  
- **조건부 VAE**로 텍스트와 잠재변수를 연결해 단일 단계 학습 실현  
- **정규화 흐름(normalizing flow)** 으로 사전 분포의 표현력 강화  
- **적대적 학습(GAN)** 과 **특징 매칭 손실(feature-matching loss)** 으로 음질 개선  
- **확률적 지속시간 예측기(stochastic duration predictor)** 도입으로 다양한 억양·리듬 표현  

## 2. 문제 정의, 제안 방법, 모델 구조, 성능, 한계

### 2.1 해결하고자 하는 문제
기존 TTS는 (1) 텍스트→스펙트로그램, (2) 스펙트로그램→파형으로 두 단계로 나누어 학습·추론하며,  
- 단계 간 순차적 학습·파인튜닝이 필요해 비효율  
- 중간 특징이 고정되므로 표현력 한계  
- 병렬 생성 모델은 리듬·억양 다양성 부족  

### 2.2 제안 방법
VITS는 조건부 VAE 기반 엔드투엔드 구조에 GAN을 결합하고, 확률적 지속시간 예측기를 도입한다.

– **변분 하한(evidence lower bound, ELBO)** 최적화  

$$\log p_\theta(x|c) \ge \mathbb{E}\_{q_\phi(z|x)}[\log p_\theta(x|z)-\log \tfrac{q_\phi(z|x)}{p_\theta(z|c)}] $$

– **재구성 손실**  

$$ L_\mathrm{recon} = \|x_\mathrm{mel}-\hat{x}_\mathrm{mel}\|_1 $$

– **KL 발산**  

$$L_\mathrm{KL} = \mathbb{E}\_{q_\phi(z|x_\mathrm{lin})}\big[\log q_\phi(z|x_\mathrm{lin})-\log p_\theta(z|c_\mathrm{text},A)\big] $$

– **정규화 흐름**  

$$ p_\theta(z|c) = N(f_\theta(z);\mu_\theta(c),\sigma_\theta(c))\Bigl|\det\tfrac{\partial f_\theta(z)}{\partial z}\Bigr| $$

– **확률적 지속시간 예측기**  
변분 탈양자화(variational dequantization)와 데이터 증강으로 음절 지속시간 분포를 모델링  

$$L_\mathrm{dur}=-\mathbb{E}\_{q_\phi(u,\nu|d,c)}[\log p_\theta(d-u,\nu|c)-\log q_\phi(u,\nu|d,c)] $$

– **적대적 학습 및 특징 매칭**  

$$L_\mathrm{adv}(G)=\mathbb{E}\_z[(D(G(z))-1)^2],\quad L_\mathrm{fm}(G)=\mathbb{E}\_{y,z}\sum_{l}\|D_l(y)-D_l(G(z))\|_1 $$

– **최종 손실**  

$$ L = L_\mathrm{recon}+L_\mathrm{KL}+L_\mathrm{dur}+L_\mathrm{adv}(G)+L_\mathrm{fm}(G) $$

### 2.3 모델 구조
1. **Posterior Encoder**: 비인과적 WaveNet 블록으로 선형 스펙트로그램→잠재변수  
2. **Prior Encoder**: 트랜스포머 기반 텍스트 인코더 + 정규화 흐름  
3. **Stochastic Duration Predictor**: DDSConv 블록 + 신경 스플라인 흐름(flow)  
4. **Decoder**: HiFi-GAN 기반 생성기  
5. **Discriminator**: HiFi-GAN의 다중 주기(MPD) 디스크리미네이터  

### 2.4 성능 향상
- **자연스러움(MOS)**: LJ Speech에서 4.43, VCTK에서 4.38로 두 단계 모델 및 타 엔드투엔드보다 우수  
- **샘플 다양성**: 지속시간·F0 분포가 자율 회귀 모델 수준으로 다양  
- **속도**: Glow-TTS+HiFi-GAN 대비 약 2배 빠른 실시간 합성 속도  

### 2.5 한계
- **텍스트 전처리 의존**: 여전히 IPA 변환 등 전처리 필요  
- **약간의 음질 격차**: CMOS에서 원음 대비 소폭 선호도 차이  
- **복잡한 학습**: VAE+GAN+MAS+흐름 모두 학습 난이도 상승  

## 3. 일반화 성능 향상 가능성
- **다중 화자 확장**: VCTK 실험에서 강력한 화자 적응력  
- **화자 분리 표현 학습**: 잠재변수 분리→음성 변환(voice conversion) 활용 가능  
- **자기지도 언어 표현**: 텍스트 전처리 제거 위해 BERT·WavLM 같은 자기지도 학습 접목 시, 더욱 광범위한 도메인 일반화 가능  
- **경량화·저전력화**: 흐름·적대학습 경량화 구조 연구로 모바일·임베디드 적용 확대  

## 4. 향후 영향 및 고려사항
이번 연구는 **엔드투엔드 TTS**와 **확률모델+적대학습** 결합의 새로운 지평을 열었다.  
앞으로 고려할 점은  
- 텍스트 전처리 단계 제거를 위한 **언어·음성 자기지도 모델** 통합  
- **학습 안정성**과 **모델 경량화**를 위한 흐름 구조·GAN 손실 개선  
- **다양한 언어·악센트**로 확장해 데이터 편향성 해소  
- 실시간 애플리케이션을 위한 **추론 최적화**  

이러한 방향이 충실히 연구될 때, VITS의 접근법은 전통적 두 단계 TTS를 완전히 대체하며 폭넓은 응용을 이끌 것이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/f78563e6-6281-45bc-bb1d-d597dca42ae3/2106.06103v1.pdf)
