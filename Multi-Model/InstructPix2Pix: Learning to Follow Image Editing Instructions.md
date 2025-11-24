# InstructPix2Pix: Learning to Follow Image Editing Instructions

### 1. 핵심 주장과 주요 기여

**InstructPix2Pix**는 인간이 작성한 자연언어 명령어를 따라 이미지를 편집하는 새로운 패러다임을 제시합니다. 이 논문의 핵심 기여는 다음과 같습니다:[1]

- **합성 데이터 기반 학습**: GPT-3와 Stable Diffusion 두 개의 사전학습 모델을 결합하여 대규모 이미지 편집 데이터셋(454,445개 예제)을 자동으로 생성합니다.[1]

- **순전파 편집**: 별도의 인버전(inversion)이나 미세조정 없이 단일 순전파 과정에서 이미지를 편집하므로 초 단위의 빠른 처리가 가능합니다.[1]

- **제로샷 일반화**: 합성 데이터로만 학습했음에도 불구하고 실제 이미지와 사람이 작성한 자연스러운 명령어에 대해 우수한 일반화 성능을 달성합니다.[1]

***

### 2. 해결하는 문제와 제안 방법

#### 2.1 핵심 문제

기존 이미지 편집 방법들은 다음과 같은 제한을 갖고 있습니다:[1]

- 단일 편집 작업(예: 스타일 변환)에만 특화되어 있음
- 사용자가 입출력 이미지의 완전한 설명을 제공해야 함
- 인버전 또는 미세조정이 필수적으로 필요함

#### 2.2 제안 방법

**2단계 접근법**:[1]

**1단계: 다중모달 학습 데이터 생성**

먼저 GPT-3를 미세조정하여 이미지 캡션이 주어질 때 편집 명령과 수정된 캡션을 생성합니다:

$$\text{Input Caption} \rightarrow \text{Edit Instruction + Modified Caption}$$

예: "말을 탄 여자" → "드래곤을 타게 해" → "드래곤을 탄 여자"[1]

이미지 쌍 생성을 위해 Stable Diffusion과 **Prompt-to-Prompt** 기법을 결합합니다. Prompt-to-Prompt는 유사한 캡션으로 생성된 이미지들이 일관성을 유지하도록 교차 주의(cross-attention) 가중치를 공유하는 방식입니다.[1]

가변 유사도를 처리하기 위해 각 캡션 쌍당 100개의 이미지 쌍을 생성하고, **CLIP 기반 방향 유사도** 필터링을 적용합니다. 필터링 기준은 다음과 같습니다:[1]

- 이미지 간 CLIP 유사도: 0.75 이상
- 이미지-캡션 CLIP 유사도: 0.2 이상
- 방향 CLIP 유사도: 0.2 이상

**2단계: 조건부 확산 모델 학습**

잠재 확산 모델의 목적 함수는 다음과 같습니다:[1]

$$\mathcal{L} = \mathbb{E}_{E(x), E(c_I), c_T, \epsilon \sim \mathcal{N}(0,1), t} \left\| \epsilon - \epsilon_\theta(z_t, t, E(c_I), c_T) \right\|_2^2$$

여기서:
- $z_t = E(x) + \sqrt{\alpha_t}\epsilon$: 인코딩된 잠재 표현
- $E(c_I)$: 입력 이미지 조건
- $c_T$: 텍스트 편집 명령 조건
- $\epsilon_\theta$: 잡음 예측 네트워크

#### 2.3 이중 조건 분류기 없는 지도(Dual Classifier-Free Guidance)

InstructPix2Pix의 핵심 혁신은 두 개의 서로 다른 조건에 대해 분류기 없는 지도를 적용하는 것입니다:[1]

$$\tilde{e}_\theta(z_t, c_I, c_T) = e_\theta(z_t, \emptyset, \emptyset) + s_I \cdot (e_\theta(z_t, c_I, \emptyset) - e_\theta(z_t, \emptyset, \emptyset))$$
$$+ s_T \cdot (e_\theta(z_t, c_I, c_T) - e_\theta(z_t, c_I, \emptyset))$$

이는 다음을 의미합니다:[1]

- $s_I$: 입력 이미지와의 일관성 정도 제어 (공간 구조 보존)
- $s_T$: 편집 명령 준수 정도 제어 (편집 강도)

수식의 수학적 유도는 다음과 같습니다. 조건부 확률을 베이즈 정리로 표현하면:[1]

$$P(z|c_T, c_I) = \frac{P(z, c_T, c_I)}{P(c_T, c_I)} = \frac{P(c_T|c_I, z)P(c_I|z)P(z)}{P(c_T, c_I)}$$

로그를 취하고 미분하면:

$$\nabla_z \log P(z|c_T, c_I) = \nabla_z \log P(z) + \nabla_z \log P(c_I|z) + \nabla_z \log P(c_T|c_I, z)$$

이것이 식 (3)의 세 항에 대응됩니다.[1]

***

### 3. 모델 구조

#### 3.1 아키텍처 설계

- **기초 모델**: Stable Diffusion v1.5의 사전학습 가중치로 초기화[1]
- **입력 채널 추가**: 첫 번째 합성곱 계층에 이미지 조건 채널 추가, 새로운 채널 가중치는 0으로 초기화[1]
- **텍스트 조건 재사용**: 원래 캡션용 텍스트 조건 메커니즘을 편집 명령으로 재활용[1]

#### 3.2 학습 설정

- **해상도**: 256×256 (추론 시 512×512로 일반화)[1]
- **배치 크기**: 1,024 (8×A100 GPU)[1]
- **학습 단계**: 10,000 스텝, 약 25.5시간[1]
- **학습률**: 10⁻⁴[1]
- **조건 드롭 확률**: 각 조건에 대해 5% (이중 지도 학습)[1]

***

### 4. 성능 향상 및 한계

#### 4.1 성능 특징

**정성적 결과**:[1]

논문은 다음을 포함하여 다양한 편집을 성공적으로 수행합니다:
- 물체 교체 (예: "해바라기를 장미로 바꿔")
- 계절/날씨 변경 (예: "눈이 내리도록 해")
- 배경 변경 (예: "파리로 옮겨")
- 재료/질감 수정 (예: "가죽 재킷으로 만들어")
- 예술 매체 변환 (예: "모디글리아니 그림으로")

**정량적 평가**:[1]

CLIP 기반 메트릭으로 SDEdit과 비교:
- **CLIP 이미지 유사도**: 입력 이미지와의 일관성 측정
- **방향 CLIP 유사도**: 캡션 변화가 이미지 변화와 일치하는 정도

InstructPix2Pix는 동일한 방향 유사도에서 훨씬 높은 이미지 일관성을 달성합니다.[1]

#### 4.2 절제 연구(Ablation Study)

**데이터셋 크기의 영향**:[1]

- **10% 데이터**: 미세한 스타일 조정에만 효과적, 큰 변화 어려움
- **전체 데이터**: 객체 교체 등 큰 규모 편집 가능

**CLIP 필터링의 중요성**:[1]

필터링 제거 시 입력 이미지 일관성이 현저히 감소

**지도 스케일의 영향**:[1]

- $s_T = 5-10$ 범위: 최적의 편집 강도
- $s_I = 1-1.5$ 범위: 최적의 이미지 보존

#### 4.3 주요 한계

**공간 추론 부족**:[1]

모델은 다음 작업에서 어려움을 겪습니다:
- 객체 개수 세기 (예: "두 개의 컵을 테이블에 놓고 하나는 의자에")
- 공간 배치 변경 (예: "왼쪽으로 이동")
- 객체 위치 교환

**해상도 및 세부사항**:[1]

- 생성 품질이 기초 Stable Diffusion 모델의 성능에 의존
- 높은 해상도에서의 일관성 유지 어려움

**편향 문제**:[1]

- 기초 모델과 데이터의 편향이 상속됨
- 직업과 성별의 상관관계 등 사회적 편향 존재

***

### 5. 모델의 일반화 성능 향상 가능성

#### 5.1 현재 일반화 강점

**제로샷 성능**:[1]

합성 데이터로만 학습했음에도:
- 실제 이미지에 직접 적용 가능
- 학습에 사용되지 않은 새로운 명령어 형식 처리 가능
- 다양한 예술 매체와 스타일 전이

#### 5.2 최신 연구 기반 개선 방향

**최근 발전**:[2][3][4]

1. **InstructCV (2024)**: InstructPix2Pix 아키텍처를 확장하여 분할, 객체 감지, 깊이 추정 등 다양한 컴퓨터 비전 작업에 적용. 다중 작업 학습으로 상호 강화를 통해 개별 학습 대비 개선.[2]

2. **InstructGIE (2024)**: 일반화 성능 향상을 위해:[3]
   - VMamba 블록과 편집-시프트 매칭 전략으로 문맥 학습 기능 강화[3]
   - 언어 명령 통합으로 편집 의미와의 정렬 개선[3]
   - 시각적 프롬프트와 편집 명령을 포함한 새로운 데이터셋 구축[3]
   
   결과적으로 보이지 않은 작업에 대한 강건한 일반화 성능 달성.[3]

3. **ICEdit (2024)**: 대규모 확산 트랜스포머(DiT)의 내재된 이해도와 생성 능력을 활용하여:[5]
   - 문맥 기반 편집 패러다임 (아키텍처 수정 없음)[5]
   - 최소한의 매개변수 효율적 미세조정 (0.1% 데이터, 1% 학습 가능 매개변수)[5]
   - 상태 최첨단 편집 성능 달성[5]

4. **InsightEdit (2024)**: 고품질 데이터셋과 다중모달 대규모 언어 모델(MLLM) 활용:[6]
   - 고충실도 이미지-편집 쌍 생성 자동화 파이프라인[6]
   - 텍스트와 시각 특징을 통합한 2스트림 브리징 메커니즘[6]
   - 복잡한 명령 준수와 배경 일관성 개선[6]

#### 5.3 일반화 향상의 핵심 메커니즘

**데이터 품질 개선**:[2][3]
- 합성 데이터 필터링의 중요성 (CLIP 기반 방향 유사도)
- 다양한 편집 유형의 포함
- 고해상도 학습 데이터

**멀티태스크 학습**:[2]
- 여러 작업을 동시에 학습하여 작업 간 보조 효과
- 통합된 언어 인터페이스로 일반화 성능 강화

**모델 아키텍처**:[4][5]
- Diffusion Transformer (DiT) 기반: 장거리 의존성 포착 개선
- 병렬 처리로 고해상도 편집 가능

***

### 6. 앞으로의 연구에 미치는 영향 및 고려사항

#### 6.1 주요 영향

**패러다임 변화**:[1]

- 이미지 편집을 "명령 순종" 문제로 재정의
- 합성 데이터를 통한 대규모 멀티모달 모델 학습의 가능성 입증
- 사전학습 모델의 지식 조합을 통한 새로운 작업 창출

**다운스트림 응용**:[2][3]

- 일반화된 비전 모델로 확장 (InstructCV, InstructDiffusion)
- 의료 이미징, 저수준 이미지 처리 등 특화 적용
- 웹 기반 편집 도구의 가능성

#### 6.2 미래 연구 시 고려할 점

**기술적 도전**:[7][3][1]

1. **공간 추론 개선**: 명령어에서 공간 관계를 더 정확히 이해하는 메커니즘 필요[1]

2. **고해상도 생성**: 현재 512×512 이상 해상도에서의 일관성 유지 어려움 → DiT 기반 접근으로 개선 중[4]

3. **객체 레벨 제어**: 특정 객체 격리 및 정밀한 편집을 위한 마스크 생성 메커니즘[3]

4. **사람 중심 피드백**: 강화학습 기반 인간 피드백 통합으로 인간 의도 정렬 개선[1]

**데이터 관련 고려사항**:[7][5]

1. **데이터 효율성**: InstructPix2Pix는 450K+ 예제 필요. 최신 연구는 적은 데이터로 학습 가능한 방향 모색 중 (0.1% 데이터로 성능 유지)[5]

2. **편향 완화**: 학습 데이터와 기초 모델의 편향을 줄이는 방법 개발 필요[1]

3. **다중 데이터 소스**: LAION의 노이즈 문제를 해결하기 위한 고품질 데이터셋 구축 (AdvancedEdit 등)[6]

**평가 메트릭**:[8][1]

- CLIP 기반 메트릭의 한계 인식 (완전한 의미적 일치 평가 불가)
- 사람 평가의 중요성 증대
- 작업별 특화된 평가 지표 개발

**확장성과 일반화**:[7][3]

1. 보이지 않은 명령 유형에 대한 강건성
2. 다양한 스타일과 도메인 간 전이 학습
3. 멀티터전 학습 프레임워크의 효율성

***

## 결론

**InstructPix2Pix**는 인간 중심의 자연언어 명령어를 통한 이미지 편집이라는 새로운 패러다임을 확립했습니다. 합성 데이터 생성 및 조건부 확산 모델 학습의 혁신적 방법론은 이후 InstructCV, InstructDiffusion, InstructGIE 등으로 발전했습니다.[4][2][3][1]

**향후 연구의 주요 방향**은:[5][6]

- **효율성**: 더 적은 데이터와 컴퓨팅으로 고성능 달성
- **정밀성**: 공간 추론과 세밀한 객체 제어 개선
- **일반화**: 다양한 도메인과 작업에 대한 견고한 성능
- **실용성**: 인간 피드백과 상호작용을 통한 반복적 개선

특히 Diffusion Transformer 기반의 최신 접근법들은 장거리 의존성 포착과 고해상도 편집에서 이전 방법을 초월하고 있으며, 멀티모달 대규모 언어 모델의 통합은 더욱 정교한 명령 이해를 가능하게 하고 있습니다.[6][5]

***

### 참고 자료

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/04d6abea-5f71-4315-8589-9fbe06d8c3be/2211.09800v2.pdf)
[2](https://arxiv.org/abs/2410.06825)
[3](https://dl.acm.org/doi/10.1145/3637528.3672040)
[4](https://ieeexplore.ieee.org/document/10638254/)
[5](https://arxiv.org/abs/2404.04125)
[6](https://arxiv.org/abs/2401.13313)
[7](https://arxiv.org/abs/2407.03056)
[8](https://arxiv.org/abs/2402.05859)
[9](https://ieeexplore.ieee.org/document/10943562/)
[10](https://ieeexplore.ieee.org/document/10776968/)
[11](https://dl.acm.org/doi/10.1145/3664647.3681213)
[12](http://arxiv.org/pdf/2410.10497.pdf)
[13](https://arxiv.org/abs/2401.15657)
[14](http://arxiv.org/pdf/2411.15099.pdf)
[15](https://arxiv.org/html/2312.16794v1)
[16](http://arxiv.org/pdf/2404.16637.pdf)
[17](http://arxiv.org/pdf/2401.17547.pdf)
[18](http://arxiv.org/pdf/2310.00390.pdf)
[19](https://arxiv.org/html/2502.03950v1)
[20](https://arxiv.org/html/2504.20690v3)
[21](https://poppyxu.github.io/InsightEdit_web/)
[22](https://blog.paperspace.com/using-diffusion-models-for-image-augmentation-tasks/)
[23](https://openreview.net/forum?id=Nu9mOSq7eH)
[24](https://proceedings.neurips.cc/paper_files/paper/2023/file/c7138635035501eb71b0adf6ddc319d6-Paper-Conference.pdf)
[25](https://eccv.ecva.net/virtual/2024/poster/671)
[26](https://huggingface.co/blog/instruction-tuning-sd)
[27](https://openaccess.thecvf.com/content/CVPR2023/papers/Brooks_InstructPix2Pix_Learning_To_Follow_Image_Editing_Instructions_CVPR_2023_paper.pdf)
[28](https://www.sciencedirect.com/science/article/abs/pii/S0893608024007019)
[29](https://proceedings.neurips.cc/paper_files/paper/2024/file/98b2b307aa4aa323df2ba3a83460f25e-Paper-Conference.pdf)
[30](https://arxiv.org/abs/2310.00390)
[31](https://arxiv.org/abs/2312.04960)
[32](https://arxiv.org/pdf/2309.03895.pdf)
[33](https://arxiv.org/html/2403.09394v1)
[34](https://aclanthology.org/2023.emnlp-main.824.pdf)
[35](https://arxiv.org/html/2408.08601v1)
[36](https://arxiv.org/pdf/2212.03191.pdf)
[37](https://arxiv.org/pdf/2310.09478.pdf)
[38](https://arxiv.org/pdf/2308.06595.pdf)
[39](https://arxiv.org/abs/2403.05018)
[40](https://arxiv.org/abs/2411.03286)
[41](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/12016.pdf)
[42](https://github.com/AlaaLab/InstructCV)
[43](https://arxiv.org/html/2403.05018v1)
[44](https://eccv.ecva.net/virtual/2024/poster/667)
[45](https://openaccess.thecvf.com/content/CVPR2024/html/Geng_InstructDiffusion_A_Generalist_Modeling_Interface_for_Vision_Tasks_CVPR_2024_paper.html)
