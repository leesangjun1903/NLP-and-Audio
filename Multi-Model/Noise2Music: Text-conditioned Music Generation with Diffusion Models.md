
# Noise2Music: Text-conditioned Music Generation with Diffusion Models

## 1. 논문의 핵심 주장과 주요 기여

**Noise2Music**은 Google Research에서 발표한 텍스트 조건부 음악 생성을 위한 캐스케이드 확산 모델(Cascaded Diffusion Models) 기반 시스템입니다. 이 논문의 핵심 주장은 다음과 같습니다.[1]

**핵심 주장**

음악 생성 분야에서 기존의 자동회귀(autoregressive) 접근법의 한계를 극복하기 위해 확산 모델 기반의 캐스케이드 생성 파이프라인이 효과적이며, 대규모 자동 라벨링 데이터셋을 활용하면 고품질의 텍스트-음악 매핑이 가능하다는 것입니다.[1]

**주요 기여**

1. **캐스케이드 확산 모델 아키텍처**: 중간 표현(스펙트로그램 또는 저충실도 오디오)을 생성하는 제너레이터 모델과 최종 고충실도 오디오를 생성하는 캐스케이더 모델로 구성된 이중 구조[1]

2. **대규모 음악-텍스트 쌍 데이터셋**: LaMDA와 MuLan을 활용한 자동 의사 라벨링 방식으로 약 150,000시간의 음악 데이터에 대한 미세한 의미론적 설명 생성[1]

3. **MuLaMCap 데이터셋**: AudioSet 기반으로 400,000개의 음악-텍스트 쌍으로 구성된 새로운 평가 데이터셋 공개[1]

4. **비교 가능한 성능 달성**: 동시대 모델들(Riffusion, Mubert)을 능가하는 FAD(Fréchet Audio Distance)와 MuLan 유사도 점수 달성[1]

***

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조 및 성능

### 2.1 해결하고자 하는 문제

**근본적 문제점**

텍스트 기반 음악 생성 분야는 다음과 같은 핵심 도전과제를 직면하고 있습니다:[1]

- **데이터 부족**: 고품질의 텍스트-음악 쌍 데이터가 매우 제한적
- **음악의 복잡성**: 음악은 장시간의 일관된 구조, 중첩된 악기, 섬세한 음악적 뉘앙스를 포함
- **의미론적 정렬**: 자유로운 형식의 텍스트 설명과 실제 음악 특성 간의 매핑이 비자명함
- **고충실도 생성**: 24kHz 고충실도 오디오의 직접 생성은 계산적으로 매우 비효율적
- **미세한 속성 제어**: 장르, 악기, 분위기뿐만 아니라 활동("운전 음악") 또는 감정("차분한 느낌") 같은 주관적 속성의 표현

### 2.2 제안하는 방법

**캐스케이드 확산 기반 접근**

Noise2Music은 순차적 정제 프레임워크를 제안합니다:[1]

**단계 1: 제너레이터 모델 (Generator Model)**

텍스트 조건부로 중간 표현을 생성합니다. 두 가지 옵션이 있습니다:

- **파형 기반**: 3.2kHz 저충실도 오디오 생성
- **스펙트로그램 기반**: 80채널 로그-멜 스펙트로그램 생성

**단계 2: 캐스케이더 모델 (Cascader Model)**

제너레이터의 중간 표현과 텍스트 조건을 결합하여:

- **파형 캐스케이더**: 16kHz 오디오 생성
- **스펙트로그램 보코더**: 스펙트로그램에서 16kHz 오디오로 변환

**단계 3: 초고해상도 캐스케이더**

경량 모델이 16kHz에서 24kHz로 업샘플링

**확산 모델의 수식**

확산 모델은 다음과 같은 기본 손실함수로 학습됩니다:[1]

$$\mathcal{L} = \mathbb{E}_{x,c,\epsilon,t}\left[ w_t \|\epsilon_\theta(x_t, c, t) - \epsilon\|^2 \right]$$

여기서:
- \(x_t\): 가우시안 확산으로 손상된 샘플
- \(c\): 텍스트 조건
- \(\epsilon_\theta\): 노이즈 예측 네트워크
- \(w_t\): 시간-의존적 가중치 함수

노이즈 일정은 다음과 같이 매개변수화됩니다:[1]

$$x_t = \alpha_t x + \sigma_t \epsilon, \quad \text{where} \quad \alpha_t = \sqrt{1 - \sigma_t^2}$$

**추론 과정 (Ancestral Sampling)**

$$x_s = \frac{\alpha_s}{\alpha_t} x_t - (1 - e^{\lambda_t - \lambda_s}) \cdot \frac{\alpha_s}{\alpha_t} \cdot \sigma_t \cdot \epsilon_\theta(x_t, c, t) + \tilde{\sigma}^{1-\gamma}_{s|t} \cdot \sigma^{\gamma}_{t|s} \cdot \tilde{\epsilon}$$

여기서 \(\gamma\)는 확산 과정의 확률성을 제어합니다.[1]

**분류기-자유 지도 (Classifier-Free Guidance, CFG)**

조건과 조건 없는 노이즈 예측을 결합합니다:[1]

$$\tilde{\epsilon}_\theta = \epsilon_\theta(x_t, \emptyset) + w(\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset))$$

여기서 \(w > 1\)은 조건 준수를 강화하는 지도 척도입니다.

### 2.3 모델 구조

**1D Efficient U-Net 아키텍처**

Noise2Music은 1D 효율적 U-Net을 핵심 아키텍처로 사용합니다:[1]

**구조 특징**:

- **다운샘플링-업샘플링 블록**: 대칭적 구조로 직렬 연결을 통해 다중 스케일 표현 학습
- **잔여 연결 (Residual Connections)**: 다운샘플링 출력이 업샘플링 출력에 직접 연결
- **입력 경로 (Four Input Paths)**:
  1. 노이즈가 추가된 신호 \(x_t\) (항상 포함)
  2. 확산 시간 \(t\) (시간 임베딩으로 인코딩)
  3. 텍스트 프롬프트 (크로스 어텐션을 통해)
  4. 정렬된 압축 표현 (캐스케이더의 경우)

**블록 구조**:

각 다운/업샘플링 블록은 다음으로 구성됩니다:
- 1D 합성곱 층
- 자기 주의(Self-Attention) 층 (선택적)
- 크로스 어텐션(Cross-Attention) 층 (텍스트 조건)
- 결합 층(Combine Layer) (시간 임베딩 상호작용)

**텍스트 인코딩**

T5 인코더를 사용하여 자유 형식의 텍스트를 토큰 임베딩 시퀀스로 변환합니다. 크로스 어텐션 메커니즘을 통해 U-Net 블록에 통합됩니다.[1]

**모델 파라미터**

| 모델 | 파라미터 수 | 훈련 샘플 | 손실 가중치 | 노이즈 일정 |
|------|----------|---------|----------|----------|
| 파형 제너레이터 | 724M | 1.6M | 단순화 | 코사인 |
| 파형 캐스케이더 | 487M | 460k | 단순화 | 선형 |
| 스펙트로그램 제너레이터 | 745M | 1.8M | 시그마 | 선형 |
| 스펙트로그램 보코더 | 25.7M | 840k | 단순화 | 선형 |
| 초고해상도 캐스케이더 | 81M | 270k | 단순화 | 선형 |

### 2.4 의사 라벨링을 통한 데이터 마이닝

**문제**: 고품질의 음악-텍스트 쌍이 극히 부족

**해결책**: MuLan과 LaMDA를 활용한 자동 라벨링[1]

**프로세스**:

1. **캡션 어휘 구축**:
   - **LaMDA-LF** (4M): 150,000개 인기곡의 제목과 아티스트로부터 생성된 긴 형식 설명
   - **Rater-LF** (35K): 전문가가 작성한 MusicCaps 캡션
   - **Rater-SF** (24K): 짧은 형식 태그 (예: "50's pop", "passionate male vocal")

2. **MuLan을 통한 점수 계산**:

각 음악 클립에 대해 \(K\)개의 상위 유사 캡션 선택:

$$\text{similarity}(\text{audio}, \text{caption}) = \cos(\mathbf{e}_{\text{audio}}, \mathbf{e}_{\text{caption}})$$

여기서 \(\mathbf{e}\)는 MuLan 임베딩입니다.

3. **빈도 기반 샘플링**:

\(K'\) 개의 캡션을 역빈도 확률로 샘플링하여 라벨 분포 균형 조정:

$$P(\text{caption}_i) = \frac{1/\text{frequency}_i}{\sum_j 1/\text{frequency}_j}$$

**결과**: 약 150,000시간의 음악에 미세한 의미론적 설명 부여[1]

### 2.5 성능 평가 및 향상

**평가 지표**

1. **Fréchet Audio Distance (FAD)**

세 가지 오디오 인코더를 사용한 다차원 평가:[1]

- **FAD_VGG**: YouTube-8M으로 훈련된 VGG1 임베딩 (일반 오디오 품질)
- **FAD_Trill**: 음성 학습 기반 (음성 품질)
- **FAD_MuLan**: 음악-텍스트 의미론 포착

2. **MuLan 유사도 점수**

텍스트 프롬프트와 생성된 오디오 간의 의미론적 정렬:

```math
\text{MuLan\_sim} = \frac{1}{N} \sum_{i=1}^{N} \cos(\mathbf{e}_{\text{text}_i}, \mathbf{e}_{\text{audio}_i})
```

3. **인간 평가**

MusicCaps 테스트셋에서 의미론적 정렬에 대한 쌍별 비교

**성능 결과**[1]

| 모델 | FAD_VGG | FAD_Trill | FAD_MuLan | 텍스트-오디오 유사도 |
|------|---------|----------|-----------|-----------------|
| Riffusion | 13.371 | 0.763 | 0.487 | 0.342 |
| Mubert | 9.620 | 0.449 | 0.366 | 0.323 |
| **Noise2Music 파형** | **2.134** | **0.405** | **0.110** | **0.478** |
| **Noise2Music 스펙트로그램** | **3.840** | **0.474** | **0.180** | **0.434** |

**핵심 성능 향상**:

- **FAD_VGG**: Riffusion 대비 84% 개선, Mubert 대비 78% 개선
- **MuLan 유사도**: 동시대 최고 수준(MusicLM과 비교 가능)
- **인간 평가**: 1,200개의 쌍별 비교에서 959승 (MusicLM: 718승)

**추론 비용**[1]

| 모델 | 시간/단계 (ms) | 단계 수 | 총 시간 (s) |
|------|-------------|--------|-----------|
| 파형 제너레이터 | 25.0 | 1000 | 25.0 |
| 파형 캐스케이더 | 75.0 | 800 | 60.0 |
| 스펙트로그램 제너레이터 | 8.3 | 1000 | 8.3 |
| 스펙트로그램 보코더 | 29.9 | 100 | 0.3 |
| 초고해상도 캐스케이더 | 71.7 | 800 | 57.3 |

***

## 3. 모델의 일반화 성능 향상 가능성 (핵심 포커스)

### 3.1 현재 일반화 성능의 강점

**1. 대규모 다양한 훈련 데이터**

약 340,000시간의 음악(6.8M개 음악 파일에서 추출)을 포함하는 대규모 훈련셋이 다양한 장르, 악기, 스타일에 대한 일반화를 지원합니다.[1]

**2. 미세 의미론적 라벨의 효과**

메타데이터 태그(장르, 아티스트)만으로는 부족하며, 의사 라벨링을 통한 미세한 의미론적 설명(예: "분위기", "활동", "작곡 요소")이 다음을 향상시킵니다:[1]

- 분위기 표현력 ("차분한 느낌", "에너지 있는")
- 활동 기반 조건 ("운전 음악", "휴식 음악")
- 미세한 악기 특성 설명

**3. 계층적 캐스케이드 접근**

저충실도 → 중간 충실도 → 고충실도로의 순진행은:[1]

- 초기 단계에서 의미론적 일관성 보장
- 후기 단계에서 음향적 세부사항 개선
- 각 단계가 특정 추상화 수준에 최적화될 수 있게 함

### 3.2 일반화 성능 제한 요인

**1. 데이터 분포 편향**

논문에서 인정하는 바와 같이:[1]

> "음악 샘플은 음악의 녹음 및 디지털화가 고르지 못하여 전역 음악 문화의 제한된 말뭉치를 반영할 수 있습니다."

이는 다음을 야기합니다:
- 과대 표현된 장르 (서양 팝/록)
- 과소 표현된 장르 (비서양 전통 음악, 신흥 장르)
- 문화적 뉘앙스 손실 위험

**2. 분포 외 프롬프트의 약한 성능**

논문의 질적 분석 섹션에서:[1]

> "우리의 모델들은 종종 분포 외 프롬프트에서 고품질 오디오를 생성하는 데 어려움을 겪습니다."

예시:
- 매우 구체적인 문화적 장르 요청
- 혼합된 스타일 ("1300년대 푸가 하프시코드와 일렉트로 팝 융합")
- 자동생성으로 학습되지 않은 창의적 조합

**3. 성별/민족성 관련 편향**

논문에서 광범위하게 논의한 윤리적 우려:[1]

- 음성 생성에서의 "모방" 위험 (예: "영혼 있는 보컬" 요청 시 "검은 가수 모방")
- 문화적 편견의 습득 및 증폭
- 라벨링 과정에서의 고정관념 전파

### 3.3 일반화 성능 향상 방안

**1. 데이터 집합 구성 개선**

```
방안: 지리적으로 다양한 음악 데이터 포함
예: 비서양 음악 문화권, 신흥 장르 의도적 수집
결과: 문화적 대표성 향상 → 더 포용적인 생성
```

**2. 적응형 미세 튜닝 (Fine-tuning)**

특정 음악 스타일이나 문화 영역에 대한:

$$\theta' = \theta - \eta \nabla_\theta \mathcal{L}_{\text{domain}}$$

여기서 \(\mathcal{L}_{\text{domain}}\)은 해당 도메인의 손실입니다.

**3. 분포 외 일반화 강화**

대비 학습(Contrastive Learning)을 활용하여 텍스트-오디오 임베딩 공간 개선:

$$\mathcal{L}_{\text{contrastive}} = -\log \frac{\exp(\cos(\mathbf{e}_{\text{text}}, \mathbf{e}_{\text{audio}}) / \tau)}{\sum_i \exp(\cos(\mathbf{e}_{\text{text}}, \mathbf{e}^i_{\text{audio}}) / \tau)}$$

**4. 안정화된 분류기-자유 지도**

현재 CFG 적용 시 과도한 지도척도(\(w > 5\))에서 고뭉 붕괴 위험이 있습니다. 개선책:[1]

- **노이즈 의존적 CFG 스케줄**: 초기 단계에서는 낮은 가중치, 후기 단계에서 높은 가중치
- **동적 CFG**: 모델의 불확실성에 기반한 적응형 지도

### 3.4 전이 학습 및 도메인 적응

**다른 오디오 생성 작업으로의 전이**

논문의 미래 방향에서 제시된 것처럼:[1]

> "흥미로운 방향은 본 작업에서 훈련된 모델을 음악 완성 및 수정 같은 다양한 오디오 작업으로 세밀하게 조정하는 것입니다."

이는 다음을 시사합니다:

- **음악 완성**: 부분 음악 → 전체 곡 확장
- **음악 편집**: 특정 악기 추출, 스타일 변환
- **음성 제어**: 멀티모달 조건화 (텍스트 + 참조 오디오)

***

## 4. 논문이 앞으로의 연구에 미치는 영향 및 향후 고려사항

### 4.1 학술적 영향

**1. 확산 모델의 오디오 도메인 입증**

Noise2Music은 다음을 입증했습니다:[1]

- 이미지 생성에서의 성공(Imagen, DALL-E 3)을 음악 도메인으로 확장 가능
- 캐스케이드 아키텍처가 긴 시퀀스(30초) 생성에 효과적
- 텍스트 인코딩이 음악의 미세한 속성 캡처 가능

**후속 연구에 미친 영향**:

- **2023-2024년의 다양한 후속 모델들**:[2][3][4]
  - **MusicFlow** (2024): 흐름 매칭 기반 대안으로 캐스케이드 확산 개선[3]
  - **JEN-1** (2023): 자기회귀와 비자기회귀 결합으로 다중 작업 수행[4]
  - **MusicLDM** (2023): Stable Diffusion의 음악 적응[5]

**2. 의사 라벨링의 체계적 접근**

MuLan + LaMDA를 통한 자동 라벨링 방식은:[1]

- 주석 데이터의 부족 문제 해결 경로 제시
- 대규모 모델(LLM)을 라벨링 도구로 활용하는 패턴 확립
- 음악 검색, 캡셔닝 등 다른 작업에도 적용 가능

### 4.2 기술적 기여 및 영향

**1. 1D U-Net 아키텍처의 효율성**

효율적 U-Net의 1D 적응이:[1]

- 음성, 음악, 일반 오디오 생성의 표준 아키텍처로 채택
- 자기/크로스 어텐션의 선택적 배치로 계산 효율성 달성
- 다양한 입력 경로의 유연한 설계

**영향**: AudioLDM, MusicLDM 등 후속 모델들이 유사 구조 채택[6]

**2. 분류기-자유 지도의 최적화**

CFG의 적용에서:[1]

- 지도 척도와 추론 품질 간 비선형 관계 발견
- 과도한 지도의 부작용(포화, 고뭉 붕괴) 실증
- "동적 클리핑" 기법으로 문제 완화

**후속 연구의 방향**:

- **2024-2025년 CFG 개선 연구 다수 출현**:[7][8][9]
  - 적응형 CFG (A-CFG): 동적 저신뢰 마스킹[10]
  - 비선형 CFG 일반화: 고차원에서의 이론적 분석[11]
  - 분류기 중심 관점: 결정 경계 조작 해석[12]

### 4.3 실무적 응용 및 산업 영향

**1. 콘텐츠 생성 워크플로우 혁신**

Noise2Music이 시연한 고품질 텍스트-음악 변환은:[1]

- 독립 영상 제작자의 음악 제작 비용 감소
- 팟캐스트, 광고, 게임 개발의 배경음악 자동화
- 뮤지션의 새로운 창의적 도구로 활용

**산업 규모의 성장**:[13]

- 2023년: AI 음악 시장 규모 3.9억 달러
- 2033년 예상: 387억 달러 (28.8% CAGR)

**2. 윤리적 고려사항의 규범 설정**

논문의 "광범위한 영향" 섹션은:[1]

- 생성 AI의 문화적 편향에 대한 조기 경고
- 모방 및 문화적 이용 우려의 구체적 사례화
- 책임감 있는 모델 개발 실천 모범

**후속 영향**:

- 음악 생성 모델의 필수 윤리 검토 관행 정착
- 문화적 표현의 다양성에 대한 학계의 관심 증대

### 4.4 향후 연구 시 고려할 점

#### (1) 기술적 고려사항

**다중 모달 조건화**

현재: 텍스트만 사용

향후 개선:
```
텍스트 + 참조 오디오 + 이미지 등 다중 입력
예: 텍스트 설명 + 기존 곡의 오디오 참조
→ 더 정교한 의도 표현 가능
```

**장기 음악 생성**

현재: 30초 고정 길이

과제:
- 음악의 장시간 구조 학습 (곡 형식, 반복 구조)
- \(30\) 초 이상의 일관된 생성

가능한 해결책:
- 계층적 음악 표현 (절 → 구절 → 소절)
- 재귀적 또는 오토레그레시브 확장
- 음악 이론 기반의 제약 조건 통합

**제어 가능성 향상**

현재: 자유 형식 텍스트에만 의존

개선 방향:
- 명시적 음악 매개변수 제어 (BPM, 조, 주요 악기)
- 구조적 제약 (곡 길이, 섹션 경계)
- 스타일 보존 편집 (특정 요소 유지)

#### (2) 데이터 및 평가 관련

**더 포용적인 데이터셋 구성**

논문의 한계 인정:[1]

> "음악 샘플은 음악의 녹음 및 디지털화가 고르지 못하여 제한된 음악 샘플과 장르, 지역 내 음악 다양성을 일반화할 수 없습니다."

**필요한 노력**:

- 비서양 음악 문화권의 의도적 포함 (인도 클래식, 아프리칸 리듬, 중동 악기)
- 신흥 및 혼합 장르 (K-pop, Afrobeats, 하이브리드 형식)
- 장애인 음악가, 여성 아티스트의 대표성 향상

**평가 지표의 한계 극복**

현재 평가 지표의 문제점:[1]

- FAD는 "다양성" vs "현실성" 간 트레이드오프를 정량화하지 못함
- MuLan 유사도는 MuLan 모델 자체의 편향을 상속
- 음악의 창의성, 신규성, 예술적 가치를 측정하지 못함

**개선 방향**:

- 다양한 배경의 음악가에 의한 정성적 평가 확대
- 특정 장르/스타일에 대한 세분화된 평가 메트릭
- 음악 이론 기반의 분석 (조화, 멜로디 일관성, 박자)

#### (3) 윤리 및 사회적 고려

**편향 및 고정관념 완화**

논문이 강조한 위험:[1]

- 성별/민족성 기반 "모방" 위험
- 문화적 장르의 본질화(essentialize)와 단순화
- 소수 커뮤니티 음악의 상품화

**대응 방안**:

- **라벨링 가이드라인**: 문화적으로 민감한 언어 제한
- **모니터링 시스템**: 문제적 출력 자동 탐지
- **커뮤니티 참여**: 관련 음악 전통 보유 커뮤니티와 협력

**아티스트 권리 및 저작권**

현재 상황:
- 훈련 데이터의 출처 및 라이선싱 불명확
- 생성된 음악의 원본 데이터 추적 불가능

**필요한 정책**:

- 투명한 데이터 소싱 문서화
- 아티스트 귀속 및 보상 메커니즘
- 학습 과정에서의 명시적 동의 (옵트-아웃 vs 옵트-인)

#### (4) 음악-학술적 고려

**음악 이론적 일관성**

현재 모델의 한계:
- 음악 문법(harmony, voice leading) 미학습
- 타이밍, 강박 안정성 부족 가능성

**개선 방향**:

음악 이론 제약을 명시적으로 통합:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{diffusion}} + \lambda_{\text{theory}} \mathcal{L}_{\text{theory}}$$

여기서 \(\mathcal{L}_{\text{theory}}\)는 음악 이론 원칙을 인코딩합니다.

***

## 5. 2020년 이후 관련 최신 연구 탐색

### 5.1 텍스트-음악 생성 모델의 진화 계보

**주요 선행 연구 (2020-2022)**

| 연도 | 모델 | 주요 특징 |
|------|------|---------|
| 2020 | Jukebox | 자동회귀, 계층적 토큰화, 아티스트/장르 조건화[2] |
| 2022 | Riffusion | 스펙트로그램 확산, Stable Diffusion 기반[1] |
| 2022 | Mubert | 사전정의 어휘 기반 조건화[1] |

**Noise2Music의 위치 (2023.02)**

Noise2Music은 이 분야의 "턴키 포인트"로 작용:[1]

- 확산 모델의 오디오 도메인 입증
- 캐스케이드 아키텍처의 실용성 입증
- 대규모 의사 라벨링의 효과성 입증

**동시대 및 후속 연구 (2023-2024)**

**A. 확산 기반 접근 (Diffusion-based)**

1. **AudioLDM 2** (2023.08)[6]
   - 잠재 공간 확산 (Latent Diffusion)
   - 자기 감독 사전 학습 (Self-supervised Pretraining)
   - 텍스트-음악 + 텍스트-음성 통합

2. **MusicLDM** (2023.08)[5]
   - Stable Diffusion 아키텍처 적응
   - Beat-synchronous Mixup 전략 (신규성 향상)
   - FAD 기준: Noise2Music과 유사 수준

3. **MusicFlow** (2024.10)[3]
   - 흐름 매칭(Flow Matching) 기반 대안
   - 캐스케이드 구조 유지
   - 추론 속도 개선 주요 목표

4. **JEN-1** (2023.08)[4]
   - "Omnidirectional Diffusion Models"
   - 자기회귀 + 비자기회귀 혼합
   - 음악 삽입(inpainting), 계속(continuation) 가능

**B. 자동회귀 기반 접근 (Autoregressive)**

1. **MusicLM** (2023.01)[14]
   - 계층적 LM (의미론적 → 음향적)
   - MuLan 임베딩 활용 (Noise2Music과 유사)
   - 더 긴 음악 생성 가능 (최대 3분)

2. **MusicGen** (Meta/Facebook, 2023.06)[14]
   - 단일 LM 기반 (단순함)
   - EnCodec 토큰화
   - 실시간 생성 가능

3. **AudioGen** (2022.09)[2]
   - 일반 사운드 이벤트 생성 (음악 포함)
   - Transformer 기반 자동회귀

**C. 다중모달 및 고급 조건화**

1. **MeLFusion** (2024.06)[15]
   - 텍스트 + 이미지 멀티모달 조건화
   - 확산 기반

2. **MuMu-LLaMA** (2024.12)[16]
   - 대규모 언어 모델(LLaMA) 통합
   - 다중 작업: 이해, 생성, 편집
   - AudioLDM 2 + MusicGen 활용

3. **M²UGen** (2024.12)[17]
   - 음악 + 이미지 + 비디오 다중 입력
   - LLaMA 2를 통한 통합

**D. 제어 가능성 강화**

1. **Mustango** (2024.06)[18]
   - 음악 도메인 지식 기반 제어
   - 구조, 악기, 강도 등 세밀한 조정
   - Noise2Music 대비 제어 가능성 5배 향상

2. **MusicRL** (2024.02)[19]
   - 강화 학습으로 인간 선호도 정렬
   - MusicLM 미세 튜닝
   - 텍스트-준수 및 음질 개선

3. **DiffRhythm** (2025.03)[20]
   - 전곡 생성 (vocals + accompaniment)
   - 캐스케이드 확산으로 고속 추론

### 5.2 기술적 진화 트렌드

**1. 확산 vs 자동회귀의 수렴**

초기 (2020-2022): 명확한 분리
- 확산: 이미지 중심, 느린 추론
- 자동회귀: 텍스트/음성, 빠른 추론

현재 (2023-2025): 상호 보완
- 캐스케이드 확산이 자동회귀 속도에 근접
- 자동회귀 모델이 다단계 계층화 채택
- 하이브리드 접근 (JEN-1, Mustango) 등장

**2. 텍스트 인코딩의 고도화**

| 세대 | 기법 | 특징 |
|------|------|------|
| 1세대 | CLIP-like (MuLan) | 단일 벡터 임베딩 |
| 2세대 | T5 시퀀스 | 토큰 수준 표현 |
| 3세대 | 다중 인코더 결합 | 전역 + 로컬 임베딩 혼합[21] |
| 4세대 | LLM 통합 | 대규모 모델 활용 (MuMu-LLaMA) |

**3. 계산 효율성 진보**

| 모델 | 추론 방식 | 속도 |
|------|---------|------|
| Noise2Music (파형) | 캐스케이드 확산 | ~145초/30초 |
| Noise2Music (스펙트로그램) | 캐스케이드 확산 | ~66초/30초 |
| MusicGen | 자동회귀 | 실시간 가능 |
| DiffRhythm | 최적화 캐스케이드 | ~10초/곡 |

**진행 방향**: 확산의 계산 비용을 자동회귀 수준으로 근접화

### 5.3 성능 벤치마크 비교 (최근 모델들)

**FAD 메트릭 비교** (MusicCaps 테스트셋)[22][5][6][1]

| 모델 | 연도 | FAD_VGG | FAD_Trill | 특징 |
|------|------|---------|----------|------|
| Riffusion | 2022 | 13.371 | 0.763 | 기준선 |
| Noise2Music | 2023.02 | **2.134** | **0.405** | 캐스케이드 확산 |
| AudioLDM 2 | 2023.08 | ~5-6 | ~0.5-0.6 | 잠재 확산 |
| MusicLDM | 2023.08 | ~4-5 | ~0.47 | Stable Diffusion 적응 |
| MusicGen | 2023.06 | ~3-4 | ~0.4 | 자동회귀 LM |

**의미론적 정렬 (MuLan 유사도)**

| 모델 | 텍스트-오디오 | 오디오-오디오 | 평가 |
|------|-------------|-------------|------|
| Noise2Music 파형 | **0.478** | **0.489** | 최고 수준 |
| MusicLM | 0.51 | - | 동시대 최강 |
| MusicGen | ~0.47 | - | 유사 수준 |

### 5.4 분류기-자유 지도(CFG) 연구의 진화

Noise2Music이 CFG의 한계를 식별한 이후, 광범위한 개선 연구 등장[44-72]:[1]

**2023-2024년 CFG 개선 방향**

| 방법 | 핵심 아이디어 | 적용 분야 |
|------|----------|---------|
| **노이즈 의존적 스케줄** | 고노이즈 단계에서는 CFG 약화 | 일반 확산 모델 |
| **동적 CFG** | 모델 불확실성 기반 적응형 가중치 | 이미지/비디오 생성 |
| **적응형 CFG (A-CFG)** | 저신뢰 토큰 동적 마스킹 | 언어 모델 |
| **비선형 CFG 일반화** | Power-law 일반화 | 고차원 데이터 |
| **분류기 중심 관점** | 결정 경계 조작 | 이론적 이해 |

**최신 발견** (2025년):[23][11][12]

- **고차원의 축복**: 충분히 고차원 데이터에서 CFG 왜곡이 사라짐[11]
- **고정점 반복 관점**: CFG를 고정점 찾기로 재해석하여 더 효율적 알고리즘 제안[23]

### 5.5 윤리 및 책임감 있는 AI 논의의 진화

Noise2Music의 "광범위한 영향" 섹션이 촉발한 연쇄 효과:[1]

**2023-2024년의 관련 연구**

1. **음악 생성에서의 편향 분석**[24]

   > "AI 음악 생성 모델은 훈련 데이터의 편향을 학습하고 증폭한다. 특히 문화적 장르 표현에서 심각하다."

2. **저작권 및 데이터 출처 이슈**[13]

   - 학습에 사용된 음악의 법적 지위 불명확
   - 아티스트 보상 메커니즘 부재
   - 동의(consent) 절차의 필요성 증대

3. **음악 산업 영향 분석**[25]

   - AI 음악의 보급이 독립 아티스트에 미치는 영향
   - 플랫폼의 경제적 선호도 변화 가능성

***

## 6. 종합 분석 및 결론

### 6.1 Noise2Music의 위상

**학술적 위상**: 

확산 모델의 오디오 도메인 실증이라는 마일스톤을 기록했으며, 후속 모델들의 설계 철학(캐스케이드 구조, 텍스트 인코딩, 의사 라벨링)에 직접적 영향을 미침.[3][4][5][6][1]

**기술적 기여 순위**:

1. **캐스케이드 확산의 실무적 입증** (순위: 1위)
   - 이론에서 실제 고음질 생성으로의 전환
   - 후속 모델 대다수가 채택

2. **대규모 의사 라벨링 방식론** (순위: 2위)
   - 데이터 부족 문제의 실마리 제시
   - 다른 도메인(비전, 자연어) 확장 적용

3. **종합적 윤리 고찰** (순위: 3위)
   - 생성 AI의 문화적 편향에 대한 조기 경고
   - 책임감 있는 AI 개발의 모범

### 6.2 현재의 제약 및 개선 과제

| 영역 | 제약 | 개선 필요도 | 예상 해결 시점 |
|------|------|-----------|-------------|
| 분포 외 일반화 | 창의적/혼합 스타일 약함 | 높음 | 2025년 |
| 장시간 생성 | 30초 고정 | 높음 | 2024-2025년 |
| 문화적 대표성 | 서양 음악 편향 | 높음 | 2025년 이후 |
| 제어 가능성 | 텍스트 기반만 | 중간 | 2024년 |
| 생성 속도 | ~145초 (파형) | 중간 | 2024년 |

### 6.3 향후 10년 전망

**2025-2027년**: 

다중모달 조건화, 제어 가능성 강화, 문화적 다양성 포함이 경쟁 지점. Noise2Music의 기본 구조는 산업 표준으로 정착될 가능성 높음.

**2027-2030년**:

- 실시간 고품질 음악 생성 상용화
- 음악가의 보조 창의 도구로 확립
- 저작권 및 윤리 규범의 법제화

**2030년 이후**:

음악 생성 AI와 인간 뮤지션의 상호보완 생태계 형성.

***

## 참고자료 목록

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/bb1589d5-9acd-4a2b-a85e-b219e1565dc3/2302.03917v2.pdf)
[2](https://dl.acm.org/doi/10.1145/3707292.3707367)
[3](http://arxiv.org/pdf/2410.20478.pdf)
[4](https://arxiv.org/pdf/2308.04729.pdf)
[5](https://arxiv.org/pdf/2308.01546.pdf)
[6](http://arxiv.org/pdf/2308.05734.pdf)
[7](https://arxiv.org/abs/2403.11968)
[8](https://arxiv.org/abs/2507.08965)
[9](https://arxiv.org/abs/2506.14399)
[10](https://arxiv.org/abs/2505.20199)
[11](https://www.semanticscholar.org/paper/90c82b06bc3b44165d956aba259c639379592ee3)
[12](https://arxiv.org/abs/2503.10638)
[13](https://www.digitalocean.com/resources/articles/ai-music-generators)
[14](https://arxiv.org/pdf/2306.05284.pdf)
[15](https://openaccess.thecvf.com/content/CVPR2024/papers/Chowdhury_MeLFusion_Synthesizing_Music_from_Image_and_Language_Cues_using_Diffusion_CVPR_2024_paper.pdf)
[16](https://arxiv.org/html/2412.06660v1)
[17](https://arxiv.org/html/2311.11255v3)
[18](http://arxiv.org/pdf/2311.08355.pdf)
[19](https://arxiv.org/pdf/2402.04229.pdf)
[20](https://arxiv.org/html/2503.01183v1)
[21](https://research.samsung.com/blog/Diffusion-based-Text-to-Music-Generation-with-Global-and-Local-Text-based-Conditioning)
[22](https://ieeexplore.ieee.org/document/10447265/)
[23](https://arxiv.org/abs/2510.21512)
[24](https://arxiv.org/html/2409.03715v1)
[25](https://www.forbes.com/sites/virginieberger/2024/12/30/ais-impact-on-music-in-2025-licensing-creativity-and-industry-survival/)
[26](https://arxiv.org/abs/2207.12598)
[27](http://pubs.rsna.org/doi/10.1148/radiol.231971)
[28](https://ieeexplore.ieee.org/document/11248334/)
[29](https://www.semanticscholar.org/paper/06ca869b5e1d3904a7bbb1bc2fadfd0e51068ddc)
[30](https://arxiv.org/pdf/2302.03917.pdf)
[31](https://arxiv.org/ftp/arxiv/papers/2301/2301.13267.pdf)
[32](https://arxiv.org/html/2409.02845v2)
[33](https://picovoice.ai/blog/state-of-generative-ai-for-audio/)
[34](https://aclanthology.org/2024.acl-long.437.pdf)
[35](https://openai.com/index/introducing-our-next-generation-audio-models/)
[36](https://arxiv.org/abs/2302.03917)
[37](https://www.reddit.com/r/MachineLearning/comments/14ygk36/d_overview_of_recent_developments_in/)
[38](https://arxiv.org/abs/2301.11757)
[39](https://developer.nvidia.com/blog/achieving-state-of-the-art-zero-shot-waveform-audio-generation-across-audio-types/)
[40](https://artsmart.ai/blog/ai-in-music-industry-statistics/)
[41](https://arxiv.org/pdf/2305.15719.pdf)
[42](https://ambientartstyles.com/assessing-ai-llms-2/)
[43](https://www.isca-archive.org/interspeech_2024/kim24e_interspeech.pdf)
[44](https://www.sciencedirect.com/science/article/pii/S1319157823003154)
[45](https://musiclm.com)
[46](https://research.google.com/pubs/archive/45871.pdf)
[47](https://arxiv.org/html/2406.04673v1)
[48](https://www.nature.com/articles/s41598-025-13064-6)
[49](https://arxiv.org/html/2405.02801v2)
[50](https://kth.diva-portal.org/smash/get/diva2:1845150/FULLTEXT01.pdf)
[51](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/musiclm/)
[52](https://www.semanticscholar.org/paper/7002ae048e4b8c9133a55428441e8066070995cb)
[53](https://ieeexplore.ieee.org/document/11094644/)
[54](https://arxiv.org/abs/2503.20240)
[55](https://arxiv.org/abs/2407.02687)
[56](https://arxiv.org/html/2503.17593v1)
[57](https://arxiv.org/pdf/2311.00938.pdf)
[58](https://arxiv.org/html/2503.20240v2)
[59](https://arxiv.org/html/2503.10638)
[60](https://arxiv.org/html/2502.07849)
[61](https://arxiv.org/pdf/2307.09568.pdf)
[62](https://theaisummer.com/classifier-free-guidance/)
[63](https://vskadandale.github.io/pdf/MMSP_2020.pdf)
[64](http://www.diva-portal.org/smash/get/diva2:1893350/FULLTEXT01.pdf)
[65](https://apxml.com/courses/intro-diffusion-models/chapter-6-conditional-generation-diffusion/classifier-free-guidance)
[66](https://archives.ismir.net/ismir2020/paper/000046.pdf)
[67](https://studios.disneyresearch.com/2025/04/23/no-training-no-problem-rethinking-diffusion-guidance-for-diffusion-models/)
[68](https://ieeexplore.ieee.org/document/10246115/)
[69](https://arxiv.org/html/2405.18386v3)
[70](http://www.peterholderrieth.com/blog/2023/Classifier-Free-Guidance-For-Diffusion-Models/)
[71](https://apxml.com/courses/intro-diffusion-models/chapter-6-conditional-generation-diffusion/implementing-classifier-free-guidance)
[72](https://docs.openvino.ai/2023.3/notebooks/250-music-generation-with-output.html)
