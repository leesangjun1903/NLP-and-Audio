
# PixelLM: Pixel Reasoning with Large Multimodal Model

> **논문 정보:** Zhongwei Ren et al., *PixelLM: Pixel Reasoning with Large Multimodal Model*, arXiv:2312.02228, **CVPR 2024**
> **저자:** Zhongwei Ren, Zhicheng Huang, Yunchao Wei, Yao Zhao, Dongmei Fu, Jiashi Feng, Xiaojie Jin
> **소속:** Beijing Jiaotong University, University of Science and Technology Beijing, ByteDance Inc., Peng Cheng Laboratory

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장

대형 멀티모달 모델(LMM)이 눈부신 발전을 이루었음에도 불구하고, 다수의 오픈-월드 타겟을 포함하는 이미지 추론 태스크에서 픽셀 수준의 마스크를 생성하는 것은 여전히 도전 과제로 남아 있다. 이를 해결하기 위해 PixelLM을 제안하며, 이는 픽셀 수준 추론 및 이해를 위한 효과적이고 효율적인 LMM이다.

### 1.2 주요 기여 (3가지)

| # | 기여 항목 | 설명 |
|---|-----------|------|
| ① | **PixelLM 모델** | 픽셀 수준 추론을 위한 새로운 LMM |
| ② | **MUSE 데이터셋** | 고품질 멀티 타겟 추론 분할 벤치마크 |
| ③ | **SOTA 달성** | 다수의 벤치마크에서 기존 최고 성능 초과 |

PixelLM은 픽셀 수준 추론 및 이해를 위한 새로운 LMM으로, 임의의 수의 오픈-셋 타겟과 다양한 추론 복잡도를 갖는 태스크를 능숙하게 처리한다. 그 설계는 LMM의 기본 구조를 유지하면서 추가적인 고비용 분할 모델을 피하여, 효율성과 다양한 응용으로의 전이 가능성을 향상시킨다.

MUSE는 고품질 멀티 타겟 추론 분할 데이터셋으로, 향후 연구에서 모델 훈련 및 평가를 용이하게 한다. GPT-4V 기반 데이터 큐레이션 파이프라인을 활용하여 246k 개의 질문-응답 쌍을 생성하였으며, 이는 90만 개의 인스턴스를 포함한다.

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능, 한계

### 2.1 해결하고자 하는 문제

LLM의 성공을 기반으로 구축된 대형 멀티모달 모델은 고수준 시각 인식 및 사용자 상호작용 경험을 크게 향상시켰지만, 대부분은 전역 이미지나 영역에 대한 텍스트 설명을 생성하는 데 그치며 객체 마스크와 같은 픽셀 수준 응답 능력이 제한적이다.

이러한 연구 격차는 이미지 편집, 자율주행, 로보틱스와 같은 세밀한 태스크에서 멀티모달 시스템의 실용적인 응용을 제한한다.

특히, 기존 방법인 LISA의 두 가지 주요 한계가 직접적인 동기가 된다:

첫째, 다수의 타겟 객체를 포함하는 태스크를 처리할 수 없으며, 이는 실제 세계 시나리오에서 필수적이다. 둘째, SAM과 같은 사전 훈련된 이미지 분할 모델에 의존하는데, 이 의존성은 상당한 계산 수요를 유발하고 전체 모델의 성능을 분할 모델의 능력에 묶어 놓아 추가 훈련 스케일업을 통한 성능 향상 가능성을 저해한다.

---

### 2.2 제안하는 방법 (수식 포함)

#### (A) 전체 파이프라인

입력 이미지 $x_{\text{img}}$에 대해, 비전 인코더 $\mathcal{I}$는 $\mathcal{I}(x_{\text{img}})$로부터 $L$개의 멀티스케일 시각 특징 스펙트럼 $I_{\text{img}} = \{ I_{\text{img}}^{\ell} \}\_{\ell=1}^{L}$을 추출한다. 최종 레이어의 출력 $I_{\text{img}}^{L}$은 전역 이미지 정보를 캡슐화하며, 비전-언어 프로젝션 레이어 $p_{V\rightarrow T}$를 통해 언어 공간으로 변환된다. 동시에, 비전-디코더 프로젝션 $p_{V\rightarrow D}$는 모든 $I_{\text{img}}$ 특징을 변환하여 

```math
f_{\text{img}} = \left\{f_{\text{img}}^{\ell}=p_{V\rightarrow D}(I_{\text{img}}^{\ell})\right\}_{\ell=1}^{L}
```

을 생성한다.

LLM의 자동회귀 응답 생성은 다음과 같이 표현된다:

$$y_{\text{res}} = \mathcal{F}\left(p_{V\rightarrow T}(I_{\text{img}}^{L}),\ x_{\text{txt}},\ C_{\text{seg}}\right)$$

여기서 $C_{\text{seg}}$는 세그멘테이션 코드북 토큰, $x_{\text{txt}}$는 입력 텍스트이다.

#### (B) 세그멘테이션 코드북 (Segmentation Codebook)

PixelLM의 핵심은 새로운 경량 디코더와 전체론적 세그멘테이션 코드북이다. 코드북은 서로 다른 시각적 스케일을 참조하는 타겟과 관련된 맥락과 지식을 인코딩하는 학습 가능한 토큰을 포함한다. 픽셀 디코더는 이미지 특징과 함께 코드북 토큰의 숨겨진 임베딩을 기반으로 타겟 마스크를 생성한다.

코드북 내 각 타겟 $i$에 대한 세그멘테이션 토큰 시퀀스는 $N$개의 토큰으로 구성되며, $N > 1$로 설정하는 이유는 다음과 같다:

다수의 타겟 또는 고유한 복잡성이 있는 시나리오에서는 단일 토큰이 타겟 의미론을 완전히 캡슐화하는 능력에 한계가 생기며, LLM이 정확한 텍스트 응답을 제공하더라도 마찬가지이다.

#### (C) 픽셀 디코더의 마스크 생성

픽셀 디코더는 비전 인코더의 특징과 $C_{\text{seg}}$로부터의 숨겨진 임베딩을, 정밀한 세그멘테이션 마스크로 변환하는 방법을 학습하는 데 사용된다. 이는 추가적인 고비용 분할 모델의 필요성을 제거한다. 디코더는 $L$개의 어텐션 블록으로 구성되며, 각 블록은 이미지 특징과 코드북의 서로 다른 스케일에 대응한다. 각 타겟 마스크 생성을 위해, 디코더는 각 스케일 $\ell$에서 순차적으로 마스크 스코어 맵 $m^{\ell}$을 생성하고, 이는 다음 스케일 $\ell-1$에서 더 높은 관련성 영역에 모델의 주의를 유도한다.

스케일별 피처 변조(Feature Modulation):

$$f_{\text{img}}^{\ell\prime} = f_{\text{img}}^{\ell} \odot \sigma(m^{\ell+1})$$

여기서 $f_{\text{img}}^{\ell\prime}$는 스케일 $\ell$에서의 변조된 특징, $\sigma$는 시그모이드 함수, $\odot$은 원소별 곱셈이다.

최종 세그멘테이션 마스크는 모든 스케일의 마스크 맵을 가중 합산하여 산출된다:

$$\hat{M} = \sum_{\ell=1}^{L} \gamma^{\ell} m^{\ell}$$

여기서 $\gamma = [\gamma^{\ell}]_{\ell=1}^{L}$은 각 스케일의 학습 가능한 가중 계수이다.

#### (D) Target Refinement Loss

다수의 타겟을 구별하는 모델의 능력을 향상시키기 위해 Target Refinement Loss를 제안하며, 이는 마스크 품질을 실질적으로 향상시킨다.

전체 훈련 손실은 다음과 같이 구성된다:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{text}} + \lambda_{\text{mask}} \mathcal{L}_{\text{mask}} + \lambda_{\text{refine}} \mathcal{L}_{\text{refine}}$$

- $\mathcal{L}_{\text{text}}$: LLM의 자동회귀 텍스트 생성 손실 (Cross-Entropy)
- $\mathcal{L}_{\text{mask}}$: Binary Cross-Entropy + DICE Loss 결합
- $\mathcal{L}_{\text{refine}}$: 타겟 간 구별을 강화하는 Refinement Loss

---

### 2.3 모델 구조

PixelLM은 간결화된 아키텍처를 특징으로 하며, 네 가지 주요 부분으로 구성된다: (i) 텍스트와 정렬되는 사전 훈련된 CLIP-ViT 비전 인코더, (ii) 대형 언어 모델, (iii) 경량 픽셀 디코더, (iv) 세그멘테이션 코드북. PixelLM은 이미지와 쿼리 텍스트를 처리하여 다양한 타겟에 대한 인터리브드 텍스트 설명과 대응하는 마스크를 생성한다.

```
┌─────────────────────────────────────────────────────────────────┐
│                        PixelLM Architecture                      │
│                                                                   │
│  Input Image ──→ [CLIP-ViT Encoder] ──→ Multi-scale Features    │
│       │              │                    (I_img^1 ~ I_img^L)    │
│       │         p_{V→T}↓          p_{V→D}↓                      │
│  Input Text ──→ [LLM (+ LoRA)] ←── Language-space features      │
│                     │ + C_seg                                     │
│                     ↓                                             │
│               Interleaved Response y_res                          │
│          (Text + Segmentation Token Sequences)                    │
│                     │                                             │
│            [Pixel Decoder] ←── f_img features                    │
│          (L Attention Blocks, multi-scale)                        │
│                     ↓                                             │
│              Final Mask M̂ = Σ γ^ℓ · m^ℓ                        │
└─────────────────────────────────────────────────────────────────┘
```

학습 가능한 LoRA 파라미터가 LLM에 통합되며, CLIP 인코더와 LLM을 제외한 모든 파라미터가 학습 가능하다.

---

### 2.4 성능 향상

MUSE 벤치마크에서 평가한 결과, PixelLM은 SEEM 및 LISA와 같은 모델을 gIoU 및 cIoU 점수에서 능가하면서 계산 비용을 최대 50%까지 절감하였다.

PixelLM은 MUSE, 단일 및 다중 참조 분할을 포함한 다양한 픽셀 수준 이미지 추론 및 이해 태스크에서 탁월한 성능을 발휘하며, 여러 벤치마크에서 잘 확립된 방법들을 능가한다.

| 벤치마크 | 지표 | PixelLM 성능 |
|----------|------|-------------|
| MUSE | gIoU / cIoU | SEEM, LISA 대비 향상 |
| Single Referring Seg | cIoU | SOTA |
| Multi Referring Seg | gIoU / cIoU | SOTA |
| ReasonSeg | gIoU / cIoU | 경쟁력 있는 성능 |

---

### 2.5 한계점

기존 픽셀 그라운딩 모델들은 단일 이미지 설정에서 작동하여, 여러 이미지에 걸친 세밀하고 정교한 비교를 수행하는 능력이 제한된다. 반면 현재의 다중 이미지 이해 모델은 픽셀 수준 그라운딩이 부족하다.

추가적인 한계는 다음과 같다:

1. **비디오 처리 불가**: PixelLM은 추론 분할 태스크에서 다수 객체 처리에 독보적인 장점을 보이나, 이러한 연구들은 여전히 비디오 처리가 불가능하다.
2. **외부 모델 의존성 완전 제거의 어려움**: SAM 의존성을 제거했지만, CLIP-ViT 인코더에는 여전히 의존한다.
3. **고해상도 이미지의 한계**: 기존 MLLM들은 일반적으로 저해상도 시각 토큰에 의존하여, 고해상도 이미지의 세밀한 디테일을 버리는 경우가 많다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 오픈-셋 타겟 처리

PixelLM은 픽셀 수준 추론 및 이해를 위한 새로운 LMM으로, 임의의 수의 오픈-셋 타겟과 다양한 추론 복잡도를 갖는 태스크를 능숙하게 처리한다. 그 설계는 LMM의 기본 구조를 유지하면서 추가적인 고비용 분할 모델을 피하여, 효율성과 다양한 응용으로의 전이 가능성(transferability)을 향상시킨다.

### 3.2 MUSE 데이터셋의 일반화 기여

MUSE는 최초의 포괄적인 멀티 타겟 추론 분할 데이터셋으로, 오픈-셋 개념, 상세한 객체 설명, 복잡한 멀티 타겟 질문-응답 쌍, 인스턴스 수준 마스크 어노테이션이 두드러진다.

광범위한 절제 연구(ablation studies)가 이 데이터셋이 모델의 픽셀 추론 능력을 자극하는 데 효과적임을 확인한다.

### 3.3 LoRA 기반 효율적 파인튜닝의 범용성

훈련 가능한 LoRA 파라미터가 LLM에 통합되며, CLIP 인코더와 LLM을 제외한 모든 파라미터가 훈련 가능하다.

이는 기존 LLM 백본(LLaMA, Vicuna 등)을 그대로 활용하며 새로운 도메인으로 파인튜닝이 가능하다는 것을 의미한다. 향후 방향으로는 더 다양한 데이터셋과 태스크로 PixelLM의 능력을 확장하고, 점점 더 복잡한 멀티모달 데이터 스트림을 활용할 수 있는 더 발전된 학습 목표를 통합하는 것을 고려할 수 있다.

### 3.4 다양한 도메인으로의 확장 가능성

PixelLM은 경량 픽셀 디코더와 세그멘테이션 코드북을 통합하여 외부 분할 모듈을 제거하고, 다중 객체 및 다중 스케일 추론을 지원함으로써 UAV 이미지, 원격 탐사(Remote Sensing), 의료 이미징 등 다양한 특수 도메인으로의 확장 가능성을 보인다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려사항

### 4.1 연구에 미치는 영향

PixelLM은 이미지 분할과 대형 언어 모델 사이의 격차를 해소하는 중요한 진전을 나타내지만, 이 접근 방식의 잠재력을 완전히 실현하기 위해서는 지속적인 연구와 개선이 필요할 것이다. 새로운 픽셀 디코더와 세그멘테이션 코드북을 갖춘 대형 멀티모달 모델인 PixelLM의 도입은 픽셀 수준 이미지 추론 및 이해 분야에서 중요한 발전을 나타낸다.

이러한 연구는 이미지 편집, 자율주행, 로보틱스와 같은 세밀한 태스크에서 멀티모달 시스템의 실용적인 응용을 촉진하는 데 기여한다.

### 4.2 앞으로의 연구 시 고려할 점

**① 비디오/시계열 데이터로의 확장**

이미지 태스크에서 인상적인 성능을 달성했음에도 불구하고, 비디오 처리에는 여전히 불가능하다. 비디오에서 객체 분할을 위해, LLM의 추론 능력을 활용하여 현재의 한계를 극복하는 연구는 매우 드물다.

**② 다중 이미지 추론으로의 확장**

다중 이미지 픽셀 그라운딩 추론 분할 태스크 및 픽셀 수준 그라운딩과 강력한 다중 이미지 추론 능력을 통합하여 맥락이 풍부한 픽셀 그라운딩 설명을 생성하는 방향으로 연구가 발전해야 한다.

**③ Chain-of-Thought(CoT)와의 결합**

일부 방법들은 짧은 응답에서 세그멘테이션 토큰을 단순히 사용함으로써 Chain-of-Thought(CoT)와 같은 추론 과정을 간과하고 있다. 향후 연구에서는 PixelLM과 같은 픽셀 추론 모델에 CoT 메커니즘을 통합하여 더 깊은 추론 과정을 거친 세그멘테이션을 수행하는 것이 중요한 연구 방향이 될 것이다.

**④ 고해상도 이미지 처리**

높은 해상도 구조를 저해상도 의미 추론에 주입하여 시각 토큰 비용을 증가시키지 않고 MLLM이 픽셀 수준 분할을 수행할 수 있도록 하는 아이디어가 필요하다. 아키텍처는 전역 의미와 세밀한 구조 디테일을 별도로 모델링하는 방향으로 발전해야 한다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 주요 관련 연구 비교표

| 모델 | 연도 | 핵심 방법 | 멀티 타겟 | 외부 분할 모델 | 추론 능력 |
|------|------|-----------|-----------|---------------|-----------|
| **CLIP** | 2021 | 대조 학습 시각-언어 정렬 | ❌ | ❌ | △ |
| **SAM** | 2023 | Promptable Segmentation | ❌ | ✅ (자체) | ❌ |
| **SEEM** | 2023 | 다중 방식 인터랙티브 분할 | △ | ✅ | △ |
| **LISA** | 2023 | `<SEG>` 토큰 + SAM | ❌ (단일) | ✅ (SAM) | ✅ |
| **GLaMM** | 2024 | 그라운디드 대화 생성 | △ | ✅ | ✅ |
| **PixelLM** | 2024 | 코드북 + 경량 픽셀 디코더 | ✅ | ❌ (불필요) | ✅ |
| **PixDLM** | 2025 | 이중 경로 인코더 + UAV | ✅ | △ | ✅ |

### 5.2 LISA와의 비교

LISA는 분할 태스크를 위해 SAM을 LLM과 통합하며, 복잡한 명령 추론에 LMM을 사용하는 것을 탐구한다. 그러나 LISA는 이미지 내 단일 타겟 처리에 제한되며, SAM의 통합은 상당한 계산 오버헤드를 추가한다.

LISA 기반으로 구축된 PixelLM은 추론 분할 태스크에서 다수 객체 처리에 독보적인 장점을 보인다.

### 5.3 LISA와 PixelLM의 접근 방식 차이 요약

$$\underbrace{\text{LISA}}_{\text{SAM 의존}}:\ \langle\text{SEG}\rangle \xrightarrow{\text{hidden emb.}} \text{SAM Decoder} \rightarrow \hat{M}$$

$$\underbrace{\text{PixelLM}}_{\text{자립형}}:\ C_{\text{seg}} \xrightarrow{\text{hidden emb.}} \text{Lightweight Pixel Decoder} \rightarrow \hat{M} = \sum_{\ell=1}^{L} \gamma^{\ell} m^{\ell}$$

LISA는 `<SEG>` 토큰과 임베딩-as-마스크 디코더로 멀티모달 LLM을 도입하여 픽셀 수준 출력과 함께 암묵적인 언어 추론을 가능하게 하였다. PixelLM은 경량 픽셀 디코더와 세그멘테이션 코드북을 통합하여 외부 분할 모듈을 추가로 제거하고, 다중 객체 및 다중 스케일 추론을 지원한다.

---

## 📚 참고 자료 및 출처

| # | 제목 | 출처 |
|---|------|------|
| 1 | **PixelLM: Pixel Reasoning with Large Multimodal Model** (Official Page) | https://pixellm.github.io/ |
| 2 | **arXiv:2312.02228** — PixelLM 논문 원문 | https://arxiv.org/abs/2312.02228 |
| 3 | **CVPR 2024 Paper** — PixelLM (CVF Open Access) | https://openaccess.thecvf.com/content/CVPR2024/papers/Ren_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model_CVPR_2024_paper.pdf |
| 4 | **ar5iv** — PixelLM HTML 렌더링 버전 | https://ar5iv.labs.arxiv.org/html/2312.02228 |
| 5 | **GitHub** — MaverickRen/PixelLM | https://github.com/MaverickRen/PixelLM |
| 6 | **arXiv:2308.00692** — LISA: Reasoning Segmentation via Large Language Model | https://arxiv.org/abs/2308.00692 |
| 7 | **CVPR 2024** — LISA (CVF Open Access) | https://openaccess.thecvf.com/content/CVPR2024/html/Lai_LISA_Reasoning_Segmentation_via_Large_Language_Model_CVPR_2024_paper.html |
| 8 | **NeurIPS 2024** — One Token to Seg Them All: Language Instructed Reasoning Segmentation in Videos | https://proceedings.neurips.cc/paper_files/paper/2024/file/0cf3e7eefb9d643e93e16ff1d94090a7-Paper-Conference.pdf |
| 9 | **arXiv:2604.15670** — PixDLM: A Dual-Path Multimodal Language Model for UAV Reasoning Segmentation | https://arxiv.org/html/2604.15670 |
| 10 | **Hugging Face Papers** — PixelLM 페이퍼 페이지 | https://huggingface.co/papers/2312.02228 |
| 11 | **Emergent Mind** — PixelLM 분석 | https://www.emergentmind.com/papers/2312.02228 |
| 12 | **Medium (Utsav Desai)** — PixelLM 리뷰 | https://medium.com/@utsavmdesai/pixellm-pixel-reasoning-with-large-multimodal-model-96b3db5a193c |
| 13 | **ResearchGate** — PixelLM 관련 인용 분석 | https://www.researchgate.net/publication/384161382_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model |
