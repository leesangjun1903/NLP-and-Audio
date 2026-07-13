# Transcending Scaling Laws with 0.1% Extra Compute

## 1. 핵심 주장과 주요 기여

이 논문은 **UL2R (UL2 Restore)**라는 방법을 제안합니다. 핵심 주장은 이미 학습된 대규모 언어모델(PaLM)에 아주 적은 추가 연산(약 0.1~1%)만으로 UL2의 mixture-of-denoisers 목적함수를 사용해 계속 학습시키면, 스케일링 곡선 자체를 근본적으로 개선할 수 있다는 것입니다.

**주요 기여:**
- PaLM 8B, 62B, 540B에 UL2R을 적용한 **U-PaLM** 모델 공개
- 540B 규모에서 약 **2배의 연산 절감 효과** (동일 성능 도달에 필요한 FLOPs가 절반, 약 440만 TPUv4 시간 절약)
- BIG-Bench의 "emergent" 태스크에서 성능을 더 작은 스케일(62B, 8B)에서도 끌어올림
- Infilling(양방향 채우기) 등 새로운 프롬프팅 능력을 PaLM에 부여

## 2. 문제, 방법론, 모델 구조, 성능, 한계

### 해결하고자 하는 문제
기존 causal LM 기반 스케일링 법칙(Kaplan et al., 2020; Hoffmann et al., 2022)은 대규모 학습에 막대한 컴퓨팅 자원을 요구합니다. 저자들은 "이미 학습된 모델을 처음부터 다시 학습하지 않고도 스케일링 곡선을 개선할 수 있는가?"라는 질문에 답하고자 합니다.

### 제안 방법: UL2R

**UL2 mixture-of-denoisers 구성** (세 가지 디노이저):

1. **Regular denoising (R-denoiser)**: 표준 span corruption
$$\text{corruption rate} = 15\%, \quad \text{mean span length} = 3$$

2. **Extreme denoising (X-denoiser)**: 더 긴 span 또는 높은 corruption rate
$$\text{mean span length} = 32 \text{ OR corruption rate up to } 50\%$$

3. **Sequential denoising (S-denoiser)**: PrefixLM 방식
$$\text{noise sampled from start of text to a randomly chosen point}$$

최종 믹스처 비율: $50\%$ S-denoiser, $25\%$ R-denoiser, $25\%$ X-denoiser

**학습 설정:**
- 540B 모델: 20k steps, batch size 32
- 총 추가 토큰: 약 1.3B (전체 학습 대비 약 $0.16\%$)
- 학습률 스케줄: 코사인 감쇠, $10^{-4} \to 10^{-6}$

**Prefix Padding 최적화**: 기존 UL2 파이프라인은 prefix에 먼저 패딩을 적용한 뒤 target과 결합하여 $[\text{prefix}][\text{prefix's padding}][\text{target}]$ 형태로 비효율적이었으나, 이 논문은 prefix와 target을 먼저 결합한 뒤 패딩을 적용하여 샘플 효율성을 개선했습니다.

### 모델 구조
U-PaLM은 PaLM과 동일한 아키텍처를 사용하되, **PrefixLM (non-causal decoder-only)** 방식으로 학습됩니다. 즉, 입력(prefix) 부분에는 양방향 어텐션을, 타겟 부분에는 causal 어텐션을 적용합니다. 시퀀스 길이는 2048(입력 1024 + 타겟 1024)로 구성됩니다.

### 성능 향상
- **NLP 태스크 평균**: 540B 규모에서 PaLM 대비 상당한 향상, 26개 태스크 중 21개에서 SOTA 달성
- **BIG-Bench Emergent Suite**: 21개 태스크 중 19개에서 개선, navigate 태스크는 $55.3\% \to 67.0\%$ ($+21.2\%$)
- **Reasoning/CoT**: GSM8K $54.9 \to 58.5$ ($+6.6\%$), BBH $44.8 \to 49.6$ ($+10.7\%$)
- **다국어**: MGSM $+8.7\%$, TydiQA $+3.2\%$
- **MMLU**: $69.3\% \to 70.7\%$
- **Fine-tuning**: SuperGLUE 8B 스케일에서 $+3.2\%$

### 한계
- 새로운 데이터/토큰을 사용하지 않으므로 **지식(knowledge) 집약적 태스크**에서는 개선폭이 제한적 (예: logical_sequence, english_proverbs에서는 오히려 성능 저하)
- 540B 모델은 아직 수렴하지 않은 상태였기에, 절감률(2.35x)이 실제로 어디까지 확장 가능한지는 불확실
- "emergence"의 근본 원인(귀납적 편향 vs 단순 추가 학습)에 대한 이론적 설명이 부족함

## 3. 모델의 일반화 성능 향상 가능성

이 논문에서 가장 흥미로운 지점은 **일반화 성능**과 관련된 다음의 관찰들입니다.

**(1) 귀납적 편향(inductive bias)의 역할**: navigate, geometric_shapes와 같은 공간적/시각적 추론 태스크에서 U-PaLM 8B가 PaLM 540B를 능가합니다. 저자들은 이를 PrefixLM 아키텍처와 양방향 어텐션이 제공하는 귀납적 편향 덕분이라고 설명합니다. 즉, 모델 크기가 아니라 **사전학습 목적함수의 다양성**이 일반화에 기여할 수 있음을 시사합니다.

**(2) Emergent ability의 조기 발현**: crass_ai, vitaminc, identify_odd_metaphor 등의 태스크에서 PaLM은 540B에서만 emergence를 보이지만, U-PaLM은 62B에서 이미 emergence를 보입니다. 이는 스케일이 아닌 **학습 목적함수의 다양성**이 emergent ability를 유도할 수 있다는 강력한 증거입니다. 저자들은 이를 "More is different, but different can also more"로 표현합니다.

**(3) Fine-tuning 일반화 개선**: SuperGLUE, TydiQA fine-tuning에서 특히 8B 규모에서 큰 개선을 보였는데, 이는 PaLM이 causal LM만으로 학습되어 fine-tuning 시 상대적으로 약점(예: T5-large보다 낮은 성능)이 있었음을 UL2R이 보완한 것으로 해석됩니다. 이는 사전학습 목적함수의 다양성이 다운스트림 전이 학습의 일반화 능력을 근본적으로 향상시킬 수 있음을 보여줍니다.

**(4) 새로운 프롬프팅 능력의 일반화**: Infilling, mode token을 통한 특정 사전학습 지식 접근 등은 기존 causal LM으로는 불가능했던 방식으로 문제를 해결할 수 있는 능력을 부여합니다. 이는 모델이 여러 방식으로 동일 문제에 접근할 수 있게 하여 실질적인 태스크 범용성(generalization in usage)을 높입니다.

## 4. 향후 연구에 미치는 영향과 고려할 점

### 연구에 미치는 영향

1. **"Continued pretraining" 패러다임의 확립**: 이 논문 이후 대규모 모델을 처음부터 재학습하지 않고 추가 목적함수로 짧게 재학습시키는 방식이 표준적 연구 방향으로 자리잡았습니다. 후속 연구인 FLAN(Chung et al., 2022)과의 결합 실험은 이 방향의 확장성을 보여줍니다.

2. **스케일링 법칙 재정의**: 기존 Kaplan et al. (2020), Hoffmann et al. (2022)의 스케일링 법칙은 모델 크기/데이터량에만 초점을 맞췄으나, 이 논문은 **"어떤 목적함수로 학습하는가"**도 스케일링 곡선의 중요한 축임을 보여주며 스케일링 연구의 차원을 확장했습니다.

3. **Emergent ability 연구에 대한 새로운 해석 제공**: Wei et al. (2022a)의 emergent ability를 순수 스케일의 함수로 보던 관점에, "훈련 목적함수의 다양성"이라는 새로운 변수를 추가했습니다.

### 향후 연구 시 고려할 점

1. **최적 믹스처 비율의 이론적 규명 필요**: 논문은 50/25/25 비율을 경험적으로 선택했을 뿐, 왜 이 비율이 최적인지에 대한 이론적 근거는 부족합니다. 후속 연구에서는 이 비율과 모델 크기, 태스크 종류 간의 관계를 체계적으로 규명할 필요가 있습니다.

2. **Continued training의 "커리큘럼" 효과 검증**: 저자들 스스로도 "UL2R을 처음부터 적용했다면 어땠을까"라는 질문에 명확한 답을 하지 못했습니다(Appendix 7.3.1). 사전학습 태스크의 순서(curriculum)가 최종 성능에 미치는 영향에 대한 후속 연구가 필요합니다.

3. **지식 집약적 태스크의 한계 보완**: UL2R은 새로운 데이터를 사용하지 않으므로 지식(factual knowledge) 향상에는 한계가 있습니다. 데이터 증강이나 retrieval augmentation과의 결합이 향후 연구 방향이 될 수 있습니다.

4. **다른 아키텍처와 모델 계열로의 일반화 검증**: 이 논문은 PaLM에 국한된 실험입니다. GPT 계열, LLaMA 계열 등 다른 decoder-only 모델에도 동일한 효과가 나타나는지 검증이 필요합니다.

5. **Instruction tuning과의 결합**: 저자들이 언급했듯이, UL2R과 FLAN(instruction tuning)의 결합 효과에 대한 심층 분석이 후속 연구(Chung et al., 2022, FLAN-PaLM)에서 다뤄졌으며, 이러한 "복합 적응(compound adaptation)" 방법론이 향후 표준이 될 가능성이 있습니다.

---

## 참고문헌 (논문 내 인용 전체 목록)

1. Tay, Y., Wei, J., Chung, H. W., Tran, V. Q., So, D. R., Shakeri, S., ... & Dehghani, M. (2022). **Transcending Scaling Laws with 0.1% Extra Compute**. arXiv:2210.11399
2. Kaplan, J., et al. (2020). Scaling laws for neural language models. arXiv:2001.08361
3. Hoffmann, J., et al. (2022). Training compute-optimal large language models. arXiv:2203.15556
4. Chowdhery, A., et al. (2022). PaLM: Scaling language modeling with Pathways. arXiv:2204.02311
5. Tay, Y., et al. (2022b). Unifying language learning paradigms (UL2). arXiv:2205.05131
6. Wei, J., et al. (2022a). Emergent abilities of large language models. TMLR
7. Wei, J., et al. (2022b). Chain of thought prompting elicits reasoning in large language models. NeurIPS
8. Chung, H. W., et al. (2022). Scaling instruction-finetuned language models (FLAN-PaLM). arXiv preprint
9. Raffel, C., et al. (2019). Exploring the limits of transfer learning with a unified text-to-text transformer (T5). arXiv:1910.10683
10. Brown, T. B., et al. (2020). Language models are few-shot learners (GPT-3). arXiv:2005.14165
11. Rae, J. W., et al. (2021). Scaling language models: Methods, analysis & insights from training Gopher. arXiv:2112.11446
12. Suzgun, M., et al. (2022). Challenging big-bench tasks and whether chain-of-thought can solve them (BBH). arXiv preprint
13. Srivastava, A., et al. (2022). Beyond the imitation game: Quantifying and extrapolating the capabilities of language models (BIG-Bench). arXiv:2206.04615
14. Lewkowycz, A., et al. (2022). Solving quantitative reasoning problems with language models (Minerva). arXiv:2206.14858
15. Hendrycks, D., et al. (2020). Measuring massive multitask language understanding (MMLU). arXiv:2009.03300
16. Shi, F., et al. (2022). Language models are multilingual chain-of-thought reasoners (MGSM). arXiv:2210.03057
17. Cobbe, K., et al. (2021). Training verifiers to solve math word problems (GSM8K). arXiv:2110.14168

**주의**: 위 답변은 제공된 arXiv:2210.11399v2 PDF 문서 내용을 기반으로 작성되었으며, "2020년 이후 관련 최신 연구 비교 분석"과 관련해서는 문서 내에 언급된 참고문헌 정보만을 사용했습니다. 문서에 명시되지 않은 외부 정보는 포함하지 않았습니다.
