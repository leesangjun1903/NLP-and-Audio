# Ovis2.5 Technical Report

## 1. 핵심 주장과 주요 기여  
Ovis2.5는 **가변 해상도(native-resolution) 비전 인코더**와 **반성적(self-reflective) 심층(reasoning) 학습**을 결합하여, 복잡한 차트·도표와 같은 시각적으로 밀집된 콘텐츠에 대해 현존하는 오픈소스 MLLM 중 최상위 성능을 보인다.  
- Native-resolution ViT(NaViT)를 도입해 고정 크기 패치 분할로 인한 정보 손실을 제거  
- 추론 과정에 **사고·검토·수정** 단계를 포함하는 “thinking mode”를 제안  
- 멀티모달 학습을 위한 **5단계 커리큘럼**(시각·멀티모달 사전학습, 지침 조정, DPO, GRPO) 설계  
- 대규모 모델 학습 효율을 3–4× 개선하는 **데이터 패킹** 및 **하이브리드 병렬화** 인프라 구축  

## 2. 문제 정의  
기존 MLLM은  
① 고정 해상도 ViT로 인해 차트·다이어그램의 전역 구조와 세부 정보가 왜곡되고  
② 선형 체인오브쏘트(linear CoT)만 학습해 오류 검토·수정 능력이 부족하다.  

이로 인해 “시각적으로 복잡한 문제”와 “심층적 추론 문제”에서 성능 한계가 발생한다.

## 3. 제안 방법  
### 3.1 Native-Resolution ViT  
– NaViT는 입력 이미지를 패치로 나누지 않고 원본 해상도로 처리  
– RoPE(rotary position embeddings)를 모든 블록에 적용해 공간적 위치 정보 강화  

### 3.2 반성적 딥 리즈닝(Thinking Mode)  
– 모델 출력에 `<think>…</think>` 태그로 추론·검토·수정 과정을 명시적으로 학습  
– 이 과정을 활성화하면 복잡 문제에서 정확도가 상승하나 지연(latency)이 증가  

### 3.3 수식: 패턴 그리드 예시  
격자 형태 $$n \times n$$ 패턴의 이음쇠 개수  

$$
T(n) = 2n(n+1)
$$  

연속 패턴 간 추가 개수  

$$
T(n+1)-T(n)=4(n+1)
$$

### 3.4 5단계 학습 커리큘럼  
1. VET Pre-training: 시각 임베딩 테이블 학습  
2. Multimodal Pre-training: VT·VET·LLM 전체 파라미터 학습  
3. Instruction Tuning: 멀티모달 지시문 학습, 반성적 데이터 포함  
4. Direct Preference Optimization (DPO): 멀티모달 선호 학습  
5. Group Relative Policy Optimization (GRPO): 강화학습을 통한 심층 추론 성능 강화  

## 4. 모델 구조  
– Visual Tokenizer(VT): 이미지 패치 특징을 시각 토큰 분포로 변환  
– Visual Embedding Table(VET): 시각 단어 임베딩 테이블  
– LLM(Qwen3): 시각·텍스트 임베딩 융합 후 생성 디코더  
– NaViT와 RoPE 통합, Qwen2.5→Qwen3 업그레이드로 추론력 강화

## 5. 성능 향상  
– OpenCompass 서브-40B 부문에서 평균 78.3점으로 오픈소스 SOTA 달성  
– 차트 분석(ChartQA Pro)·OCR·수학 문제(MMMU, MathVista)·비디오 이해 벤치마크에서 모두 선두권  
– 전작 대비 평균 6.5점 이상 향상, 핵심 차트 분석에서 10%p 이상 개선  

## 6. 한계 및 일반화 성능  
– **4K급 초고해상도** 처리 및 **추론 지연** 완화 필요  
– 반성적 모드 활성화 시 연산 비용 증가  
– 특정 도메인 편향 데이터에 대한 일반화 검증 미흡  
– 향후 다양한 해상도·도메인·비디오 길이에 대한 확장성 연구 요구

## 7. 향후 영향 및 연구 고려점  
Ovis2.5는 MLLM의 시각 해상도 처리와 심층 추론을 결합한 선구적 사례로,  
- 차트·과학 논문·의료 영상 등 **고해상도 멀티모달 태스크**에 응용 가능  
- 반성적 추론(self-reflection) 학습 기법이 MLLM의 **정확도·안정성** 향상에 기여  
- 후속 연구에서는 **연산 효율**과 **도메인 일반화**를 균형 있게 달성하는 하이브리드 학습 전략, 그리고 4K 비디오·다중 단계 작업에 특화된 **경량화된 반성적 모드** 개발이 필요하다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/9021b905-33b7-4713-a58b-f899bb12a2cc/2508.11737v1.pdf)
