# PolySketchFormer: Fast Transformers via Sketching Polynomial Kernels

## 1. 핵심 주장 및 주요 기여  
PolySketchFormer는 기존의 소프트맥스(self‐attention) 기반 트랜스포머가 가지는 $$O(n^2)$$ 시간·공간 복잡도를 완화하고, 모델 품질 저하 없이 선형 시간(attention)으로 학습할 수 있음을 보인다.  
- **고차원 다항 꼴(attention) 도입**: $$\sigma_p(x,y)=\langle x,y\rangle^p$$ (짝수 $$p\ge4$$) 형태의 다항 커널이 softmax attention과 동등한 모델 품질을 유지함을 실험적으로 검증.  
- **Sketching 기법을 이용한 선형화**: 수치 선형대수의 ‘폴리노미얼 스케칭(polynomial sketching)’을 적용하여 차수- $$p$$ 다항 attention을 차수- $$p/2$$ feature 맵 + self-tensoring으로 근사, 계산 복잡도를 $$O(n)$$으로 감소시키고 비음수성(non-negativity) 보장.  
- **블록 기반 블록 순차(prefix-sum) 알고리즘**: 인과(causal) 마스크를 효율 처리하는 하부삼각행렬 곱셈을 블록 단위 prefix sum으로 구현, 연산 의존 단계를 최소화.  
- **PolySketchFormer 구조 제안**: 위 기법들을 종합하여 대규모 문맥 길이(최대 32k) 언어 모델 학습 시 FlashAttention 대비 2× 속도 향상, 품질 저하 無를 달성.  

## 2. 해결 과제 및 제안 방법

### 2.1. 문제 정의  
- **문맥 길이 증가에 따른 자기-어텐션의 $$O(n^2)$$ 병목**  
- 기존 sparse/approximate softmax 방법은 속도 또는 품질 면에서 한계.

### 2.2. 다항 커널 Attention  
- **정규화 다항 attention**  
  
$$
    A^{(p)}\_{i,j}=\frac{\bigl\langle\,\mathrm{LN}(q_i),\,\mathrm{LN}(k_j)\bigr\rangle^p}
    {\,1+\sum_{j'}\langle\mathrm{LN}(q_i),\mathrm{LN}(k_{j'})\rangle^p\,},
  $$
 
여기서 $$\mathrm{LN}$$은 레이어 정규화, $$p\ge4$$ 짝수.  

- **실험 결과**: $$p\ge4$$인 경우 GPT-2 small 구조 기준 패럴렉시티, 생성 품질이 softmax 대비 동등함을 확인.  

### 2.3. Sketching 기반 근사  
- **정확 feature 맵 차원**: $$h^p$$에 비해 불합리하게 큼.  
- **Ahle et al. (2020) Sketch**: $$S\in\mathbb{R}^{h^{p/2}\times r}$$, $$r=O(p/\varepsilon^2)$$로 $$\langle x,y\rangle^p\approx\langle (x^{\otimes (p/2)})S,\,(y^{\otimes (p/2)})S\rangle$$를 보장.  
- **Self-tensoring**:  
  
$$
    \phi'(x)=\bigl[(x^{\otimes(p/2)})S\bigr]^{\otimes 2},\quad
    \phi'(q_i)^\top\!\phi'(k_j)\ge0,
  $$
  
$$\implies$$ 비음수 근사 보장 및 $$\ell_2$$ 오차 $$O(\varepsilon)$$.  

### 2.4. 인과 마스크 처리  
- **블록 크기 $$b$$, 블록 수 $$t=n/b$$**로 분할  
- 각 블록 간 하부삼각(prefix) 누적합과 블록 내 완전 곱셈을 교차 적용하여 전체 $$lt(A B^\top)\,C$$를 $$O(n b(m+k))$$에서 처리.  

## 3. 모델 구조  
표준 GPT-2 디코더 블록에서 softmax attention을 PolySketchAttention으로 교체.  
- **다항 차수**: $$p=4$$  
- **Sketch 크기**: $$r=32$$ 또는 64  
- **로컬 정확 다항**: 블록 내 $$(QK^\top)^p$$ 직접 계산 옵션  
- **Learnable Sketch**: 랜덤 프로젝션 → 소형 MLP(3 hidden layers, GELU, layer-norm)로 대체  

## 4. 성능 및 한계

| Method                                      | Context 32k 속도 개선 | 품질(패럴렉시티) | 한계                      |
|---------------------------------------------|-----------------------|-----------------|---------------------------|
| FlashAttention (softmax, 블록512)           | 기준                  | 동등            | $$O(n^2)$$ 메모리 한계    |
| PolySketchFormer (learned+local, $$r=32$$)  | 2×↑                   | 동등 또는 개선  | Sketch 차원 제약·튜닝 필요|
| Performer (2k features)                     | 1.1×↑                 | 약간 열화       | ℓ₂ 노름 의존성(특히 큰 벡터) |

- **한계**  
  - Sketch 크기 $$r$$과 학습 스케줄 최적화 필요  
  - 인퍼런스 KV 캐시 연산 최적화 검토 필요  

## 5. 일반화 성능 향상 가능성  
- 다항 커널의 “argmax” 특성: softmax 대비 극단 확률 집중 능력 보유 → 긴 문맥 내 희소 신호 포착에 유리  
- Local exact 적용과 학습 가능한 스케치 결합으로, 적응적 지역·글로벌 패턴 학습 가능 → 과적합 감소 및 일반화 강화 잠재력  

## 6. 향후 연구 영향 및 고려 사항  
- **영역 확장**: 트랜스포머 인코더, 비언어(비전·음성) 분야 적용  
- **인퍼런스 최적화**: KV 선형화·캐시 전략으로 추론 비용 절감  
- **하이브리드**: HyperAttention 등 sparse 기법과 결합하여 극대화  
- **Sketch 학습**: MLP 구조·정규화·학습률 스케줄링 세밀 조정으로 안정성·품질 추가 개선  

***

PolySketchFormer는 선형 시간 다항식 attention을 통해 장문맥 학습의 병목을 해소하며, 향후 트랜스포머의 대규모·장문맥 처리 연구에 중요한 토대를 제공할 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/f5abd426-e19a-4924-a343-0598416da44d/2310.01655v3.pdf
