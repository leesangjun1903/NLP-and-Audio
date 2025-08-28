
# Reducing Activation Recomputation in Large Transformer Models

# 핵심 요약  
**“Reducing Activation Recomputation in Large Transformer Models”** 논문은 대규모 트랜스포머 모델 학습 시 활성화값(activations) 메모리 부담을 줄이면서 재계산(recomputation)에 따른 연산 오버헤드를 최소화하는 두 가지 주요 기법을 제안한다. 

첫째, **시퀀스 병렬화(sequence parallelism)**를 도입해 비병렬화(non-parallel) 영역의 활성화 메모리를 시퀀스 차원으로 분산 저장하고, 텐서 병렬화(tensor parallelism)와 조합하여 전반적 활성화 메모리를 $$t$$ 배 저감한다. 

둘째, **선택적 활성화 재계산(selective activation recomputation)**을 통해 트랜스포머 레이어 내에서 메모리 점유가 크지만 재계산 비용이 낮은 어텐션 후반부 연산만 체크포인트(checkpoint)하고 나머지는 저장함으로써, 전체 활성화 메모리를 추가로 크게 줄이면서도 재계산 오버헤드를 90% 이상 회복한다.  

# 문제 정의  
대규모 트랜스포머 모델(수십억~수조 파라미터)은 GPU 메모리 한계로 전체 활성화를 저장할 수 없어, 전통적으로 전체 레이어를 체크포인트하고 backward 시 재계산(full activation recomputation)한다. 그러나 이 방식은 30~40%의 연산 오버헤드를 유발한다. 본 논문은  
- 모델 파라미터 및 옵티마이저 상태 외에도 **활성화 메모리**가 병목이 됨  
- **전면적 재계산**이 메모리는 절약하나 막대한 연산 낭비 초래  

# 제안 기법  
## 1. 시퀀스 병렬화 (Sequence Parallelism)  
텐서 병렬화(t-way)만 적용 시 활성화 메모리:  

$$
\text{Mem}_{\text{layer}} = sbh\Bigl(10 + \tfrac{24}{t} + \tfrac{5as}{ht}\Bigr)
$$  

그러나 레이어노름 등 비병렬화 구간의 $$10\,sbh$$는 분산되지 않음.  
시퀀스 차원 $$s$$으로 분할하고 통신합(gather/scatter)을 텐서 병렬 통신과 결합하여,  

$$
\text{Mem}_{\text{layer}} = \frac{sbh}{t}\Bigl(34 + \tfrac{5as}{h}\Bigr)
$$  

로 **메모리를 $$t$$배 절감**. 통신 대역폭은 기존 all-reduce와 동등하게 유지.  

## 2. 선택적 활성화 재계산 (Selective Activation Recomputation)  
트랜스포머 레이어 전체(잔여 항 34) 대비 어텐션 후반부 항 $$\tfrac{5as}{h}$$ 비율이 크므로,  

$$
\text{Total Mem} = \frac{34\,sbhL}{t}
$$  

로 줄이며, 재계산 오버헤드는 어텐션 후반부만 추가 수행해 **전체 연산 오버헤드를 1.6–2.7%**로 낮춤.  

# 모델 구조 및 수식 정리  
Transformer 한 레이어 활성화 메모리(무병렬)  

$$
\text{Mem}_{\text{layer}} = sbh\bigl(34 + 5a\tfrac{s}{h}\bigr)
$$  

Tensor + Sequence Parallelism + Selective Recomputation  

$$
\text{Total Mem} 
= \frac{34\,sbhL}{t}
\quad,\quad
\text{Recompute Overhead FLOPs Ratio}\approx 1 + \tfrac{s}{6h}
$$  

# 성능 평가  
| 모델 규모 | 파라미터 | GPU 수 | 메모리 절감 | 재계산 오버헤드 | 학습 속도 향상 |
|-----------|----------|--------|-------------|----------------|---------------|
| 22B       | 22억     | 64     | 5× 이하      | 4%            | 29.0%↑        |
| 175B      | 175억    | 512    | 5× 이하      | 3%            | 31.8%↑        |
| 530B      | 530억    | 280    | 5× 이하      | 2%            | 29.7%↑        |
| 1T        | 1조      | 512    | 5× 이하      | 2%            | 32.1%↑        |

# 한계 및 일반화 성능 향상 관점  
- **한계**: 시퀀스 병렬화 관련 통신 대기 시간이 모델 아키텍처나 네트워크 환경에 따라 달라질 수 있으며, 파이프라인 병렬화 상충 시 1단계 메모리 부담은 여전히 큼.  
- **일반화 성능 향상**: 학습 배치 사이즈나 시퀀스 길이를 늘릴 때 생기는 메모리 여유를 확보해, 더 큰 컨텍스트(시퀀스 길이)를 활용하거나 배치 노이즈를 줄이는 방향으로 일반화 성능 개선 연구에 기여할 수 있음.  

# 향후 연구 영향 및 고려 사항  
앞으로 대규모 언어 모델 학습에서  
- 시퀀스·텐서·파이프라인 병렬화 조합 최적화  
- 메모리 단편화(Fragmentation) 해결 및 비균일 메모리 할당 기법  
- 첫 파이프라인 스테이지 메모리 압박 완화 방안  
등을 고려하여, 더욱 효율적이고 일반화 성능이 우수한 모델 학습 전략 설계에 영향을 줄 전망이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/e8aca2f3-57ad-44ba-b4c9-0522968cac61/2205.05198v1.pdf)
