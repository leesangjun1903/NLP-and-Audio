# Positional Encoding 가이드

핵심은 Transformer에서 순서 정보를 주는 방법과 실제 구현 코드입니다.[1][2][3]

## 한눈에 요약
- Transformer는 순서를 직접 다루지 않으므로, 입력 임베딩에 **positional encoding**을 더해 위치를 표현합니다.[3][4]
- 기본 방식은 사인/코사인 주파수 대역을 섞는 **sinusoidal** 방식이며, 길이 외삽에 유리합니다.[2][3]
- 실습은 PyTorch 모듈 한 개로 구현하며, NLP 1D와 비전 2D/3D 응용 모두 예시를 제공합니다.[5][6]

### Sinusoidal
Sinusoidal이란, 주기적이고 부드러운 곡선을 보이는 사인파 또는 사인곡선을 의미합니다. 이는 수학, 물리학, 공학, 신호처리 등 다양한 분야에서 흔히 나타나는 개념입니다.

이 곡선은 기본적으로 ( $y = \sin(x)$ ) 형태이며, 다음과 같은 특징을 가집니다:

주기(Period): ($2\pi$) 단위로 반복됩니다.  
진폭(Amplitude): 기본 사인함수는 1이며, 이는 중간선에서 최대/최소점까지의 거리입니다.  
영점(Zeros): ($\pi n$) (n은 정수) 위치에서 사인 값이 0이 됩니다.  
대칭성: 원점 대칭(홀함수)입니다.  
또한, 사인 함수뿐만 아니라 코사인 함수도 시상함수로서, 수평 이동(위상 이동)으로 표현 가능합니다.

일반적인 시상함수는 다음과 같은 형태를 가집니다:

```math
[y = A \sin(B(t - C)) + D
]
```

(A): 진폭 (곡선의 높이)  
(B): 주기를 조절하며, 주기 ($= \frac{2\pi}{B}$)  
(C): 위상 이동 (수평 이동)  
(D): 수직 이동 (중간선 위치 조정)  
이러한 함수는 시간이나 공간에 따라 주기적으로 변하는 자연 및 공학 현상을 모델링하는 데 광범위하게 쓰입니다.

## 왜 위치 정보가 필요한가
Self-Attention은 순서를 직접 보존하지 않습니다. RNN처럼 시간적 연결이 없기 때문에 토큰 간 상대적/절대적 순서를 알려주지 않으면 문장 구조 이해에 문제가 생깁니다. 이를 위해 입력 임베딩에 위치 신호를 더하거나, 어텐션 스코어에 거리 기반 바이어스를 주는 방식이 사용됩니다.[7][4][3]

### Attention Mechanism with bias
어텐션 스코어에 거리 기반 바이어스를 주는 방식은 쿼리(Query)와 키(Key) 간의 기본 어텐션 스코어 계산에 위치(거리) 정보를 추가하여, 위치가 가까운 토큰에 더 높은 가중치를 주도록 하는 방법입니다.  
이는 어텐션 스코어에 거리 함수(예: 음의 거리 값 또는 거리에 따른 가중치)를 더하거나 곱하는 방식으로 구현합니다.

기본 어텐션 스코어는 쿼리와 키 간 내적 또는 유사도로 계산되며, 여기에 거리 편향(bias)을 더해주면 원거리보다 근거리 토큰에 더 집중하도록 유도할 수 있습니다. 예를 들어, 거리 d에 대해 바이어스 (b(d))를 두어:

```math
[\text{AttentionScore}(Q,K) = \frac{QK^T}{\sqrt{d_k}} + b(d)
]
```

여기서 (b(d))는 거리가 멀수록 감소하는 함수가 될 수 있습니다. 이렇게 하면 어텐션 분포를 구할 때 공간적/순서적 거리 정보를 반영해 학습 성능을 개선할 수 있습니다.

요약하면,

- 기본 스코어: $(QK^T / \sqrt{d_k})$ (쿼리-키 유사도)  
- 거리 기반 바이어스: 위치 차이(거리)를 반영하는 추가 값 (b(d))를 더함  
- 최종: 스코어에 거리 바이어스 더해서 소프트맥스 적용  

## 설계 목표
좋은 위치 표현은 다음을 만족해야 합니다.  
- 각 위치가 **유일**하게 구분됩니다.[2][3]
- 문장 길이가 달라도 거리 관계가 **일관**합니다.[7][3]
- 더 긴 길이로도 **외삽**이 됩니다.[3][2]

이 기준을 바탕으로 Transformer는 사인/코사인 주파수 기반의 고정식 인코딩을 채택했습니다.[4][3]

## 수식으로 이해하기
Transformer의 sinusoidal positional encoding은 다음과 같습니다.[4][3]

- 위치를 $$pos$$, 임베딩 차원을 $$i$$, 전체 임베딩 차원을 $$d$$라 하면,

$$
PE_{(pos,2i)}=\sin\left(\frac{pos}{10000^{\frac{2i}{d}}}\right),\quad
PE_{(pos,2i+1)}=\cos\left(\frac{pos}{10000^{\frac{2i}{d}}}\right).
$$

[3][4]

여기서 ( d )는 전체 임베딩 차원 수입니다. 이때 서로 다른 차원 ( i )는 서로 다른 주파수를 나타내며, 주파수는 지수적으로 변화합니다. 이는 Fourier 기저 함수들(다양한 주파수의 사인과 코사인 함수)의 선형 조합으로 위치 정보를 표현하는 것과 동일한 원리입니다.

즉, sinusoidal positional encoding은 임의의 위치 ( pos )를 다수의 서로 다른 주파수를 가진 사인/코사인 함수로 매핑하여 위치 정보를 주며, 이 함수들이 Fourier basis 역할을 하고 있습니다. Fourier 변환에서 임의의 신호나 함수가 주파수 성분들의 합으로 분해되는 것과 같은 원리입니다.

- 각 차원은 서로 다른 파장을 갖고, 파장은 $$2\pi$$에서 $$10000\cdot 2\pi$$까지 기하급수적으로 증가합니다.[2][3]
- 저주파 차원은 전역적 패턴(멀리 떨어진 위치), 고주파 차원은 국소적 패턴(인접 위치)을 포착합니다. 이는 **Fourier 기저**와 유사한 분해 직관을 제공합니다.[2][3]

이 구조 덕분에 내적/선형변환을 통해 상대 위치 정보가 자연스럽게 암시되며, 길이 외삽 성질이 관찰됩니다.[7][2]

### Fourier basis
**푸리에 기저(Fourier basis)**는 주기함수를 삼각함수의 무한급수로 표현하기 위해 사용하는 함수들의 집합입니다. 각 함수는 일정 구간 내에서 정의되며, 주로 정현파(sine)와 코사인 함수로 구성되어 있습니다.

푸리에 기저 함수는 기본적으로 상수 함수 ($\phi_0(t) = \frac{1}{\sqrt{2}}$)와 함께, 주기 (T)에 따라 정의된 정현파와 코사인 함수로 이루어집니다:

```math
[ \phi_{2n -1}(t) = \frac{\sin\left(\frac{2 \pi n}{T} t\right)}{\sqrt{\frac{T}{2}}}, \quad \phi_{2n}(t) = \frac{\cos\left(\frac{2 \pi n}{T} t\right)}{\sqrt{\frac{T}{2}}}.
]
```

이 함수들은 주어진 주기 구간 내에서 서로 직교(orthonormal)합니다.
푸리에 급수는 임의의 주기 함수 (s(x))를 삼각함수의 계수 합으로 표현하여 해석하거나 근사하는 강력한 도구이며, 이를 통해 신호 분석이나 미분방정식 해법 등에 활용할 수 있습니다.

## 구현: PyTorch 1D 모듈
다음 모듈은 배치 입력 텐서 (N, S, D)에 위치 임베딩을 더합니다. 튜토리얼 스타일 구현을 간결화했습니다.[6][8]

```python
import math
import torch
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model은 짝수여야 합니다.")
        position = torch.arange(max_len).unsqueeze(1)               # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )                                                           # (d_model/2,)
        pe = torch.zeros(max_len, d_model)                          # (max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)                # 짝수 차원
        pe[:, 1::2] = torch.cos(position * div_term)                # 홀수 차원
        pe = pe.unsqueeze(0)                                        # (1, max_len, d_model)
        self.register_buffer("pe", pe)                              # 학습 제외, 디바이스 이동 포함

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, S, D)
        N, S, D = x.shape
        return x + self.pe[:, :S, :D]
```
- 사용 예:  
```python
N, S, D = 8, 128, 512
x = torch.randn(N, S, D)
pe = SinusoidalPositionalEncoding(D, max_len=4096)
y = pe(x)  # (N, S, D)
```
이 구현은 CS231n/공식 튜토리얼 구현과 같은 수식으로 동작합니다. 버퍼로 등록하여 저장/로드와 디바이스 전환이 안전합니다.[8][6]

## 2D/3D 이미지용 확장
비전 모델에서는 H×W 위치(혹은 3D 볼륨)에 대해 2D/3D positional encoding을 씁니다. 구현은 각 축의 1D 인코딩을 합치거나 결합합니다. 참고 구현은 다음 저장소에서 확인할 수 있습니다.[5]
- 2D에서는 높이/너비 축 각각에 사인/코사인을 만들고 채널을 분할해 배치하거나 concat 후 투영합니다.[5]
- 3D도 동일 아이디어로 (D,H,W) 각 축에 대해 확장합니다.[5]

## Transformer에 적용하는 법
- 입력 토큰 임베딩 E에 위치 인코딩 P를 더해 $$E' = E + P$$로 사용합니다. 이후 Multi-Head Attention과 FFN이 이어집니다.[4][3]
- 학습 파라미터가 없는 고정형이라 과적합 위험이 낮고, 메모리도 효율적입니다.[3][2]

PyTorch 예시(간단한 인코더 블록 개념):
```python
class TinyTransformerEncoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, dim_ff=2048, num_layers=6, max_len=4096):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=max_len)

    def forward(self, x, src_key_padding_mask=None, attn_mask=None):
        # x: (N, S, D)
        x = self.pos(x)
        return self.encoder(x, mask=attn_mask, src_key_padding_mask=src_key_padding_mask)
```
위 구조는 교과서적 파이프라인으로, 실전에서도 동일한 연결을 사용합니다.[6][3]

## 디버깅 체크리스트
- 텐서 shape 불일치: 입력은 (N, S, D), 위치 버퍼는 (1, max_len, D)인지 확인하세요. 임베딩 차원 D가 맞지 않으면 브로드캐스팅 오류가 납니다.[8][6]
- d_model 짝수 조건: 사인/코사인 짝을 위해 짝수 권장입니다. 홀수면 마지막 차원 처리를 별도로 해야 합니다.[6][3]
- 길이 외삽: max_len은 훈련보다 큰 길이로 설정해두는 것이 안전합니다.[2][3]

## 대안: 학습형/바이어스/회전 방식
- Learnable absolute embedding: 토큰 위치별 학습 파라미터를 둡니다. 간단하지만 길이 외삽은 약합니다.[9][3]
- ALiBi: 어텐션에서 거리 비례 패널티를 가산하여 길이 외삽 성능이 강합니다. 스코어 단계에서 선형 바이어스를 추가합니다.[7]
- RoPE(회전 위치 임베딩): 쿼리/키 공간에서 회전 연산으로 상대 위치 상호작용을 내장합니다. 최근 LLM과 비전 트랜스포머에 널리 쓰입니다.[10][11]

학습/추론 길이 외삽이 중요하면 ALiBi나 RoPE를 검토하세요. 문헌은 길이 일반화, 상대위치 성질 면에서 이점을 보고합니다.[10][7]

### Learnable absolute embedding
Learnable absolute embedding은 위치 정보를 학습 가능한 벡터로 표현하는 방식으로, 각 토큰 위치 (i)에 대해 고정된 사인코사인 함수가 아니라 임베딩 행렬에서 해당 위치의 벡터를 직접 학습한다는 점이 핵심입니다.  
수식으로는 보통 위치 인덱스 (i)에 대응하는 임베딩 벡터 ($\mathbf{p}_i$)를 학습하며, 이는 토큰 임베딩 ($\mathbf{x}_i$)에 더해져 입력으로 사용됩니다:

```math
[\mathbf{z}_i = \mathbf{x}_i + \mathbf{p}_i
]
```

여기서 ($\mathbf{p}_i$)는 모델이 학습하는 파라미터로서 절대 위치 (i)에 따라 고유하게 존재합니다.

더 자세한 동작 원리로, 학습된 절대 위치 임베딩은 실험적으로 사인·코사인 함수와 유사한 주기적인 패턴을 스스로 학습하는 경향이 있으며, 이는 Transformer의 attention이 주변 토큰에 집중할 수 있도록 돕습니다.

따라서 Learnable absolute embedding 수식은 다음과 같이 요약할 수 있습니다:

- 임베딩 행렬 $(P \in \mathbb{R}^{L \times d})$ (길이 (L), 차원 (d))를 학습한다.
- 각 위치 (i < L)에 대해 $(\mathbf{p}_i = P[i])$ 를 가져온다.
- 입력 임베딩 $(\mathbf{x}_i)$ 에 더해 Transformer에 입력한다: $(\mathbf{z}_i = \mathbf{x}_i + \mathbf{p}_i)$.
이 방식은 고정 함수 대신에 위치별로 최적화된 임베딩을 학습하여 더 유연하게 위치 정보를 반영할 수 있지만, 학습 데이터 길이 이상의 위치에 대해서는 일반화가 어려울 수 있습니다.

### ALiBi(Attention With Linear Biases Enables Input Length Extrapolation)
ALiBi는 Transformer의 위치 임베딩 문제를 해결하는 기법으로, query와 key 사이의 거리에 따라 선형으로 편향(linear bias)를 더해 주는 방식입니다.  
수식적으로는 query-key dot product 결과에 각 헤드별로 미리 정해진 기울기(slope)를 곱한 선형 편향을 더하는 형태로, 이 편향은 학습되지 않고 고정되어 있습니다.

구체적으로, 각 어텐션 헤드마다 거리(두 토큰의 위치 차이)에 비례한 선형 bias (b(d))가 추가되어, 실제 어텐션 계산 시

```math
[\text{Attention score} = QK^T + b(d)
]
```

형태로 처리됩니다.  
여기서 (d)는 query와 key의 위치 차이이며, (b(d))는 고정된 기울기를 갖는 선형 함수입니다.  

```math
[b(d) = -m_h \times d
]
```

이 기울기는 보통 각 헤드마다 기하수열(geometric sequence)로 정의되어 모델이 학습된 최대 길이보다 긴 입력에 대해 확장성을 갖도록 설계되었습니다.

즉, ALiBi 수식은 별도의 위치 임베딩 없이, 거리 기반 선형 편향을 어텐션 점수에 직접 더하여 Transformer가 더 긴 시퀀스를 처리할 수 있게 만드는 핵심 아이디어입니다.

ALiBi는 거리가 멀어질수록 쿼리-키 간의 점수에 더 큰 음수 바이어스를 더해 멀리 떨어진 토큰에 대한 어텐션 값을 점진적으로 감소시키는 방식으로 작동합니다.

이 정의로 인해 모델은 학습 시 짧은 시퀀스를 사용해도, 더 긴 시퀀스에 대해 자연스럽게 어텐션 점수를 조절하며 입력 길이에 따른 외삽(extrapolation) 능력을 가질 수 있습니다.

### Rotary Position Embedding (RoPE)
Rotary Position Embedding (RoPE)는 벡터 공간에서 각 토큰의 임베딩을 위치에 따라 회전(Rotate)시켜, 절대 위치를 회전 각도로 인코딩하는 방법입니다.  
다시 말하면, Rotary Position Embedding (RoPE)은 위치 정보를 각 토큰 임베딩 벡터에 회전 행렬(rotation matrix)을 곱하여 인코딩하는 방식입니다.  
이때 위치 (m)에 대해 임베딩 벡터를 각도 ($\theta_m$)만큼 회전시켜 위치 정보를 통합합니다.

RoPE의 핵심 수식은 다음과 같습니다.

```math
[\begin{aligned}
\hat{q}_m &= R(\theta_m) q_m, \
\hat{k}_n &= R(\theta_n) k_n,
\end{aligned}
]
```

여기서

($q_m, k_n \in \mathbb{R}^d$)는 위치 (m, n)에서의 쿼리(query)와 키(key) 벡터,
($\hat{q}_m, \hat{k}_n$)는 위치 정보가 반영된 회전된 벡터,
($R(\theta) \in \mathbb{R}^{d \times d}$)는 각도 ($\theta$)만큼 회전하는 블록 대각 행렬(rotation matrix)입니다.
특히 (d)를 짝수 차원으로 나누어 2차원씩 묶어 각 쌍마다 다음과 같은 2D 회전 행렬을 사용합니다:

```math
[R(\theta) =
\begin{bmatrix}
\cos \theta & -\sin \theta \
\sin \theta & \cos \theta
\end{bmatrix}
]
```

이 행렬은 각 2차원 성분 쌍을 ($\theta$)만큼 복소평면 상에서 회전시키는 역할을 합니다.

위치 ($\theta_m$)는 주로 다음과 같이 정의합니다:

```math
[\theta_{m, 2i} = m / 10000^{2i/d}, \quad i = 0, 1, \dots, d/2 - 1,
]
```

그래서 각 임베딩 차원 쌍마다 서로 다른 주기로 위치 값을 반영하도록 설계되었습니다.

이 방식을 통해 쿼리와 키 벡터의 내적 계산에서 자연스럽게 상대 위치 정보가 반영되어 self-attention 계산에 포함됩니다. RoPE는 기존의 절대 위치 인코딩이나 가법적 위치 인코딩(additive positional encoding)과 달리 곱셈(multiplicative) 기반으로 작동하여 길이에 유연하고 상대 위치 의존성을 효과적으로 처리할 수 있습니다.



## 예제: RoPE로 바꿔보기(개념)
RoPE는 쿼리/키를 위치별 회전시킵니다. 구현은 다음 논문을 참고하여 각 헤드 차원에 복소수 회전(혹은 2D 로테이션)으로 적용합니다.[11][10]
- 장점: 상대 거리 감쇠, 길이 유연성, 선형 어텐션과도 결합 가능.[11][10]
- 실습 시에는 Hugging Face 통합 모델이나 오픈 소스 구현을 재사용하는 것이 안전합니다.[10][11]

## 학습 팁과 응용
- NLP: 토큰화→임베딩→positional encoding 덧셈→Transformer. 길이가 긴 시퀀스에서는 ALiBi/RoPE 고려.[7][3]
- 비전: 패치 임베딩 후 2D/3D positional encoding 적용. 다중 해상도에서는 보간/스케일 조정 전략이 필요합니다.[5][2]
- 분석: attention map이 거리 민감하게 반응하는지, 길이 증가 시 성능 저하 패턴을 점검하세요.[7][2]

## 마무리 포인트
- 기본으로는 **sinusoidal**이 여전히 유효한 선택입니다. 파라미터 없음, 길이 외삽 친화, 구현 용이성이 장점입니다.[3][2]
- 길이 외삽이 과제의 핵심이면 **ALiBi**나 **RoPE**를 검토하세요. 최신 결과에서 경쟁력이 확인됩니다.[10][7]

참고 문헌  
- Attention Is All You Need: 공식 수식과 설계 배경을 제공합니다.[3]
- D2L과 IBM 설명: 수식과 직관을 정리한 학습 자료입니다.[1][4]
- LearnOpenCV 글: 수식, 직관, 코드까지 종합적으로 설명합니다.[2]
- PyTorch 예제/포럼: 실전 구현과 shape 이슈를 점검할 수 있습니다.[8][6]
- ALiBi/RoPE: 길이 외삽과 상대위치 관점의 대표적 대안입니다.[10][7]
- 2D/3D 구현 레포: 비전 응용 시 구조를 참고하세요.[5]

추가로 읽을거리  
- Positional Encoding 변천사 요약과 최근 서베이: 개념 지도를 잡는 데 유용합니다.[12][13]
- 길이 일반화 영향 분석 및 최신 변형들: 설계 선택이 길이에 미치는 영향을 다룹니다.[9][7]

[1](https://www.ibm.com/think/topics/positional-encoding)
[2](https://learnopencv.com/sinusoidal-position-embeddings/)
[3](https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf)
[4](https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html)
[5](https://github.com/tatp22/multidim-positional-encoding)
[6](https://self-deeplearning.tistory.com/entry/PyTorch%EB%A1%9C-Transformer-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0)
[7](https://www.semanticscholar.org/paper/9ca329408813d209b1dcb36936f7f9cba82506bd)
[8](https://discuss.pytorch.org/t/positional-encoding/175953)
[9](https://arxiv.org/pdf/2305.19466.pdf)
[10](https://arxiv.org/abs/2104.09864)
[11](https://arxiv.org/pdf/2104.09864.pdf)
[12](https://dongkwan-kim.github.io/blogs/a-short-history-of-positional-encoding/)
[13](https://arxiv.org/html/2312.17044v4)
[14](https://skyjwoo.tistory.com/entry/positional-encoding%EC%9D%B4%EB%9E%80-%EB%AC%B4%EC%97%87%EC%9D%B8%EA%B0%80)
[15](https://www.semanticscholar.org/paper/af6a43eb3a44a96280def594c96188b790f295b9)
[16](https://www.semanticscholar.org/paper/98629ad5f76adbbdd5ca2971e0023ef33ad34fbe)
[17](http://aclweb.org/anthology/N19-1401)
[18](https://www.jstage.jst.go.jp/article/jnlp/27/2/27_445/_article)
[19](https://arxiv.org/abs/2307.01115)
[20](https://www.jstage.jst.go.jp/article/jnlp/28/2/28_682/_article/-char/ja/)
[21](http://arxiv.org/abs/1904.07418)
[22](http://arxiv.org/pdf/2407.09370.pdf)
[23](https://www.aclweb.org/anthology/N18-2074.pdf)
[24](https://arxiv.org/pdf/1803.02155.pdf)
[25](https://arxiv.org/pdf/2107.02561.pdf)
[26](https://arxiv.org/pdf/2312.16045.pdf)
[27](http://arxiv.org/pdf/2405.04585.pdf)
[28](http://arxiv.org/pdf/2405.09061.pdf)
[29](https://arxiv.org/html/2310.06743)
[30](https://arxiv.org/pdf/2204.08142.pdf)
[31](http://arxiv.org/pdf/2104.08698.pdf)
[32](https://arxiv.org/pdf/2407.20912.pdf)
[33](http://arxiv.org/pdf/2401.09686.pdf)
[34](https://tigris-data-science.tistory.com/entry/%EC%B0%A8%EA%B7%BC%EC%B0%A8%EA%B7%BC-%EC%9D%B4%ED%95%B4%ED%95%98%EB%8A%94-Transformer5-Positional-Encoding)
[35](https://openaccess.thecvf.com/content/CVPR2025/papers/Yu_ComRoPE_Scalable_and_Robust_Rotary_Position_Embedding_Parameterized_by_Trainable_CVPR_2025_paper.pdf)
[36](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/roformer/)
[37](https://code-angie.tistory.com/9)
[38](https://velog.io/@smkm1568/RoFormer-Enhanced-Transformer-with-Rotary-Position-Embedding-Paper-review)
[39](https://www.youtube.com/watch?v=LlZL1X0n1FM)
[40](https://asidefine.tistory.com/282)
[41](https://tutorials.pytorch.kr/beginner/transformer_tutorial.html)
