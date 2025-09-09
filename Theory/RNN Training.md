# RNN Training(RNN and BPTT)

요청하신 두 글을 바탕으로, RNN의 순전파와 BPTT(시간을 통한 역전파)를 한눈에 이해하고, 연구 수준의 예제 코드까지 그대로 실습할 수 있는 블로그 글 형태로 정리했습니다. 핵심은 한 문장으로 요약하면, 기본 RNN은 동일한 가중치를 시간축으로 공유하면서 순전파를 수행하고, 손실을 시간 합으로 정의한 뒤 체인룰로 모든 가중치의 기울기를 누적해 업데이트하는 구조입니다.[1][2][3][4][5]

## 개요
- 본 글은 RNN 단일 셀의 순전파 수식, 시퀀스 전개(Through Time), 교차 엔트로피+소프트맥스 손실, 그리고 BPTT의 기울기 유도까지 단계적으로 설명합니다.[2][3][1]
- 마지막에 NumPy로 구현한 미니멀 RNN(BPTT 포함)과 PyTorch 학습 루프를 제공합니다. 실험 재현과 연구 확장에 바로 활용 가능합니다.[3][4]

## 표기와 기본 구성
- 입력 시퀀스: $$x_1,\dots,x_T$$. 은닉 상태: $$a_t \in \mathbb{R}^{H}$$, 초기 상태 $$a_0$$는 0 또는 학습 파라미터로 둡니다.[1][3]
- 파라미터: $$W_{xh}\in\mathbb{R}^{H\times D}, W_{ah}\in\mathbb{R}^{H\times H}, b_h\in\mathbb{R}^{H}, W_{ao}\in\mathbb{R}^{C\times H}, b_o\in\mathbb{R}^{C}$$로 두고, 모든 시간 $$t$$에서 공유합니다.[2][3][1]

## 순전파: 단일 셀
- pre-activation: $$h_t = W_{xh}x_t + W_{ah}a_{t-1} + b_h$$ 입니다.[3][1]
- 은닉 상태: $$a_t = \tanh(h_t)$$ 또는 ReLU를 사용할 수 있으나, 여기서는 $$\tanh$$를 채택합니다.[1][3]
- 로짓: $$o_t = W_{ao}a_t + b_o$$, 출력 확률: $$\hat{y}_t = \mathrm{softmax}(o_t)$$ 입니다.[3][1]

## 순전파: 시간 전개
- 동일 파라미터를 모든 $$t$$에서 공유하며, $$t=1$$부터 $$T$$까지 위 연산을 반복합니다.[1][3]
- 손실은 시퀀스 합: $$L = \sum_{t=1}^{T} \ell_t$$, $$\ell_t = -\sum_{c=1}^{C} y_{t,c}\log \hat{y}_{t,c}$$ (크로스 엔트로피)로 둡니다.[4][3]

## 소프트맥스+크로스 엔트로피의 핵심 미분
- 소프트맥스와 크로스 엔트로피를 결합하면 출력 로짓 $$o_t$$에 대한 그라디언트는 $$\frac{\partial \ell_t}{\partial o_t} = \hat{y}_t - y_t$$로 단순화됩니다.[5][4]
- 이 결과는 BPTT의 출발점이 되며, 수치적으로 안정적이고 구현이 간단합니다.[4][5]

## BPTT(Back Propagation Through Time): 기울기 흐름
- 출력층: $$\frac{\partial \ell}{\partial W_{ao}} = \sum\_t (\hat{y}\_t - y_t)a_t^\top$$ , $$\frac{\partial \ell}{\partial b_o} = \sum_t (\hat{y}_t - y_t)$$ 입니다.[2][4][3]
- 은닉으로의 전파:

```math
\delta^{(a)}_t = W_{ao}^\top(\hat{y}_t - y_t) + W_{ah}^\top \delta^{(h)}_{t+1}
```
로 누적됩니다. 여기서 $$\delta^{(h)}_t = \delta^{(a)}_t \odot (1 - a_t^{\odot 2})$$는 $$\tanh$$의 도함수까지 포함한 은닉 전-활성 기울기입니다.[2][3]

- 은닉층 파라미터: $$\frac{\partial \ell}{\partial W_{xh}} = \sum_t \delta^{(h)}\_t x_t^\top$$, $$\frac{\partial \ell}{\partial W\_{ah}} = \sum_t \delta^{(h)}\_t a_{t-1}^\top$$, $$\frac{\partial \ell}{\partial b_h} = \sum_t \delta^{(h)}_t$$ 입니다.[3][2]

## BPTT: 직관과 팁
- 시간 공유 파라미터 때문에 각 시간의 그라디언트를 모두 합산해야 합니다. 이것이 “Through Time”의 핵심입니다.[2][3]
- 기울기 소실/폭주의 원인은 반복 곱(예: $$W_{ah}^\top$$)과 비선형 도함수 누적입니다. 클리핑, 정규화, 게이트(RNN→LSTM/GRU) 등으로 완화합니다.[3]

## 연구 수준 미니멀 NumPy 구현
- 아래는 소프트맥스+크로스 엔트로피, 전체 BPTT, 그라디언트 클리핑 포함 RNN을 NumPy로 구현한 예시입니다.[4][3]

```python
import numpy as np

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    expz = np.exp(z)
    return expz / np.sum(expz, axis=1, keepdims=True)

def one_hot(y, C):
    oh = np.zeros((len(y), C))
    oh[np.arange(len(y)), y] = 1.0
    return oh

class RNN:
    def __init__(self, D, H, C, seed=42):
        rng = np.random.default_rng(seed)
        self.Wxh = rng.normal(0, 0.1, (H, D))
        self.Wah = rng.normal(0, 0.1, (H, H))
        self.bh  = np.zeros((H,))
        self.Wao = rng.normal(0, 0.1, (C, H))
        self.bo  = np.zeros((C,))
        self.H = H

    def forward(self, X):
        # X: list of (N,D) for t=1..T
        N = X.shape
        a = [np.zeros((N, self.H))]  # a_0
        h_list, o_list, yhat_list = [], [], []
        for t in range(len(X)):
            h_t = X[t] @ self.Wxh.T + a[-1] @ self.Wah.T + self.bh
            a_t = np.tanh(h_t)
            o_t = a_t @ self.Wao.T + self.bo
            yhat_t = softmax(o_t)
            h_list.append(h_t)
            a.append(a_t)
            o_list.append(o_t)
            yhat_list.append(yhat_t)
        return h_list, a, o_list, yhat_list

    def loss(self, yhat_list, Y_list):
        # Y_list: list of (N,C) one-hot
        loss = 0.0
        for yhat_t, Y_t in zip(yhat_list, Y_list):
            loss += -np.sum(Y_t * np.log(yhat_t + 1e-12)) / yhat_t.shape
        return loss

    def bptt(self, X, Y_list, clip=1.0):
        # Forward
        h_list, a, o_list, yhat_list = self.forward(X)
        N = X.shape
        T = len(X)

        # Grad init
        gWxh = np.zeros_like(self.Wxh)
        gWah = np.zeros_like(self.Wah)
        gbh  = np.zeros_like(self.bh)
        gWao = np.zeros_like(self.Wao)
        gbo  = np.zeros_like(self.bo)

        # Backward through time
        delta_a_next = np.zeros((N, self.H))
        for t in reversed(range(T)):
            yhat_t = yhat_list[t]
            Y_t = Y_list[t]
            a_t = a[t+1]
            a_prev = a[t]
            h_t = h_list[t]

            # dL/do_t
            d_o = (yhat_t - Y_t) / N  # CE + softmax
            # output layer grads
            gWao += d_o.T @ a_t
            gbo  += d_o.sum(axis=0)

            # back to hidden activation
            delta_a = d_o @ self.Wao + delta_a_next @ self.Wah
            # tanh derivative
            d_h = delta_a * (1.0 - np.tanh(h_t)**2)

            # hidden grads
            gWxh += d_h.T @ X[t]
            gWah += d_h.T @ a_prev
            gbh  += d_h.sum(axis=0)

            # pass to previous time (via a_{t-1})
            delta_a_next = d_h

        # gradient clipping
        for g in [gWxh, gWah, gbh, gWao, gbo]:
            np.clip(g, -clip, clip, out=g)

        return (gWxh, gWah, gbh, gWao, gbo), (h_list, a, o_list, yhat_list)

    def step(self, grads, lr=1e-2):
        gWxh, gWah, gbh, gWao, gbo = grads
        self.Wxh -= lr * gWxh
        self.Wah -= lr * gWah
        self.bh  -= lr * gbh
        self.Wao -= lr * gWao
        self.bo  -= lr * gbo

# toy usage
D, H, C, T, N = 16, 32, 5, 8, 4
rng = np.random.default_rng(0)
X = [rng.normal(size=(N, D)).astype(np.float32) for _ in range(T)]
y_seq = [rng.integers(0, C, size=(N,)) for _ in range(T)]
Y = [one_hot(y, C) for y in y_seq]

rnn = RNN(D, H, C)
for epoch in range(50):
    grads, _ = rnn.bptt(X, Y, clip=1.0)
    rnn.step(grads, lr=5e-2)
```


## PyTorch 구현과 학습 루프
- 실전에서는 자동미분을 활용해 구현 복잡도를 낮춥니다. 아래 코드는 시간 전개와 교차 엔트로피를 사용하며, teacher-forcing 없이 각 시점 분류를 수행합니다.[4][3]

```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, D, H, C):
        super().__init__()
        self.rnn = nn.RNN(input_size=D, hidden_size=H, nonlinearity='tanh', batch_first=True)
        self.classifier = nn.Linear(H, C)

    def forward(self, x, h0=None):
        # x: (N, T, D)
        out, hT = self.rnn(x, h0)      # out: (N, T, H)
        logits = self.classifier(out)  # (N, T, C)
        return logits, hT

N, T, D, H, C = 32, 20, 64, 128, 10
model = SimpleRNN(D, H, C)
criterion = nn.CrossEntropyLoss()  # per-step classification

opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(1000):
    x = torch.randn(N, T, D)
    y = torch.randint(0, C, (N, T))

    logits, _ = model(x)
    loss = criterion(logits.reshape(-1, C), y.reshape(-1))

    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
```


## 수식 요약(핵심 블록)
- 순전파  
  - $$h_t = W_{xh}x_t + W_{ah}a_{t-1} + b_h$$, $$a_t=\tanh(h_t)$$, $$o_t=W_{ao}a_t+b_o$$, $$\hat{y}_t=\mathrm{softmax}(o_t)$$ 입니다.[1][3]
- 손실  
  - $$L=\sum_{t=1}^{T}-\sum_{c=1}^{C} y_{t,c}\log \hat{y}_{t,c}$$ 입니다.[4][3]
- 역전파(대표식)  
  - $$\frac{\partial \ell}{\partial o_t}=\hat{y}\_t-y_t$$, $$\frac{\partial \ell}{\partial W_{ao}}=\sum_t(\hat{y}_t-y_t)a_t^\top$$, $$\frac{\partial \ell}{\partial b_o}=\sum_t(\hat{y}_t-y_t)$$ 입니다.[5][4]
  - $$\delta^{(a)}\_t = W_{ao}^\top(\hat{y}\_t-y_t) + W_{ah}^\top \delta^{(h)}_{t+1}$$, $$\delta^{(h)}_t = \delta^{(a)}_t \odot (1-a_t^{\odot 2})$$ 입니다.[2][3]
  - $$\frac{\partial \ell}{\partial W_{xh}}=\sum_t \delta^{(h)}\_t x_t^\top$$, $$\frac{\partial \ell}{\partial W_{ah}}=\sum_t \delta^{(h)}\_t a_{t-1}^\top$$, $$\frac{\partial \ell}{\partial b_h}=\sum_t \delta^{(h)}_t$$ 입니다.[2][3]

## 구현 체크리스트
- 초기화: $$W_{ah}$$는 직교 초기화가 안정적입니다. 입력 스케일과 학습률을 함께 조정하세요.[3]
- 클리핑: 폭주 방지를 위해 $$\ell_2$$ 클리핑을 기본 적용합니다. PyTorch의 clip_grad_norm_를 사용합니다.[3]
- 마스킹: 가변 길이 시퀀스는 패딩 마스크로 손실을 무효화합니다. 배치 학습 안정성에 중요합니다.[3]

## 확장 포인트
- 장기 의존성: 기본 RNN은 기울기 소실에 취약합니다. LSTM/GRU, 잔차 연결, 스킵 연결, 정규화 등을 고려하세요.[3]
- 대안 손실: 언어 모델링은 토큰 시프트된 시퀀스에 대해 토큰별 크로스 엔트로피를 사용합니다. 분류는 시퀀스 마지막 시점 로짓만 사용해도 됩니다.[4][3]

## 결론
- 핵심은 동일 가중치 공유 하의 시간 전개, 소프트맥스-크로스 엔트로피의 단순한 출력 기울기, 그리고 체인룰로 시간 역방향 누적을 수행하는 **BPTT**입니다.[1][2][4][3]
- 제공한 NumPy/PyTorch 예제를 그대로 실행하면, 이론-구현 간 연결고리를 빠르게 체득할 수 있습니다.[4][3]

참고 원문  
- “RNN Training: Welcome to your Tape - Side A/B”의 순전파·BPTT 단계 서술을 토대로 본문 수식과 파이프라인을 정리했습니다.[1][2]
- 교차 엔트로피+소프트맥스의 기울기 단순화는 표준 유도 결과를 참조했습니다.[5][4]
- 추가로, Denny Britz의 튜토리얼은 수식·개념 정리에 도움이 됩니다.[3]

[1](https://medium.com/learn-love-ai/step-by-step-walkthrough-of-rnn-training-part-i-7aee5672dea3)
[2](https://arxiv.org/pdf/1706.02480.pdf)
[3](https://dennybritz.com/posts/wildml/recurrent-neural-networks-tutorial-part-3/)
[4](https://www.mldawn.com/back-propagation-with-cross-entropy-and-softmax/)
[5](https://wikidocs.net/235711)
[6](http://arxiv.org/pdf/1608.05343.pdf)
[7](https://arxiv.org/pdf/2307.04205.pdf)
[8](http://arxiv.org/pdf/2404.08573.pdf)
[9](http://arxiv.org/pdf/2405.12443.pdf)
[10](http://arxiv.org/pdf/2501.08040.pdf)
[11](https://arxiv.org/pdf/2311.18130.pdf)
[12](http://arxiv.org/pdf/2302.05440.pdf)
[13](https://arxiv.org/pdf/2312.09391.pdf)
[14](https://arxiv.org/pdf/2106.08318.pdf)
[15](https://arxiv.org/pdf/2103.09935.pdf)
[16](https://arxiv.org/pdf/1907.02649.pdf)
[17](https://arxiv.org/pdf/2411.04036.pdf)
[18](https://arxiv.org/pdf/1606.04671.pdf)
[19](http://arxiv.org/pdf/1909.01311.pdf)
[20](https://arxiv.org/pdf/2212.13345.pdf)
[21](http://arxiv.org/pdf/2206.07340.pdf)
[22](https://arxiv.org/pdf/2405.08967.pdf)
[23](http://arxiv.org/pdf/2205.10356.pdf)
[24](http://arxiv.org/pdf/2309.01775.pdf)
[25](https://www.youtube.com/watch?v=u8utlK_c5C8)
[26](https://www.youtube.com/watch?v=BjWqCcbusMM)
[27](https://www.youtube.com/watch?v=b0Yhl1KVHRI)
[28](https://www.youtube.com/watch?v=CQvpgLTmBhE)
[29](https://www.youtube.com/watch?v=9VF-PvW5ZqI)
[30](https://hess.copernicus.org/articles/26/5163/2022/)
[31](https://www.iitp.ac.in/~shad.pcs15/data/rnn-shad.pdf)
[32](https://www.nature.com/articles/s41598-023-49579-z)
[33](https://5ly.co/blog/recurrent-neural-networks-training-mechanics/)
[34](https://oa.upm.es/66259/1/TFG_MAXIMO_GARCIA_MARTINEZ.pdf)
[35](https://stackoverflow.com/questions/27089932/cross-entropy-softmax-and-the-derivative-term-in-backpropagation)
[36](https://www.youtube.com/watch?v=MqgZ-RyZuVo)
[37](https://iris.polito.it/retrieve/3b7c07ea-d14b-4d5d-9759-f71b3e415ea1/conv_phd_thesis_simone.pdf)
[38](https://coconote.app/notes/57ba346a-563b-41ab-bc62-04306d444e89/transcript)
[39](https://www.sciencedirect.com/science/article/pii/S0009250924000459/pdf)
[40](https://www.cs.cmu.edu/~mgormley/courses/10601-s23/slides/lecture12-backprop-ink.pdf)
[41](https://wikidocs.net/230764)
