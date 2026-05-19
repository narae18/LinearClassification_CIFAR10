import numpy as np
import pickle
import os
import matplotlib.pyplot as plt


# ────────────────────────────────────────────────
# CIFAR-10 데이터 로드
# ────────────────────────────────────────────────

def 배치_로드(파일경로):
    with open(파일경로, 'rb') as f:
        배치 = pickle.load(f, encoding='bytes')
    X = 배치[b'data'].astype(np.float64)       # (10000, 3072)
    y = np.array(배치[b'labels'])               # (10000,)
    return X, y


def CIFAR10_로드(데이터_폴더):
    """
    CIFAR-10 데이터셋 로드
    https://www.cs.toronto.edu/~kriz/cifar.html 에서 다운로드 후
    cifar-10-batches-py 폴더를 data/ 하위에 위치시키세요.
    """
    X_목록, y_목록 = [], []
    for i in range(1, 6):
        경로 = os.path.join(데이터_폴더, f'data_batch_{i}')
        X, y = 배치_로드(경로)
        X_목록.append(X)
        y_목록.append(y)

    X_훈련 = np.concatenate(X_목록)   # (50000, 3072)
    y_훈련 = np.concatenate(y_목록)   # (50000,)

    X_테스트, y_테스트 = 배치_로드(os.path.join(데이터_폴더, 'test_batch'))

    return X_훈련, y_훈련, X_테스트, y_테스트


def 전처리(X_훈련, X_테스트):
    # 픽셀값 평균 빼기 (zero-centering)
    평균 = X_훈련.mean(axis=0)
    X_훈련 = X_훈련 - 평균
    X_테스트 = X_테스트 - 평균

    # bias trick: 마지막 열에 1 추가 → (N, 3073)
    X_훈련 = np.hstack([X_훈련, np.ones((X_훈련.shape[0], 1))])
    X_테스트 = np.hstack([X_테스트, np.ones((X_테스트.shape[0], 1))])

    return X_훈련, X_테스트


# ────────────────────────────────────────────────
# SVM (Hinge Loss)
# ────────────────────────────────────────────────

def svm_손실(W, X, y, reg):
    """
    멀티클래스 SVM 손실 + L2 정규화
    W: (D, C)  X: (N, D)  y: (N,)
    반환: 손실값, 기울기
    """
    N = X.shape[0]
    점수 = X @ W                                    # (N, C)
    정답_점수 = 점수[np.arange(N), y].reshape(-1, 1)  # (N, 1)
    마진 = np.maximum(0, 점수 - 정답_점수 + 1)         # delta=1
    마진[np.arange(N), y] = 0                        # 정답 클래스 제외

    손실 = 마진.sum() / N + reg * np.sum(W * W)

    # 기울기 계산
    마스크 = (마진 > 0).astype(float)
    마스크[np.arange(N), y] = -마스크.sum(axis=1)
    dW = X.T @ 마스크 / N + 2 * reg * W

    return 손실, dW


# ────────────────────────────────────────────────
# Softmax (Cross-Entropy Loss)
# ────────────────────────────────────────────────

def softmax_손실(W, X, y, reg):
    """
    Softmax Cross-Entropy 손실 + L2 정규화
    W: (D, C)  X: (N, D)  y: (N,)
    반환: 손실값, 기울기
    """
    N = X.shape[0]
    점수 = X @ W                                    # (N, C)
    점수 -= 점수.max(axis=1, keepdims=True)          # 수치 안정성

    exp_점수 = np.exp(점수)
    확률 = exp_점수 / exp_점수.sum(axis=1, keepdims=True)  # (N, C)

    손실 = -np.log(확률[np.arange(N), y]).sum() / N + reg * np.sum(W * W)

    # 기울기 계산
    dP = 확률.copy()
    dP[np.arange(N), y] -= 1
    dW = X.T @ dP / N + 2 * reg * W

    return 손실, dW


# ────────────────────────────────────────────────
# SGD 학습기
# ────────────────────────────────────────────────

def 학습(손실함수, X, y, 학습률=1e-3, reg=1e-5,
         에폭=200, 배치크기=256, 출력간격=20):
    N, D = X.shape
    C = y.max() + 1
    W = 0.001 * np.random.randn(D, C)

    손실_기록 = []

    for 에폭번호 in range(1, 에폭 + 1):
        # 미니배치 샘플링
        인덱스 = np.random.choice(N, 배치크기, replace=False)
        X_배치 = X[인덱스]
        y_배치 = y[인덱스]

        손실값, dW = 손실함수(W, X_배치, y_배치, reg)
        W -= 학습률 * dW
        손실_기록.append(손실값)

        if 에폭번호 % 출력간격 == 0:
            print(f'  에폭 {에폭번호:>4d} | 손실: {손실값:.4f}')

    return W, 손실_기록


def 정확도(W, X, y):
    예측 = (X @ W).argmax(axis=1)
    return (예측 == y).mean()


# ────────────────────────────────────────────────
# 시각화
# ────────────────────────────────────────────────

def 손실_그래프(svm_기록, softmax_기록):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(svm_기록)
    plt.title('SVM Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(softmax_기록, color='orange')
    plt.title('Softmax Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig('loss_curve.png', dpi=100)
    plt.show()
    print('Loss curve saved: loss_curve.png')


def 가중치_시각화(W_svm, W_softmax):
    클래스명 = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

    fig, axes = plt.subplots(2, 10, figsize=(15, 4))
    for i in range(10):
        for 행, W, 제목 in zip([0, 1], [W_svm, W_softmax], ['SVM', 'Softmax']):
            템플릿 = W[:-1, i].reshape(3, 32, 32).transpose(1, 2, 0)
            # 시각화를 위해 [0, 255] 범위로 정규화
            템플릿 -= 템플릿.min()
            최대값 = 템플릿.max()
            if 최대값 > 0:
                템플릿 /= 최대값
            axes[행, i].imshow(템플릿)
            axes[행, i].axis('off')
            if 행 == 0:
                axes[행, i].set_title(클래스명[i], fontsize=8)
        axes[0, 0].set_ylabel('SVM', fontsize=9)
        axes[1, 0].set_ylabel('Softmax', fontsize=9)

    plt.suptitle('Learned Weight Templates')
    plt.tight_layout()
    plt.savefig('weight_templates.png', dpi=100)
    plt.show()
    print('Weight templates saved: weight_templates.png')


# ────────────────────────────────────────────────
# 메인 실행
# ────────────────────────────────────────────────

if __name__ == '__main__':
    # CIFAR-10 경로 설정 (다운로드 후 경로 수정)
    데이터_경로 = './data/cifar-10-batches-py'

    print('=== CIFAR-10 데이터 로드 ===')
    X_훈련, y_훈련, X_테스트, y_테스트 = CIFAR10_로드(데이터_경로)
    print(f'훈련: {X_훈련.shape}, 테스트: {X_테스트.shape}')

    # 소규모로 빠르게 실험하려면 아래 주석 해제
    # X_훈련, y_훈련 = X_훈련[:5000], y_훈련[:5000]

    X_훈련, X_테스트 = 전처리(X_훈련, X_테스트)

    # ── SVM ──────────────────────────────────────
    print('\n=== SVM (Hinge Loss) 학습 ===')
    np.random.seed(42)
    W_svm, svm_기록 = 학습(
        손실함수=svm_손실,
        X=X_훈련,
        y=y_훈련,
        학습률=5e-7,
        reg=1e-4,
        에폭=2000,
        배치크기=512
    )
    svm_훈련_정확도 = 정확도(W_svm, X_훈련, y_훈련)
    svm_테스트_정확도 = 정확도(W_svm, X_테스트, y_테스트)
    print(f'SVM  훈련 정확도: {svm_훈련_정확도:.4f}')
    print(f'SVM  테스트 정확도: {svm_테스트_정확도:.4f}')

    # ── Softmax ──────────────────────────────────
    print('\n=== Softmax (Cross-Entropy) 학습 ===')
    np.random.seed(42)
    W_softmax, softmax_기록 = 학습(
        손실함수=softmax_손실,
        X=X_훈련,
        y=y_훈련,
        학습률=1e-6,
        reg=1e-4,
        에폭=2000,
        배치크기=512
    )
    softmax_훈련_정확도 = 정확도(W_softmax, X_훈련, y_훈련)
    softmax_테스트_정확도 = 정확도(W_softmax, X_테스트, y_테스트)
    print(f'Softmax  훈련 정확도: {softmax_훈련_정확도:.4f}')
    print(f'Softmax  테스트 정확도: {softmax_테스트_정확도:.4f}')

    # ── 결과 요약 ─────────────────────────────────
    print('\n========== 최종 결과 ==========')
    print(f'SVM      테스트 정확도: {svm_테스트_정확도*100:.2f}%')
    print(f'Softmax  테스트 정확도: {softmax_테스트_정확도*100:.2f}%')

    # ── 시각화 ────────────────────────────────────
    손실_그래프(svm_기록, softmax_기록)
    가중치_시각화(W_svm, W_softmax)