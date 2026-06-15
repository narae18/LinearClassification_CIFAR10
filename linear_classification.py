import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ============================================================
# CIFAR-10 데이터 로드 함수
# ============================================================

def load_batch(file_path):
    """
    CIFAR-10 배치 파일 하나를 불러와서 이미지와 레이블을 반환합니다.
    """
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    X = batch[b'data'].astype(np.float64)       # (10000, 3072)
    y = np.array(batch[b'labels'])               # (10000,)
    return X, y


def load_cifar10(data_dir):
    """
    CIFAR-10 전체 데이터셋(훈련 5개 배치 + 테스트 1개 배치)을 불러옵니다.
    https://www.cs.toronto.edu/~kriz/cifar.html 에서 다운로드 후
    cifar-10-batches-py 폴더를 지정된 경로에 위치시키세요.
    """
    X_train_list, y_train_list = [], []
    for i in range(1, 6):
        path = os.path.join(data_dir, f'data_batch_{i}')
        X, y = load_batch(path)
        X_train_list.append(X)
        y_train_list.append(y)

    X_train = np.concatenate(X_train_list)   # (50000, 3072)
    y_train = np.concatenate(y_train_list)   # (50000,)

    X_test, y_test = load_batch(os.path.join(data_dir, 'test_batch'))

    return X_train, y_train, X_test, y_test


def preprocess(X_train, X_test):
    """
    픽셀값을 평균으로 정규화(Zero-centering)하고, bias trick을 위해 1열을 추가합니다.
    """
    # 픽셀값 평균 빼기
    mean = X_train.mean(axis=0)
    X_train = X_train - mean
    X_test = X_test - mean

    # bias trick: 각 샘플 끝에 1 추가 -> (N, 3073)
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])

    return X_train, X_test


# ============================================================
# SVM 분류기 (Hinge Loss)
# ============================================================

def svm_loss(W, X, y, reg):
    """
    멀티클래스 SVM 손실(Hinge Loss)과 기울기(Gradient)를 L2 정규화와 함께 계산합니다.
    W: (D, C)  X: (N, D)  y: (N,)
    반환값: loss, dW
    """
    N = X.shape[0]
    scores = X @ W                                    # (N, C)
    correct_scores = scores[np.arange(N), y].reshape(-1, 1)  # (N, 1)
    margins = np.maximum(0, scores - correct_scores + 1)         # delta=1 마진 설정
    margins[np.arange(N), y] = 0                        # 정답 클래스는 손실 계산에서 제외

    loss = margins.sum() / N + reg * np.sum(W * W)

    # 기울기(경사) 계산
    mask = (margins > 0).astype(float)
    mask[np.arange(N), y] = -mask.sum(axis=1)
    dW = X.T @ mask / N + 2 * reg * W

    return loss, dW


# ============================================================
# Softmax 분류기 (Cross-Entropy Loss)
# ============================================================

def softmax_loss(W, X, y, reg):
    """
    Softmax Cross-Entropy 손실과 기울기를 L2 정규화와 함께 계산합니다.
    W: (D, C)  X: (N, D)  y: (N,)
    반환값: loss, dW
    """
    N = X.shape[0]
    scores = X @ W                                    # (N, C)
    scores -= scores.max(axis=1, keepdims=True)          # 지수 오버플로우 방지를 위한 수치 안정성 확보

    exp_scores = np.exp(scores)
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)  # (N, C) 확률 분산 변환

    loss = -np.log(probs[np.arange(N), y]).sum() / N + reg * np.sum(W * W)

    # 기울기(경사) 계산
    dP = probs.copy()
    dP[np.arange(N), y] -= 1
    dW = X.T @ dP / N + 2 * reg * W

    return loss, dW


# ============================================================
# SGD 학습기 (Stochastic Gradient Descent)
# ============================================================

def train(loss_fn, X, y, learning_rate=1e-3, reg=1e-5,
          epochs=200, batch_size=256, print_every=20):
    """
    미니배치 확률적 경사하강법(SGD)을 사용하여 분류기를 학습시킵니다.
    """
    N, D = X.shape
    C = y.max() + 1
    W = 0.001 * np.random.randn(D, C)

    loss_history = []

    for epoch in range(1, epochs + 1):
        # 무작위 미니배치 샘플링
        indices = np.random.choice(N, batch_size, replace=False)
        X_batch = X[indices]
        y_batch = y[indices]

        loss_val, dW = loss_fn(W, X_batch, y_batch, reg)
        W -= learning_rate * dW
        loss_history.append(loss_val)

        if epoch % print_every == 0:
            print(f'  에폭 {epoch:>4d} | 손실(Loss): {loss_val:.4f}')

    return W, loss_history


def get_accuracy(W, X, y):
    """
    최종 모델의 예측 분류 정확도를 계산합니다.
    """
    predictions = (X @ W).argmax(axis=1)
    return (predictions == y).mean()


def get_predictions(W, X):
    """
    혼동 행렬(Confusion Matrix) 계산을 위해 테스트 데이터의 예측 레이블을 반환합니다.
    """
    return (X @ W).argmax(axis=1)


# ============================================================
# 시각화 함수 영역
# ============================================================

def plot_loss(svm_history, softmax_history):
    """
    SVM과 Softmax의 학습 손실 곡선을 그래프로 그리고 저장합니다.
    """
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(svm_history)
    plt.title('SVM Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(softmax_history, color='orange')
    plt.title('Softmax Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig('loss_curve.png', dpi=100)
    plt.show()
    print('손실 곡선 그래프 저장 완료: loss_curve.png')


def plot_weight_templates(W_svm, W_softmax):
    """
    학습된 가중치 행렬 W를 클래스별 가중치 템플릿 이미지로 시각화하고 저장합니다.
    """
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    fig, axes = plt.subplots(2, 10, figsize=(15, 4))
    for i in range(10):
        for row, W in zip([0, 1], [W_svm, W_softmax]):
            template = W[:-1, i].reshape(3, 32, 32).transpose(1, 2, 0)
            # 시각화를 위해 픽셀값을 [0, 1] 범위로 이미지 정규화
            template -= template.min()
            max_val = template.max()
            if max_val > 0:
                template /= max_val
            axes[row, i].imshow(template)
            axes[row, i].axis('off')
            if row == 0:
                axes[row, i].set_title(class_names[i], fontsize=8)

    plt.suptitle('Learned Weight Templates')
    plt.tight_layout()
    plt.savefig('weight_templates.png', dpi=100)
    plt.show()
    print('가중치 템플릿 이미지 저장 완료: weight_templates.png')


def plot_confusion_matrices(y_true, svm_pred, softmax_pred):
    """
    두 분류기의 실제 정답과 예측값을 바탕으로 혼동 행렬(Confusion Matrix)을 시각화하고 저장합니다.
    """
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # SVM 혼동 행렬 시각화
    cm_svm = confusion_matrix(y_true, svm_pred)
    disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=class_names)
    disp_svm.plot(cmap=plt.cm.Blues, ax=axes[0], xticks_rotation=45)
    axes[0].set_title('SVM Confusion Matrix')
    
    # Softmax 혼동 행렬 시각화
    cm_softmax = confusion_matrix(y_true, softmax_pred)
    disp_softmax = ConfusionMatrixDisplay(confusion_matrix=cm_softmax, display_labels=class_names)
    disp_softmax.plot(cmap=plt.cm.Oranges, ax=axes[1], xticks_rotation=45)
    axes[1].set_title('Softmax Confusion Matrix')
    
    plt.suptitle('Confusion Matrices Comparison')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=100)
    plt.show()
    print('혼동 행렬(Confusion Matrix) 그래프 저장 완료: confusion_matrix.png')


# ============================================================
# 메인 실행부
# ============================================================

if __name__ == '__main__':
    # CIFAR-10 데이터 세트 폴더 경로 설정
    DATA_DIR = './data/cifar-10-batches-py'

    print('=== CIFAR-10 데이터셋 로드 시작 ===')
    X_train, y_train, X_test, y_test = load_cifar10(DATA_DIR)
    print(f'  훈련 데이터 세트 크기: {X_train.shape}, 테스트 데이터 세트 크기: {X_test.shape}')

    # 빠른 실험을 위해 데이터 샘플 일부만 활용하고자 하는 경우 아래 2줄의 주석을 해제하세요.
    # X_train, y_train = X_train[:5000], y_train[:5000]

    X_train, X_test = preprocess(X_train, X_test)

    # --------------------------------------------------------
    # SVM 분류기 학습 실행
    # --------------------------------------------------------
    print('\n=== SVM (Hinge Loss) 모델 학습 시작 ===')
    np.random.seed(42)
    W_svm, svm_history = train(
        loss_fn=svm_loss,
        X=X_train,
        y=y_train,
        learning_rate=5e-7,
        reg=1e-4,
        epochs=2000,
        batch_size=512
    )
    svm_train_acc = get_accuracy(W_svm, X_train, y_train)
    svm_test_acc = get_accuracy(W_svm, X_test, y_test)
    print(f'  SVM 훈련 데이터 정확도: {svm_train_acc:.4f}')
    print(f'  SVM 테스트 데이터 정확도: {svm_test_acc:.4f}')

    # --------------------------------------------------------
    # Softmax 분류기 학습 실행
    # --------------------------------------------------------
    print('\n=== Softmax (Cross-Entropy) 모델 학습 시작 ===')
    np.random.seed(42)
    W_softmax, softmax_history = train(
        loss_fn=softmax_loss,
        X=X_train,
        y=y_train,
        learning_rate=1e-6,
        reg=1e-4,
        epochs=2000,
        batch_size=512
    )
    softmax_train_acc = get_accuracy(W_softmax, X_train, y_train)
    softmax_test_acc = get_accuracy(W_softmax, X_test, y_test)
    print(f'  Softmax 훈련 데이터 정확도: {softmax_train_acc:.4f}')
    print(f'  Softmax 테스트 데이터 정확도: {softmax_test_acc:.4f}')

    # --------------------------------------------------------
    # 최종 학습 성능 비교 출력
    # --------------------------------------------------------
    print('\n========== 최종 실험 결과 요약 ==========')
    print(f'  SVM      | 최종 테스트 정확도: {svm_test_acc*100:.2f}%')
    print(f'  Softmax  | 최종 테스트 정확도: {softmax_test_acc*100:.2f}%')
    print('===========================================')

    # 혼동 행렬 데이터 가공용 오답 예측 레이블 추출
    svm_test_preds = get_predictions(W_svm, X_test)
    softmax_test_preds = get_predictions(W_softmax, X_test)

    # --------------------------------------------------------
    # 결과 그래프 시각화 및 파일 디스크 저장
    # --------------------------------------------------------
    plot_loss(svm_history, softmax_history)
    plot_weight_templates(W_svm, W_softmax)
    plot_confusion_matrices(y_test, svm_test_preds, softmax_test_preds)