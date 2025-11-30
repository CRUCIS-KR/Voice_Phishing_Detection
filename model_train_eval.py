# 디렉토리 처리
import os

# JSON 처리
import json

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 평가 지표
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# KoBERT_GRU_LM
from model import KoBERT_GRU_LM

# KoBERT 토크나이저 호출 라이브러리
from transformers import AutoTokenizer

# 하이퍼 파라미터
TRAIN_PATH = ""  # 학습 데이터 경로 (수정 필요)
VAL_PATH = ""  # 평가 데이터 경로 (수정 필요)
RESULT_PATH = "./VPDM"  # 결과 저장 경로
CLASS = {"normal": 0, "phishing": 1}  # 클래스 분류
INDEX_TO_CLASS = {v: k for k, v in CLASS.items()}  # 인덱스-클래스 매핑
MODEL = "skt/kobert-base-v1"  # KoBERT 사용 모델
FREEZE = True  # 임베딩 계층 학습 여부
REMOVE_CLS = True  # CLS 토큰 제거 여부
MAX_LENGTH = 512  # 최대 길이
HIDDEN_SIZE = 256  # 은닉 상태 차원
DROPOUT = 0.3  # 드롭아웃 비율
LR = 1e-5  # 학습률
WEIGHT_DECAY = 0.02  # 가중치 감쇠
EPOCHS = 50  # 학습 횟수
BATCH_SIZE = 64  # 배치 크기
STOP = 50  # 조기 종료 기준

# 사용 KoBERT 모델의 토큰나이저 로드, fast 토크나이저 사용이 불가능하면 slow로 사용 (에러 방지)
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)


# 에러 방지용
def collate_fn(batch):
    return tokenize(batch, tokenizer, MAX_LENGTH)


# PyTorch 사용자 정의 데이터 셋
class JsonlDataset(Dataset):

    # 초기화 함수
    def __init__(self, path):  # 데이터 경로

        self.data_VP = []  # 데이터 셋
        with open(path, "r", encoding="utf-8") as f:  # .jsonl 파일 읽기

            for line in f:  # 줄 읽기

                if not line.strip():  # 빈 줄 무시
                    continue

                data = json.loads(line)  # 파싱
                text = data["text"]  # 텍스트 추출
                label = data["label"]  # 레이블 추출
                self.data_VP.append({"text": text, "label": label})  # 추가

    # 데이터 셋 개수
    def __len__(self):
        return len(self.data_VP)

    # 데이터 반환
    def __getitem__(self, idx):
        return self.data_VP[idx]


# 토크나이저 사용 함수
def tokenize(batch, tokenizer, max_length):

    texts = [b["text"] for b in batch]  # 배치에서 텍스트 데이터 추출
    labels = torch.tensor(
        [b["label"] for b in batch], dtype=torch.long
    )  # 배치에서 레이블 추출 및 텐서 변환

    token = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )  # 로드한 토크나이저에 파라미터를 적용하여 토큰화 실행, 패딩 여부 및 분할 사용 여부, 텐서로 반환

    # token_type_ids 제거 (에러 방지 및 사용 안함)
    if "token_type_ids" in token:
        del token["token_type_ids"]

    # 토큰화 및 배치 생성 (id, 마스킹, 레이블)
    return {
        "input_ids": token["input_ids"],
        "attention_mask": token["attention_mask"],
        "labels": labels,
    }


# 손실 함수 Focal Loss 정의
class FocalLoss(nn.Module):

    # 초기화 함수
    def __init__(self, alpha=1.0, gamma=2.0):  # 클래스 별 중요도, 강조 정도

        super().__init__()  # nn.Module 초기화 함수
        self.register_buffer(
            "alpha", torch.as_tensor(alpha, dtype=torch.float)
        )  # 클래스 별 알파 값
        self.gamma = float(gamma)  # 감마 값

    # 순전파
    def forward(self, logits, labels):

        log_p = F.log_softmax(logits, dim=-1)  # 로그 소프트 맥스
        p = log_p.exp()  # 소프트 맥스

        labels = labels.view(-1, 1)  # 레이블 차원 맞춤
        log_pt = log_p.gather(1, labels).squeeze(1)  # log(p_t), 로그 확률
        pt = p.gather(1, labels).squeeze(1)  # p_t, 확률

        focal = (1.0 - pt).pow(self.gamma)  # (1 - p_t)^gamma

        alpha_t = self.alpha[labels.squeeze(1)]  # 차원 맞춤
        focal = focal * alpha_t  # alpha * (1-pt)^gamma

        # Focal loss = -alpha * (1-pt)^gamma * log(pt)
        loss = -focal * log_pt

        return loss.mean()


train_normal = 12800.0  # 정상 데이터 수
train_phishing = 6400.0  # 피싱 데이터 수

counts = torch.tensor(
    [train_normal, train_phishing], dtype=torch.float
)  # 클래스 수 텐서 변환

alpha = counts.sum() / counts  # 반비례 계산
alpha = alpha / alpha.sum()  # 정규화


# 평가 함수 정의
def evaluate(model, dataloader, device, criterion):

    model.eval()  # PyTorch 평가 모드
    losses = []  # 손실률
    preds_all = []  # 예측값
    labels_all = []  # 레이블

    # 평가 (기울기 계산 제외)
    with torch.no_grad():

        # 배치 평가
        for batch in dataloader:

            batch = {
                k: v.to(device, non_blocking=True) for k, v in batch.items()
            }  # 배치를 디바이스(GPU)로 이동
            out = model(batch["input_ids"], batch["attention_mask"])  # 순전파 실행
            loss = criterion(out["logits"], batch["labels"])  # 손실 함수 계산
            out["loss"] = loss  # 손실 함수 결과 저장
            losses.append(out["loss"].item())  # 손실률 기록
            preds = torch.argmax(out["logits"], dim=-1)  # 예측 실행 (argmax 사용)
            preds_all.extend(preds.cpu().tolist())  # 예측값 기록
            labels_all.extend(batch["labels"].cpu().tolist())  # 레이블 기록

    # 평가 결과
    return {
        "loss": sum(losses) / len(losses) if losses else 0.0,  # 평균 손실률
        "accuracy": accuracy_score(labels_all, preds_all),  # 정확도
        "preds": preds_all,  # 예측
        "labels": labels_all,  # 레이블
    }


# 학습
def train():

    device = "cuda" if torch.cuda.is_available() else "cpu"  # GPU 설정

    os.makedirs(RESULT_PATH, exist_ok=True)  # 결과 저장소

    train_data = JsonlDataset(TRAIN_PATH)  # 학습 데이터 셋 정의
    val_data = JsonlDataset(VAL_PATH)  # 평가 데이터 셋 정의

    # 학습 데이터 로드, 데이터 셔플, 토크나이저 적용 (collate_fn), 핀 메모리 설정 (GPU 사용 시), 멀티 프로세싱
    train_load = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True if device == "cuda" else False,
        num_workers=6,
    )

    # 평가 데이터 로드, 데이터 셔플, 토크나이저 적용 (collate_fn), 핀 메모리 설정 (GPU 사용 시), 멀티 프로세싱
    val_load = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True if device == "cuda" else False,
        num_workers=6,
    )

    # 사용 모델 정의
    model = KoBERT_GRU_LM(
        hidden_size=HIDDEN_SIZE,
        classes=len(CLASS),
        freeze=FREEZE,
        dropout=DROPOUT,
        model=MODEL,
        remove_cls=REMOVE_CLS,
    ).to(device)

    # 옵티마이저, 스케줄러 정의
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS / 2
    )  # 에폭의 절반 주기로 설정

    criterion = FocalLoss(alpha=alpha).to(device)  # 손실 함수 정의

    best_val_loss = float("inf")  #  최저 손실
    best_state = None  # 최고 성능
    counter = 0  # 카운터

    train_losses, val_losses = [], []  # 학습 및 평가 손실률 기록
    train_accs, val_accs = [], []  # 학습 및 평가 정확도 기록

    # 학습 주기
    for epoch in range(1, EPOCHS + 1):

        model.train()  # PyTorch 학습 모드
        batch_losses = []  # 배치 별 손실률 기록
        batch_preds = []  # 배치 별 예측값 기록
        batch_labels = []  # 배치 별 레이블 기록

        # 배치 학습
        for batch in train_load:

            batch = {
                k: v.to(device, non_blocking=True) for k, v in batch.items()
            }  # 배치를 디바이스(GPU)로 이동

            optimizer.zero_grad()  # 옵티마이저 초기화
            out = model(batch["input_ids"], batch["attention_mask"])  # 순전파 실행

            logits = out["logits"]  # 순전파 결과
            loss = criterion(out["logits"], batch["labels"])  # 손실 함수 계산
            out["loss"] = loss  # 손실 함수 결과 저장
            preds = torch.argmax(logits, dim=-1)  # 예측 실행 (argmax 사용)

            loss.backward()  # 역전파 실행
            optimizer.step()  # 옵티마이저 갱신

            batch_losses.append(loss.item())  # 배치 손실률 기록
            batch_preds.extend(preds.cpu().tolist())  # 배치 예측값 기록
            batch_labels.extend(batch["labels"].cpu().tolist())  # 배치 레이블 기록

        scheduler.step()  # 스케줄러 갱신

        train_loss = sum(batch_losses) / len(batch_losses)  # 배치 손실률 평균 계산
        train_acc = accuracy_score(batch_labels, batch_preds)  # 배치 정확도 계산

        val_result = evaluate(model, val_load, device, criterion)  # 평가 실행
        val_loss = val_result["loss"]  # 평가 손실률
        val_acc = val_result["accuracy"]  # 평가 정확도

        train_losses.append(train_loss)  # 학습 손실률 기록
        val_losses.append(val_loss)  # 평가 손실률 기록
        train_accs.append(train_acc)  # 학습 정확도 기록
        val_accs.append(val_acc)  # 평가 정확도 기록

        # 학습 결과 출력 (확인 용)
        print(
            f"[Epoch {epoch:02d}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} || "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # 학습 조기 종료 판단
        if val_loss < best_val_loss:  # 개선된 경우
            best_val_loss = val_loss  # 최저 손실 갱신
            best_state = {
                k: v.cpu() for k, v in model.state_dict().items()
            }  # 최고 성능 모델 파라미터 저장
            torch.save(
                best_state, os.path.join(RESULT_PATH, "best_model.pt")
            )  # 최고 성능 모델 저장
            counter = 0  # 카운터 초기화
        else:
            counter += 1  # 카운터 증가
            if counter >= STOP:  # 조기 종료
                print("<Early stopping>")
                break  # 학습 종료

    # 최고 성능 모델 로드
    if best_state is not None:
        model.load_state_dict(best_state)

    test_result = evaluate(model, val_load, device, criterion)  # 최고 성능 모델 평가

    test_f1 = f1_score(
        test_result["labels"], test_result["preds"], average="macro"
    )  # F1 스코어 계산

    # 최종 결과 출력
    print("\n===== BEST MODEL =====")
    print(f"Test Loss: {test_result['loss']:.4f}")
    print(f"Test Acc : {test_result['accuracy']:.4f}")
    print(f"Test F1  : {test_f1:.4f}")

    # 학습 곡선 (손실률, 정확도)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_PATH, "learning_curves.png"))
    plt.close()

    # 혼동행렬
    cm = confusion_matrix(test_result["labels"], test_result["preds"])
    labels = [INDEX_TO_CLASS[i] for i in range(cm.shape[0])]
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predict")
    plt.ylabel("TRUE")
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_PATH, "confusion_matrix.png"))
    plt.close()


# 실행
if __name__ == "__main__":
    train()
