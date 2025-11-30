# 디렉토리 처리
import os

# JSON 처리
import json

# PyTorch
import torch
from torch.utils.data import Dataset, DataLoader

# KoBERT_GRU_LM
from model import KoBERT_GRU_LM

# KoBERT 토크나이저 호출 라이브러리
from transformers import AutoTokenizer

# 하이퍼 파라미터
TEST_PATH = ""  # 테스트 데이터 경로 (수정 필요)
RESULT_PATH = "./VPDM"  # best_model.pt 저장 경로
CLASS = {"normal": 0, "phishing": 1}  # 클래스 분류
INDEX_TO_CLASS = {v: k for k, v in CLASS.items()}  # 인덱스-클래스 매핑
MODEL = "skt/kobert-base-v1"  # KoBERT 사용 모델
MAX_LENGTH = 512  # 최대 길이
HIDDEN_SIZE = 256  # 은닉 상태 차원
DROPOUT = 0.3  # 드롭아웃 비율
FREEZE = True  # 임베딩 계층 학습 여부
REMOVE_CLS = True  # CLS 토큰 제거 여부

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

                if not line.strip():
                    continue

                data = json.loads(line)  # 파싱
                text = data["text"]  # 텍스트 추출
                label = data.get("label", None)  # 레이블 추출
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


# 테스트
def test():

    device = "cuda" if torch.cuda.is_available() else "cpu"  # GPU 설정

    test_data = JsonlDataset(TEST_PATH)  # 테스트 데이터 셋 정의

    # 테스트 데이터 로드
    test_loader = DataLoader(
        test_data,
        batch_size=20,
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

    # 학습 가중치 로드
    best_model_path = os.path.join(RESULT_PATH, "best_model.pt")  # best_model.pt 경로
    state_dict = torch.load(
        best_model_path, map_location=device, weights_only=True
    )  # 저장된 가중치 로드
    model.load_state_dict(state_dict)  # 모델에 가중치 적용
    model.eval()  # PyTorch 평가 모드

    all_labels = [item["label"] for item in test_data]  # 테스트 데이터 레이블
    all_preds = []  # 예측 결과 모음

    # 예측, 기울기 계산 제외
    with torch.no_grad():

        # 배치 평가
        for batch in test_loader:
            batch = {
                k: v.to(device, non_blocking=True) for k, v in batch.items()
            }  # 배치를 디바이스(GPU)로 이동
            out = model(batch["input_ids"], batch["attention_mask"])  # 순전파 실행
            preds = torch.argmax(out["logits"], dim=-1)  # 예측 실행 (argmax 사용)
            all_preds.extend(preds.cpu().tolist())  # 예측값 기록

    # 예측 결과 출력
    print("\n===== Test Result =====")
    correct = 0  # 맞춘 횟수
    for i in range(len(all_preds)):

        pred_idx = all_preds[i]  # 예측 id
        pred_label = INDEX_TO_CLASS[pred_idx]  # 예측 이름 변환
        true_label = all_labels[i]  # 실제 레이블
        true_label_name = INDEX_TO_CLASS[true_label]  # 실제 레이블 이름 변환

        # 맞춘 횟수 기록
        if pred_idx == true_label:
            correct += 1

        print(
            f"[ID {i:02d}] TRUE: {true_label_name:8s} | PRED: {pred_label:8s}"
        )  # 결과 출력

    accuracy = correct / len(all_preds)  # 정확도 계산
    print(f"\nAccuracy: {accuracy:.4f}")  # 정확도 출력


# 실행
if __name__ == "__main__":
    test()
