# pytorch
import torch
import torch.nn as nn

# 자료형 선언 라이브러리
from typing import Optional

# KoBERT 모델 호출 라이브러리
from transformers import AutoModel


# 양방향 GRU 구현
class BiGRU(nn.Module):

    # 초기화 함수
    def __init__(self, input_size, hidden_size):  # 입력 차원, 은닉 상태 차원

        super().__init__()  # nn.Module 초기화 함수

        self.hidden_size = hidden_size  # 은닉 상태 차원 설정

        # 정방향 파라미터
        self.W_x_fwd = nn.Parameter(
            torch.empty(input_size, 3 * hidden_size)
        )  # GRU 입력 가중치 초기화 (3개의 게이트)
        self.W_h_fwd = nn.Parameter(
            torch.empty(hidden_size, 3 * hidden_size)
        )  # GRU 은닉 상태 가중치 초기화 (3개의 게이트)
        self.b_fwd = nn.Parameter(
            torch.zeros(3 * hidden_size)
        )  # GRU 바이어스 초기화 (3개의 게이트)

        # 역방향 파라미터
        self.W_x_bwd = nn.Parameter(
            torch.empty(input_size, 3 * hidden_size)
        )  # GRU 입력 가중치 초기화 (3개의 게이트)
        self.W_h_bwd = nn.Parameter(
            torch.empty(hidden_size, 3 * hidden_size)
        )  # GRU 은닉 상태 가중치 초기화 (3개의 게이트)
        self.b_bwd = nn.Parameter(
            torch.zeros(3 * hidden_size)
        )  # GRU 바이어스 0으로 초기화 (3개의 게이트)
        self.reset_parameters()  # 파라미터 초깃값 설정

    # 모든 가중치를 Xavier 초깃값으로 설정
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_x_fwd)
        nn.init.xavier_uniform_(self.W_h_fwd)
        nn.init.xavier_uniform_(self.W_x_bwd)
        nn.init.xavier_uniform_(self.W_h_bwd)

    # GRU Base
    def GRU(
        self,
        x,
        mask: Optional[torch.Tensor],
        W_x,
        W_h,
        b,
    ):  # 입력 텐서, 패딩 여부, 가중치, 은닉 상태, 바이어스 -> 출력 타입 = 텐서

        N, T, _ = x.size()  # 배치 크기, 시퀀스 길이 정의
        H = self.hidden_size  # 모델 은닉 상태 크기 정의

        x_matmul = x @ W_x  # 입력 x에 대한 가중치 사전 matmul 계산
        x_z, x_r, x_g = x_matmul.chunk(3, dim=-1)  # 게이트 별 matmul 결과 저장
        b_z, b_r, b_g = b.chunk(3, dim=-1)  # 게이트 별 바이어스 분할

        # 은닉 상태 정의
        h_states = x.new_empty(N, T, H)  # 현재 은닉 상태
        h_prev = x.new_zeros(N, H)  # 이전 은닉 상태 (최초 은닉 상태는 0)

        # 패딩 처리 여부에 따른 처리
        if mask is None:

            # 시퀀스 동안 반복
            for t in range(T):

                h_matmul = h_prev @ W_h  # 이전 은닉 상태의 대한 가중치 matmul 계산
                h_z, h_r, h_g = h_matmul.chunk(3, dim=-1)  # 게이트 별 matmul 결과 저장

                zt = torch.sigmoid(x_z[:, t] + h_z + b_z)
                rt = torch.sigmoid(
                    x_r[:, t] + h_r + b_r
                )  # 시퀀스 T에 대한 리셋 게이트 계산
                gt = torch.tanh(
                    x_g[:, t] + rt * h_g + b_g
                )  # 시퀀스 T에 대한 이전 은닉 상태 반영 계산
                h_t = (1 - zt) * gt + zt * h_prev  # 시퀀스 T에 대한 최종 은닉 상태

                h_states[:, t] = h_t  # 시퀀스 T에 대한 은닉 상태 저장
                h_prev = h_t  # 은닉 상태 갱신
        else:

            # 시퀀스 동안 반복
            for t in range(T):

                h_matmul = h_prev @ W_h  # 이전 은닉 상태의 대한 가중치 matmul 계산
                h_z, h_r, h_g = h_matmul.chunk(3, dim=-1)  # 게이트 별 matmul 결과 저장

                zt = torch.sigmoid(
                    x_z[:, t] + h_z + b_z
                )  # 시퀀스 T에 대한 업데이트 게이트 계산
                rt = torch.sigmoid(
                    x_r[:, t] + h_r + b_r
                )  # 시퀀스 T에 대한 리셋 게이트 계산
                gt = torch.tanh(
                    x_g[:, t] + rt * h_g + b_g
                )  # 시퀀스 T에 대한 이전 은닉 상태 반영 계산
                h_t = (1 - zt) * gt + zt * h_prev  # 시퀀스 T에 대한 최종 은닉 상태

                m = (
                    mask[:, t].unsqueeze(-1).to(h_t.dtype)
                )  # h_t와 동일한 타입으로 캐스팅, .unsqueeze(-1) = 마지막 차원에 차원 추가
                h_t = torch.where(m == 1, h_t, h_prev)  # 패딩이면(m==0) 이전 상태 유지

                h_states[:, t] = h_t  # 시퀀스 T에 대한 은닉 상태 저장
                h_prev = h_t  # 은닉 상태 갱신

        return h_states  # 시퀀스 전체에 대한 최종 은닉 상태 결과

    # 순전파 (역전파는 자동으로 처리)
    def forward(
        self, x, mask: Optional[torch.Tensor] = None
    ):  # 입력 텐서, 패딩 여부 -> 출력 타입 = 텐서

        h_fwd = self.GRU(
            x, mask, self.W_x_fwd, self.W_h_fwd, self.b_fwd
        )  # 정방향 순전파

        x_reverse = torch.flip(x, dims=[1])  # 시퀀스 기준으로 입력 반전
        m_reverse = (
            torch.flip(mask, dims=[1]) if mask is not None else None
        )  # 시퀀스 기준으로 패딩 반전

        h_bwd = self.GRU(
            x_reverse, m_reverse, self.W_x_bwd, self.W_h_bwd, self.b_bwd
        )  # 역방향 순전파
        h_bwd = torch.flip(h_bwd, dims=[1])  # 역방향 결과 원상 복귀

        return torch.cat(
            [h_fwd, h_bwd], dim=-1
        )  # 정방향, 역방향 결과 Concatenate (N, T, 2H) = 양방향


# KoBERT + GRU LM
class KoBERT_GRU_LM(nn.Module):

    # 초기화 함수
    def __init__(
        self,
        hidden_size=256,
        classes=2,
        freeze=False,
        dropout=0.1,
        model="skt/kobert-base-v1",
        remove_cls=True,
    ):  # 은닉 상태 크기, 분류 클래스 수(피싱, 정상), 임베딩 계층 학습 여부, 드롭아웃 비율, KoBERT 모델명, fast 토크나이저 (Rust 기반) 사용 여부, CLS 토큰 사용 여부

        super().__init__()  # nn.Module 초기화 함수

        self.KoBERT = AutoModel.from_pretrained(model)  # 사전 학습된 KoBERT 모델 로드

        self.emb_size = (
            self.KoBERT.config.hidden_size
        )  # KoBERT config에 명시된 임베딩 차원 크기 정의

        self.remove_cls = remove_cls  # CLS 토큰 사용 여부 정의

        self.GRU = BiGRU(self.emb_size, hidden_size)  # GRU 계층 정의 (양방향)

        self.dropout = nn.Dropout(dropout)  # 드롭아웃 계층 정의

        final = 2 * hidden_size  # 양뱡향 GRU 최종 은닉 차원 정의

        self.norm = nn.LayerNorm(final)  # 정규화 계층 정의

        self.affine = nn.Linear(final, classes)  # Affine 계층 정의

        # 임베딩 계층 학습 여부 정의
        if freeze:
            self.freeze_bert()  # 학습 안함 (freeze)
        else:
            self.KoBERT_frozen = False  # 학습

    # requires_grad=False 설정
    def freeze_bert(self):
        for p in self.KoBERT.parameters():
            p.requires_grad = False  # 파라미터 기울기 계산 제외
        self.KoBERT_frozen = True

    # 순전파 (역전파는 자동으로 처리)
    def forward(
        self,
        input_ids,
        attention_mask,
    ):  # 입력 토큰, 패딩 구분 처리 -> 출력 타입 = 딕셔너리(문자열, 텐서)

        input_ids = input_ids.long()  # 입력 토큰 타입 캐스팅 (에러 방지)

        # KoBERT 임베딩 계층, 학습 여부에 따른 처리, KoBERT 모델에 토큰 입력 -> 벡터화
        if self.KoBERT_frozen:
            with torch.no_grad():  # 자동 역전파 제외
                output = self.KoBERT(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=None,
                    return_dict=True,
                )
        else:
            output = self.KoBERT(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=None,
                return_dict=True,
            )

        emb = output.last_hidden_state  # 최종 임베딩 결과 추출, 마지막 계층

        # CLS 사용 여부 처리
        if self.remove_cls:  # CLS 토큰 제거 (시퀀스 맨 앞 토큰)
            emb = emb[:, 1:, :]  # 은닉 벡터에서 CLS 토큰 제거
            mask = attention_mask[:, 1:]  # 마스크에서 CLS 토큰 제거
        else:
            mask = attention_mask  # 전체 마스크 사용

        emb = self.dropout(emb)  # 드롭아웃 적용

        seq_fb = self.GRU(emb, mask=mask)  # 양방향 GRU 계층, (N,T,2H)

        # 전체 평균 계산 (유효 토큰에 대한 평균)
        mask_reshape = mask.float().unsqueeze(
            -1
        )  # 마스킹을 GRU 출력과 동일한 차원으로 변환
        seq_sum = (seq_fb * mask_reshape).sum(
            dim=1
        )  # 시퀀스를 기준으로 GRU 출력과 마스킹을 곱하여 합
        token_num = mask_reshape.sum(dim=1).clamp(
            min=1.0
        )  # 시퀀스를 기준으로 패딩을 제외한 유효 토큰 수 합, 분모 0 방지
        mean = seq_sum / token_num  # 유효 토큰에 대한 전체 평균, (N, 2H) mean pooling

        mean = self.norm(mean)  # 정규화 계층 적용 (배치 크기 무시)

        mean = self.dropout(mean)  # 드롭아웃 적용

        logits = self.affine(mean)  # Affine 계층 적용

        return {"logits": logits}  # 모델 순전파 결과
