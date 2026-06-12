import pymysql
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# ==========================================
# 1. DB 연결 및 데이터 불러오기
# ==========================================
DB_HOST = "54.205.230.141"
DB_USER = "root"
DB_PASSWORD = "a1234"
DB_NAME = "ruview"

print("[*] DB에서 데이터 가져오는 중...")

db_url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
engine = create_engine(db_url)

query = "SELECT avg_amplitude AS variance, raw_data, label FROM csi_logs ORDER BY timestamp ASC"
df = pd.read_sql(query, engine)

# JSON 파형 데이터 풀기
EXPECTED_LEN = 770
def parse_raw(x):
    arr = json.loads(x)
    if len(arr) < EXPECTED_LEN:
        arr += [0.0] * (EXPECTED_LEN - len(arr))
    else:
        arr = arr[:EXPECTED_LEN]
    return arr

df['mean_wave'] = df['raw_data'].apply(parse_raw)
features_df = pd.DataFrame(df['mean_wave'].tolist())
features_df.insert(0, 'variance', df['variance'])

# ==================================
# 2. 핵심: 3초(3-Row) 슬라이딩 윈도우 생성
# ==================================
SEQ_LEN = 5  # 3초 단위
X_seq, y_seq = [], []
labels = df['label'].values
features = features_df.values

print(f"[*] {SEQ_LEN}초 단위로 데이터를 자릅니다 (Windowing)...")
for i in range(len(features) - SEQ_LEN + 1):
    window_labels = labels[i : i + SEQ_LEN]
    # 3초 동안 행동이 안 바뀌고 똑같을 때만 학습 데이터로 씀 (노이즈 방지)
    if len(set(window_labels)) == 1:
        X_seq.append(features[i : i + SEQ_LEN])
        y_seq.append(window_labels[0])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# 정규화 및 라벨 인코딩
scaler = StandardScaler()
# 3차원 데이터를 잠시 2차원으로 펴서 스케일링 후 다시 3차원으로 복구
X_seq = scaler.fit_transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)

encoder = LabelEncoder()
y_seq = encoder.fit_transform(y_seq)

X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# ==========================================
# 3. PyTorch 텐서 변환 및 데이터로더
# ==========================================
# 맥미니 GPU 가속 세팅
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[*] 학습 장치: {device}")

train_data = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
test_data = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# ==========================================
# 4. LSTM 딥러닝 모델 설계
# ==========================================

# 모델 크기 385 * 2 = 770 (s3 보드 2개로 수집)
class CSILSTM(nn.Module):
    def __init__(self, input_size=771, hidden_size=128, num_classes=4):
        super().__init__()
        # 시계열 기억 장치 (LSTM)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # 최종 판단 레이어
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        # 3초 중 가장 마지막(최신) 시점의 기억만 가져와서 정답 도출
        out = self.fc(out[:, -1, :]) 
        return out

model = CSILSTM(input_size=771, hidden_size=64, num_classes=len(encoder.classes_)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# 5. 모델 훈련 (Epochs)
# ==========================================
EPOCHS = 30
print("\n[*] LSTM 모델 훈련 시작...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")

# ==========================================
# 6. 정확도 테스트 및 모델 저장
# ==========================================
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

print(f"\n 딥러닝(LSTM) 최종 테스트 정확도: {100 * correct / total:.2f}%")

# 실전 추론을 위해 모델 뼈대와 스케일러 저장
torch.save(model.state_dict(), 'csi_lstm_weights.pth')
import joblib
joblib.dump(scaler, 'csi_scaler.pkl')
joblib.dump(encoder, 'csi_encoder.pkl')
print("[*] 모델 및 스케일러 저장 완료 (csi_lstm_weights.pth 등)")