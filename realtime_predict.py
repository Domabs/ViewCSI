import socket
import threading
import time
import requests
import numpy as np
import pandas as pd
import collections
import torch
import torch.nn as nn
import joblib

# ==========================================
# 세팅: 실시간 예측 API 주소
# ==========================================
EC2_URL = "http://54.205.230.141:8000/api/csi/upload"
BOARD_IPS = ["192.168.219.106" ] 
# "192.168.219.105", "192.168.219.107" 

# ==========================================
#  딥러닝(LSTM) 불러오기 (train_lstm.py 구조와 동일해야 함)
# ==========================================
class CSILSTM(nn.Module):
    def __init__(self, input_size=385, hidden_size=128, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

try:
    # 스케일러와 라벨 번역기 로드
    scaler = joblib.load('csi_scaler.pkl')
    encoder = joblib.load('csi_encoder.pkl')
    
    # 모델 뼈대 만들고 가중치 덮어씌우기
    model = CSILSTM(input_size=385, hidden_size=128, num_classes=len(encoder.classes_)).to(device)
    model.load_state_dict(torch.load('csi_lstm_weights.pth'))
    model.eval() # 실전(추론) 모드로 변경
    print("[*] LSTM 모델 로딩 완료")
except Exception as e:
    print(f"[!] 모델 로딩 실패 (train_lstm.py부터 돌리세요): {e}")
    exit()

#  3초(3프레임) 윈도우 저장을 위한 큐(Queue)
history_buffer = {ip: collections.deque(maxlen=5) for ip in BOARD_IPS}
csi_buffers = {ip: [] for ip in BOARD_IPS}
buffer_lock = threading.Lock()
EXPECTED_LEN = 192
# ai_model.n_features_in_ - 1

def udp_receiver():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", 8888))
    print("[*] 실시간 AI 행동 추론 모니터링을 시작")
    while True:
        try:
            data, addr = sock.recvfrom(2048)
            sender_ip = addr[0]
            if sender_ip in csi_buffers:
                raw_data = np.frombuffer(data, dtype=np.int8).astype(np.float32)
                valid_len = len(raw_data) - (len(raw_data) % 2)
                if valid_len >= 64:
                    csi_iq = raw_data[:valid_len]
                    amplitude = np.sqrt(csi_iq[0::2]**2 + csi_iq[1::2]**2)
                    with buffer_lock:
                        csi_buffers[sender_ip].append(amplitude)
        except: pass

def processor():
    while True:
        time.sleep(1.0) # 1초마다 루프
        with buffer_lock:
            batch_data = {ip: csi_buffers[ip][:] for ip in BOARD_IPS}
            for ip in BOARD_IPS: csi_buffers[ip].clear()
                
        for ip, data_list in batch_data.items():
            if not data_list: continue
            try:
                df = pd.DataFrame(data_list)
                mean_wave = df.mean(axis=0).fillna(0).round(2).tolist()
                std_wave = df.std(axis=0).fillna(0).round(2).tolist()
                
                def fix_length(wave, target_len=192):
                    if len(wave) < target_len: return wave + [0.0] * (target_len - len(wave))
                    return wave[:target_len]
                
                combined_wave = fix_length(mean_wave) + fix_length(std_wave)
                variance = float(df.var().mean()) if len(df) > 1 else 0.0
                if pd.isna(variance): variance = 0.0
                
                # 1. 특징 배열 조립 (총 385개)
                current_feature = [variance] + combined_wave 
                
                # 2. 5초 기억 장치에 밀어 넣음
                history_buffer[ip].append(current_feature)
                
                # 3. 5초 데이터 꽉 찼을 때 AI 예측 (len 확인 5로 변경)
                if len(history_buffer[ip]) == 5:
                    seq_data = np.array(history_buffer[ip])
                    
                    # 훈련할 때와 똑같이 스케일링(정규화)
                    seq_scaled = scaler.transform(seq_data) 
                    
                    # 파이토치 텐서로 변환 (배치 사이즈 1추가 -> [1, 3, 193])
                    tensor_data = torch.FloatTensor([seq_scaled]).to(device)
                    
                    # AI 예측
                    with torch.no_grad():
                        out = model(tensor_data)
                        pred_idx = torch.argmax(out, dim=1).item()
                        predicted_label = encoder.inverse_transform([pred_idx])[0]
                    
                    # EC2 실시간 테이블로 전송
                    payload = {
                        "device_id": ip,
                        "variance": round(variance, 2),
                        "mean_wave": mean_wave,
                        "predicted_action": str(predicted_label)
                    }
                    res = requests.post(EC2_URL, json=payload, timeout=2)
                    print(f"[+] {ip} | 🤖 예측:[{predicted_label.upper()}] | 전송: {res.status_code}")
                else:
                    print(f"[-] {ip} | 데이터 묶는 중... ({len(history_buffer[ip])}/3)")
                    
            except Exception as e:
                print(f"[!] {ip} 예측/전송 에러: {e}")

if __name__ == "__main__":
    threading.Thread(target=udp_receiver, daemon=True).start()
    threading.Thread(target=processor, daemon=True).start()
    while True: time.sleep(1)