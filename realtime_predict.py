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
import torch.nn.functional as F
from collections import deque

# ==========================================
# 세팅: 실시간 예측 API 주소 및 S3 보드 IP주소
# ==========================================
EC2_URL = "http://54.205.230.141:8000/api/csi/upload"
BOARD_IPS = ["192.168.219.106", "192.168.219.105" ]

# ==========================================
#  딥러닝(LSTM 모델) 불러오기 
# (train_lstm.py 구조와 동일해야 함)
# ==========================================
class CSILSTM(nn.Module):
    def __init__(self, input_size=771, hidden_size=64, num_classes=4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

try:
    scaler = joblib.load('csi_scaler.pkl')
    encoder = joblib.load('csi_encoder.pkl')
    
    # 모델 뼈대 만들고 방금 훈련한 가중치 덮어씌우기
    model = CSILSTM(input_size=771, hidden_size=64, num_classes=len(encoder.classes_)).to(device)
    model.load_state_dict(torch.load('csi_lstm_weights.pth'))
    model.eval() 
    print("[*] LSTM(듀얼 보드) 모델 로딩 완료")
except Exception as e:
    print(f"[!] 모델 로딩 실패: {e}")
    exit()

# ==========================================
# 듀얼 보드 동기화 및 5초 기억 버퍼 세팅
# 보드 2개가 하나로 합쳐지므로, 버퍼 IP별로 나누지 않고 '통합 버퍼 1개'만 씀
# ==========================================
history_buffer = collections.deque(maxlen=5) 
csi_buffers = {ip: [] for ip in BOARD_IPS}
buffer_lock = threading.Lock()
EXPECTED_LEN = 192

def udp_receiver():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", 8888))
    print("[*] 실시간 행동 추론 모니터링 시작")
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

def extract_features(data_list):
    df = pd.DataFrame(data_list)
    mean_wave = df.mean(axis=0).fillna(0).round(2).tolist()
    std_wave = df.std(axis=0).fillna(0).round(2).tolist()
    
    def fix_length(wave, target_len=EXPECTED_LEN):
        if len(wave) < target_len: return wave + [0.0] * (target_len - len(wave))
        return wave[:target_len]
    
    variance = float(df.var().mean()) if len(df) > 1 else 0.0
    if pd.isna(variance): variance = 0.0
    
    # 385개 반환 (분산 1 + 평균 192 + 편차 192)
    features = [round(variance, 2)] + fix_length(mean_wave) + fix_length(std_wave)
    return features, variance

# 다수결 스무딩을 위한 3칸짜리 버퍼 추가
pred_history = collections.deque(maxlen=3)
def processor():
    ip1, ip2 = BOARD_IPS[0], BOARD_IPS[1]
    
    while True:
        time.sleep(1.0) 
        with buffer_lock:
            data1 = csi_buffers[ip1][:]
            data2 = csi_buffers[ip2][:]
            csi_buffers[ip1].clear()
            csi_buffers[ip2].clear()
            
        # S3 보드 한쪽이라도 끊기면 값 버림
        # 딥러닝 모델 망치지 않도록
        if not data1 or not data2:
            print(f"[-] 동기화 대기/유실... (수신 패킷 - {ip1}: {len(data1)} / {ip2}: {len(data2)})")
            continue
            
        try:
            feat1, var1 = extract_features(data1)
            feat2, var2 = extract_features(data2)
            
            # S3보드 - 1. 보드 2개 데이터 병합 (385 + 385 = 770개)
            synced_features = feat1 + feat2
            # 분산값 가져옴
            avg_variance = round((var1 + var2) / 2, 2)
            


            #============================================
            # 추론 순서
            # empty - walking - 나머지 (lying, sitting)
            # 확률상 empty 추론이 가장 함듦
            # -> LSTM 모델이 파형 형태를 보고 판단하지만 사람 없는 상태(empty)면 일정한 노이즈만 나옴
            # -> sitting/lyting 형태와 구분 어려움
            # 미세한 떨림(사람의 호흡 등)이 아예 없는 variance(분산)값이 가장 낮은 값이 empty
            #============================================

            EMPTY_THRESHOLD = 0.35       # EMPTY 
            WALKING_THRESHOLD = 3.0     # 환경에 맞게 조정 (avg_variance 값)
            '''
            # variance로 walking/empty 판단 후 추론 건너뜀.
            # 그러나 sitting 상태일때 variance값이 높은 경우도 있음 
            if avg_variance >= WALKING_THRESHOLD:
                final_label = 'walking'
                confidence = 99.9
                print(f"[Filter] ({avg_variance}) ➔ [WALKING]")
                continue
            
            elif avg_variance < EMPTY_THRESHOLD:
                final_label ='empty'
                confidence = 99.9
                print(f"[Filter] ({avg_variance}) ➔ [EMPTY]")
                continue
            '''


            # empty 판단 - empty 판단시 AI모델 가동 안 함
            if avg_variance < EMPTY_THRESHOLD:
                predicted_label = 'empty'
                confidence = 100.0
                print(f"[물리 필터] 분산 낮음({avg_variance}) ➔ 판단 스킵 [EMPTY]")
            else:
                # S3보드 - 2. AI 입력용 최종 데이터 조립 (평균분산 1 + 합친 파형 770 = 771개)
                current_feature = [avg_variance] + synced_features
                # S3보드 - 3. 5초 기억 장치에 밀어 넣음
                history_buffer.append(current_feature)
            
                # 4. 5초치 데이터가 꽉 찼을 때만 AI 가동
                if len(history_buffer) == 5:
                    # (5, 771) 형태의 배열로 변환
                    seq_data = np.array(history_buffer) 
                    seq_scaled = scaler.transform(seq_data) 
                    # [1, 5, 771] 차원의 파이토치 텐서 생성
                    tensor_data = torch.FloatTensor([seq_scaled]).to(device)
                
                    # AI 예측
                    with torch.no_grad():
                        out = model(tensor_data)
                        
                        # 단순 1등 뽑기가 아니라 확률 계산
                        probs = F.softmax(out, dim=1)
                        max_prob, pred_idx = torch.max(probs, dim=1)
                        
                        confidence = max_prob.item() * 100
                        predicted_label = encoder.inverse_transform([pred_idx.item()])[0]

                    # =========================================
                    # empty 제외 판단 시작
                    # 
                    # Walking으로 잘못 판단 했을때 sitting/lying중 더 높은 확률 로 선택
                    # 물리 법칙 필터 (Hard Filter) 추가
                    # 걷기(Walking)라면 분산(avg_variance)값 높아야 함
                    # 빈상태(Empty)라면 avg_variance값 낮음 (움직임이 없음)
                    # ([]_THRESHOLD 값 이상)
                    # ==========================================
                   

                    if predicted_label == 'walking' and avg_variance < WALKING_THRESHOLD:
                        # 1. 라벨 번역기에서 'walking'의 고유 번호(Index)를 찾음
                        walking_idx = encoder.transform(['walking'])[0]
                        
                        # 2. AI (확률표)에서 'walking'의 확률만 0.0으로 완전 삭제
                        probs[0][walking_idx] = 0.0 
                        
                        # 3. 남은 후보들(empty, sitting, lying) 중에서 다시 1등(기존 2등)을 뽑음
                        new_max_prob, new_pred_idx = torch.max(probs, dim=1)
                        predicted_label = encoder.inverse_transform([new_pred_idx.item()])[0]
                        
                        # 4. 남은 애들끼리의 비율로 확신도(%)를 다시 계산
                        confidence = (new_max_prob.item() / torch.sum(probs).item()) * 100
                        
                        print(f"[!] Detected Misjudgment : Alternative [{predicted_label.upper()}] - (Confidence: {confidence:.1f}%)")

                else:
                    print(f"[-] Archiving Time Series Data for 5s ({len(history_buffer)}/5)")
                    continue

            # Confidence 컷시킴
            # 차선책으로 선택된 label도 confidence 낮으면 보류됨 (75% 미만)
            if confidence < 75.0:
                print(f"[-] (Lack Confidence: {confidence:.1f}% | Variance: {avg_variance}) -> Probably {predicted_label.upper()}")
                continue # 서버로 전송 안 하고 다음 1초로 넘어감
            
            # 2. 보정 2단계: 다수결 스무딩 (최근 3번의 확실한 예측 중 대세 따르기)
            pred_history.append(predicted_label)
            if len(pred_history) < 3:
                print(f"[-] 다수결 데이터 모으는 중... ({len(pred_history)}/3)")
                continue
                
            # 3개 중 가장 많이 나온 행동을 최종 정답으로 채택
            final_label = max(set(pred_history), key=pred_history.count)

            # EC2 전송
            payload = {
                "device_id": "DUAL_BOARD_SYNC",
                "variance": avg_variance,
                "mean_wave": synced_features,
                "predicted_action": str(final_label) #필터링된 최종값
            }
            res = requests.post(EC2_URL, json=payload, timeout=2)
            print(f"[+] Predict : [{final_label.upper():}] | Variance: {avg_variance} | Confidence:{confidence:.1f}% | Transmission status: {res.status_code}")
        
        except Exception as e:
            print(f"[!] 예측/전송 에러: {e}")

if __name__ == "__main__":
    threading.Thread(target=udp_receiver, daemon=True).start()
    threading.Thread(target=processor, daemon=True).start()
    while True: time.sleep(1)