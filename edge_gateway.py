import socket
import threading
import time
import requests
import numpy as np
import pandas as pd
import joblib # AI 모델 불러옴
import collections

UDP_IP = "0.0.0.0"
UDP_PORT = 8888
EC2_URL = "http://54.205.230.141:8000/api/csi/upload"

#  사용할 ESP32 보드들의 IP
BOARD_IPS = ["192.168.219.101" ] 
# "192.168.219.105", "192.168.219.107" 

prediction_history = {ip: collections.deque(maxlen=3) for ip in BOARD_IPS}

# Load AI Model
try:
    ai_model = joblib.load('csi_rf_model.pkl')
    EXPECTED_LEN = ai_model.n_features_in_ - 1
    print("[*] AI 모델 로딩 완료")
except Exception as e:
    print("[!] csi_rf_model.pkl 파일 없음 - train_ai.py 코드 실행 권장")
    exit()

# 보드 IP별로 리스트 따로 생성
csi_buffers = {ip: [] for ip in BOARD_IPS}
buffer_lock = threading.Lock()
current_action = "unlabeled" 

def udp_receiver_thread():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"[*] 다중 노드 수신 대기 중... 등록된 기기: {BOARD_IPS}")

    while True:
        try:
            data, addr = sock.recvfrom(2048)
            sender_ip = addr[0] # 데이터 쏜 놈의 IP 확인
            
            # 등록한 보드 IP에서 온 데이터만 처리
            if sender_ip in csi_buffers:
                raw_data = np.frombuffer(data, dtype=np.int8).astype(np.float32)
                valid_len = len(raw_data) - (len(raw_data) % 2)
                
                if valid_len >= 64:
                    csi_iq = raw_data[:valid_len]
                    amplitude = np.sqrt(csi_iq[0::2]**2 + csi_iq[1::2]**2)
                    
                    # Lock 걸고 자기 IP 리스트만 넣음
                    with buffer_lock:
                        csi_buffers[sender_ip].append(amplitude)
        except Exception:
            pass



def edge_processor_thread():
    global current_action
    while True:
        time.sleep(1.0) 
        
        with buffer_lock:
            # IP별 바구니 데이터를 복사 / 원본 비움
            batch_data = {ip: csi_buffers[ip][:] for ip in BOARD_IPS}
            for ip in BOARD_IPS:
                csi_buffers[ip].clear()
                
        # 각 보드 IP별로 전처리해서 EC2로 따로따로 쏨
        for ip, data_list in batch_data.items():
            if not data_list:
                continue
                
            try:
                df = pd.DataFrame(data_list)
                mean_wave = df.mean(axis=0).fillna(0).round(2).tolist()

                if len(mean_wave) < EXPECTED_LEN:
                    mean_wave += [0.0] * (EXPECTED_LEN - len(mean_wave)) # 모자라면 0으로 채움
                else:
                    mean_wave = mean_wave[:EXPECTED_LEN] # 넘치면 자름

                variance = float(df.var().mean()) if len(df) > 1 else 0.0
                if pd.isna(variance): variance = 0.0

                features = [[variance] + mean_wave]
                predicted_label = str(ai_model.predict(features)[0])
                    

                # 1. 모델 예측
                raw_predicted = str(ai_model.predict(features)[0])
                
                # 2. 최근 3번의 기록에 추가
                prediction_history[ip].append(raw_predicted)
                
                # 3. 최빈값(가장 많이 나온 결과) 추출 로직
                counter = collections.Counter(prediction_history[ip])
                smoothed_label = counter.most_common(1)[0][0]

                payload = {
                    "device_id": ip,  # 어느 보드 데이터인지 이름표 붙임
                    # "packet_count": len(data_list),
                    "variance": round(variance, 2),
                    "mean_wave": mean_wave,
                    # "label": current_action  
                    "predicted_action": smoothed_label,
                }
                
                res = requests.post(EC2_URL, json=payload, timeout=2)
                # print(f"[+] {ip} (패킷: {len(data_list)}개) | 움직임: {variance:.2f} | 전송: {res.status_code}")
                print(f"[+] {ip} | 서버 응답: {res.text}")
                # print(f"[+] 예측 결과: 예측결과: [{predicted_label.upper()}] | 전송: {res.status_code}")
                
            except Exception as e:
                print(f"[!] {ip} 전처리/전송 에러: {e}")

if __name__ == "__main__":
    current_action = input("[?] 수집할 행동 라벨 입력 (예: empty, walking): ")
    print(f"\n[*]  '{current_action}' 다중 데이터 수집 시작...\n")
    
    # print("\n[*] 실시간 행동 추론 모니터링 시작\n")

    threading.Thread(target=udp_receiver_thread, daemon=True).start()
    threading.Thread(target=edge_processor_thread, daemon=True).start()
    
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("\n[*] 종료")