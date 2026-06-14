import socket
import threading
import time
import requests
import numpy as np
import pandas as pd
import os

# ==========================================
# EC2 FastAPI / S3보드 연동
# ==========================================
EC2_URL = "http://54.205.230.141:8000/api/csi/upload"
BOARD_IPS = ["192.168.219.114", "192.168.219.105" ] 
# "192.168.219.105", "192.168.219.107" 

# 수집 행동
CURRENT_LABEL = input('수집할 행동 라벨 선택 (택 1)\n(empty, sitting, walking, lying) : ') 
TARGET_NUMBER = input('수집할 데이터 개수 : ')
# ==========================================

csi_buffers = {ip: [] for ip in BOARD_IPS}
buffer_lock = threading.Lock()
EXPECTED_LEN = 192


def udp_receiver():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", 8888))
    print(f"[*] 듀얼 보드 동기화 후 수집 : (목표 라벨: {CURRENT_LABEL.upper()})")

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

#  보드 1대 분량의 특징(385개)을 추출하는 함수
def extract_features(data_list):
    df = pd.DataFrame(data_list)
    mean_wave = df.mean(axis=0).fillna(0).round(2).tolist()
    std_wave = df.std(axis=0).fillna(0).round(2).tolist()
    
    def fix_length(wave, target_len=EXPECTED_LEN):
        if len(wave) < target_len: return wave + [0.0] * (target_len - len(wave))
        return wave[:target_len]
    
    variance = float(df.var().mean()) if len(df) > 1 else 0.0
    if pd.isna(variance): variance = 0.0
    
    # [분산 1개] + [평균 192개] + [표준편차 192개] = 총 385개
    features = [round(variance, 2)] + fix_length(mean_wave) + fix_length(std_wave)
    return features, variance

def processor():
    ip1, ip2 = BOARD_IPS[0], BOARD_IPS[1]
    
    count = 0
    while True:
        time.sleep(1.0) # 1초 단위 윈도우
        
        with buffer_lock:
            # 두 보드의 데이터를 동시에 복사하고 버퍼 초기화
            data1 = csi_buffers[ip1][:]
            data2 = csi_buffers[ip2][:]
            csi_buffers[ip1].clear()
            csi_buffers[ip2].clear()
            
        # #  =================================
        # #  둘 중 하나라도 통신이 끊겼다면 데이터 폐기!
        # if not data1 or not data2:
        #     print(f"[-] 동기화 대기/유실... (수신 패킷 - {ip1}: {len(data1)}개 / {ip2}: {len(data2)}개)")
        #     continue
        # #  =================================

        # =======================
        # 양쪽 s3보드 PPS 40개 이상 확인
        # 이하면 데이터 버림
        # =======================
        if len(data1) < 40 or len(data2) < 40:
            print(f"[-] Ignore Data (Lack PPS) | B1 : {len(data1):>3} | B2 : {len(data2):>3}")
            continue
            
        try:
            # 1. 각각 385개의 특징 추출
            feat1, var1 = extract_features(data1)
            feat2, var2 = extract_features(data2)
            
            # 2. 병합 (Merge): 385개 + 385개 = 총 770개의 1차원 배열 완성
            synced_features = feat1 + feat2
            avg_variance = round((var1 + var2) / 2, 2)
            min_packet_count = min(len(data1), len(data2))
            
            # 3. EC2 서버로 전송 (송장 양식에 맞춤)
            payload = {
                "device_id": "DUAL_BOARD_SYNC", # DB에서 한 쌍임을 알 수 있게 이름 고정
                "variance": avg_variance,
                "mean_wave": synced_features,   #  DB의 raw_data에는 770개가 들어감
                "packet_count": min_packet_count,
                "label": CURRENT_LABEL
            }
            
            res = requests.post(EC2_URL, json=payload, timeout=2)

            print(f"[수집 - {count}] 동기화 완료 (패킷 {min_packet_count}개) | {CURRENT_LABEL.upper()} DB 전송: {res.status_code}")
            count += 1
            if(count >= int(TARGET_NUMBER)):
                print(f'목표 데티어 {TARGET_NUMBER}개 수집 완료')
                os._exit(0)
            
        except Exception as e:
            print(f"[!] 전처리 및 전송 에러: {e}")

if __name__ == "__main__":
    threading.Thread(target=udp_receiver, daemon=True).start()
    threading.Thread(target=processor, daemon=True).start()
    while True: time.sleep(1)