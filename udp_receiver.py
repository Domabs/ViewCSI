import socket
import threading
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

UDP_IP = "0.0.0.0"
UDP_PORT = 8888

# 최신 파형 데이터를 담아둘 큐
latest_csi = deque(maxlen=1)

def udp_receiver_thread():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"[*] 실시간 그래프 수신 대기 중... (포트: {UDP_PORT})")

    while True:
        try:
            data, _ = sock.recvfrom(2048)
            # 오버플로우 방지 캐스팅
            raw_data = np.frombuffer(data, dtype=np.int8).astype(np.float32)
            
            # 짝수 길이 강제 맞춤 (에러 완벽 방어)
            valid_len = len(raw_data) - (len(raw_data) % 2)
            
            if valid_len >= 64:
                csi_iq = raw_data[:valid_len]
                i_data = csi_iq[0::2]
                q_data = csi_iq[1::2]
                
                # 진폭 계산 후 큐에 삽입
                amplitude = np.sqrt(i_data**2 + q_data**2)
                latest_csi.append(amplitude)
                
        except Exception:
            pass # 백그라운드 스레드이므로 에러나도 조용히 패스

def start_visualization():
    # 데이터 수신 스레드 백그라운드 실행
    t = threading.Thread(target=udp_receiver_thread, daemon=True)
    t.start()

    # Matplotlib 설정
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 간지나는 형광 민트색 선으로 세팅
    line, = ax.plot([], [], '-', color='#00ffcc', linewidth=2)
    
    ax.set_ylim(0, 100) # 기본 Y축 높이 0~100
    ax.set_title("RuView Real-time CSI Amplitude", fontsize=16, fontweight='bold')
    ax.set_xlabel("Subcarrier Index", fontsize=12)
    ax.set_ylabel("Amplitude", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 다크 모드 배경
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#1e1e1e')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')

    print("\n" + "="*50)
    print("[*] 다른 터미널에 Ping을 계속 날려주세요! (ping 192.168.219.136)")
    print("="*50 + "\n")

    try:
        while True:
            if latest_csi:
                current_amplitude = latest_csi[-1]
                
                # 들어온 서브캐리어 길이에 맞춰 X축 자동 조정
                x = np.arange(len(current_amplitude))
                line.set_data(x, current_amplitude)
                ax.set_xlim(0, len(current_amplitude) - 1)
                
                # Y축 자동 확장: 진폭이 기본 천장(100)을 뚫으면 천장을 자동으로 높여줌
                max_amp = np.max(current_amplitude)
                if max_amp > ax.get_ylim()[1]:
                    ax.set_ylim(0, max_amp + 20)
                
                fig.canvas.draw()
                fig.canvas.flush_events()
            
            # FPS 조절 (초당 20프레임)
            plt.pause(0.05)
    except KeyboardInterrupt:
        print("\n[*] 시각화 종료.")

if __name__ == "__main__":
    start_visualization()