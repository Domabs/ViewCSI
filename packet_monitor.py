import socket
import threading
import time

# 세팅: 측정할 S3 보드 IP
BOARD_IPS = ["192.168.219.106", "192.168.219.105" ]

# IP별 패킷 카운터 딕셔너리
packet_counts = {ip: 0 for ip in BOARD_IPS}
packet_counts["unknown"] = 0  # 등록되지 않은 IP 추적용

def monitor_pps():
    """1초마다 패킷 개수를 화면에 출력하고 0으로 초기화"""
    while True:
        time.sleep(1.0)
        print("\n" + "=" * 40)
        print("[실시간 1초당 패킷 수신량 (PPS)]")
        
        for ip in BOARD_IPS:
            count = packet_counts[ip]
            
            # 수신량에 따른 상태 표시
            if count >= 60:
                status = "좋음"
            elif count >= 30:
                status = "다소 부족 (유실 의심)"
            elif count > 0:
                status = "심각한 패킷 드랍"
            else:
                status = "수신 없음"
                
            print(f"{ip}: {count:^4} 개/초 | {status}")

        if packet_counts["unknown"] > 0:
            print(f" 미등록 IP 패킷: {packet_counts['unknown']} 개/초")
            
        print("=" * 40)

        # 카운트 초기화 (다음 1초 측정을 위해)
        for key in packet_counts:
            packet_counts[key] = 0

def udp_receiver():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # 운영체제 수신 버퍼 1MB로 강제 확장 (측정 중 병목 방지)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
    sock.bind(("0.0.0.0", 8888))

    print("[*] 패킷 수신량 정밀 모니터링 가동")

    while True:
        try:
            data, addr = sock.recvfrom(2048)
            sender_ip = addr[0]

            # 보낸 사람 IP 확인 후 카운트 +1
            if sender_ip in packet_counts:
                packet_counts[sender_ip] += 1
            else:
                packet_counts["unknown"] += 1
        except Exception as e:
            pass

if __name__ == "__main__":
    # 측정 결과 출력 스레드 가동
    threading.Thread(target=monitor_pps, daemon=True).start()
    
    # 메인 스레드는 패킷 수신에만 100% 집중
    udp_receiver()