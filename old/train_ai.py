import requests
import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

API_URL = "http://54.205.230.141:8000/api/csi/dataset"

print("[*] EC2 서버에서 학습 데이터를 가져오는 중...")
try:
    res = requests.get(API_URL).json()
    data = res['data']
    print(f"[+] 총 {len(data)}개의 정답 데이터를 불러왔습니다!\n")
except Exception as e:
    print(f"[!] 데이터 다운로드 실패. 서버 상태나 IP를 확인하세요: {e}")
    exit()

# 1. 데이터 프레임 변환 (전처리)
features = []
labels = []

for row in data:
    variance = row['variance']
    mean_wave = json.loads(row['mean_wave'])
    
    # 33개의 숫자로 이루어진 1차원 배열(Feature) 생성
    row_features = [variance] + mean_wave
    features.append(row_features)
    labels.append(row['label'])

df_X = pd.DataFrame(features)
df_y = pd.Series(labels)

# 2. 학습용(80%) / 테스트용(20%) 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=42)

print("[*] 랜덤 포레스트 AI 모델 학습 시작...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 3. 모델 정확도 평가
print("\n================ AI 채점 결과 ================")
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
print("==============================================")

# 4. 훈련된 모델 저장
joblib.dump(model, 'csi_rf_model.pkl')
print("\n[+] 인공지능 저장 완료: csi_rf_model.pkl (이제 실시간 예측 가능!)")