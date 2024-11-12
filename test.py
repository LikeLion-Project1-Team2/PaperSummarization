import requests

COLAB_SERVER_URL = "https://erp-someone-toolbox-womens.trycloudflare.com/test"  # Colab 터널 URL을 여기에 입력하세요

def test_colab_connection():
    try:
        response = requests.get(COLAB_SERVER_URL)
        if response.status_code == 200:
            print("Colab 서버 연결 성공:", response.json())
        else:
            print("Colab 서버 연결 실패: 상태 코드", response.status_code)
    except requests.exceptions.RequestException as e:
        print("연결 오류:", e)

# Colab 서버에 테스트 요청 보내기
test_colab_connection()