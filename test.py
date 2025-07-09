import requests

TOKEN = "7928426674:AAElIFCM4hXJQNFvUKPcJFbmika2gohIAIc"  # Replace with your token
CHAT_ID = "-1002357030636"  # Replace with your channel ID

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message
    }
    response = requests.post(url, json=payload)
    return response.json()

# Test
result = send_telegram("🚀 Bot is working!")
print("Telegram response:", result)
