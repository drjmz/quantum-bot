import asyncio
import os
from telegram import Bot

async def test():
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        print("❌ Error: Missing TELEGRAM_TOKEN or TELEGRAM_CHAT_ID in .env")
        return

    print(f"📲 Sending Test Message to {chat_id}...")
    try:
        bot = Bot(token=token)
        await bot.send_message(chat_id=chat_id, text="✅ <b>Quantum Node Online</b>\n\nUplink Established.", parse_mode='HTML')
        print("✅ Success! Check your phone.")
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    asyncio.run(test())
