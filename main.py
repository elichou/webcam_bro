import asyncio
import telegram

TOKEN = "7003877059:AAHuGD_HzyBrAtlL5Tc8_6Bh3PkZ68WSW6k"

async def main():
    bot = telegram.Bot(TOKEN)
    async with bot:
        await bot.send_message(text='Hi John!', chat_id=650813102)
if __name__ == '__main__':
    asyncio.run(main())