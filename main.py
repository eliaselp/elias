from bot import Bot

print("inicializando bot")
bot = Bot.load_state()
if bot is None:
    bot = Bot()


print('Iniciar de bot')
bot.iniciar()

