import logging

from aiogram import Bot, Dispatcher
from aiogram.contrib.fsm_storage.memory import MemoryStorage

API_TOKEN = '7083723406:AAELIcATy9nzzQNI2dWRvyIIRQ5S4z5vrqM'

bot = Bot(token=API_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)

logging.basicConfig(format=u'%(filename)s [LINE:%(lineno)d] #%(levelname)-8s \
                    [%(asctime)s]  %(message)s',
                    level=logging.INFO,
                    )
