import sys, os
global APP_PATH
if getattr(sys, 'frozen', False):
    APP_PATH = sys._MEIPASS
else:
    APP_PATH = os.path.dirname(os.path.abspath(__file__))