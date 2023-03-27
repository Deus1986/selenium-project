import pywinauto
from pywinauto.application import Application
from pywinauto.keyboard import send_keys
from pywinauto import keyboard as kb
import time

def test_native_app():
    app = Application(backend="uia").start('C://Users//user//Documents//FlowersBotBuildDev//WpfApp2.exe')
    time.sleep(2)

    dlg_spec = app.window(title='Авторизация')
    actionable_dlg = dlg_spec.wait('visible')

    dlg_spec.print_control_identifiers()

    dlg_spec.OKButton.click()