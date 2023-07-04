import pywinauto
from pywinauto.application import Application
from pywinauto.keyboard import send_keys
from pywinauto import keyboard as kb
import time


def test_native_app():
    app = Application(backend="uia").start('C://Users//user//Documents//FlowersBotBuildDev//WpfApp2.exe')
    time.sleep(2)

    dlg_spec = app.window(title='Авторизация')
    dlg_spec_2 = app.window(title='Короткий пароль')
    dlg_spec_3 = app.window(title='FlowersBot')

    # dlg_spec.print_control_identifiers()
    # dlg_spec_4.print_control_identifiers()
    # dlg_spec_2.print_control_identifiers()

    if dlg_spec.exists():
        dlg_spec.wait('visible'), dlg_spec.child_window(auto_id="passwordBox").set_text("Gfhjkm"), \
        dlg_spec.OKButton.click(), dlg_spec_2.wait('visible'), \
        dlg_spec_2.child_window(auto_id="passwordBox").set_text('1234'), dlg_spec_2.OKButton.click()

    else:
        dlg_spec_2.wait('visible'), dlg_spec_2.child_window(auto_id="passwordBox").set_text('1234'), \
        dlg_spec_2.OKButton.click()

    dlg_spec_3.wait('visible')
    send_keys('{VK_TAB}')
    dlg_spec_3.child_window(title="Товары", control_type="Button").click()
    time.sleep(2)
    send_keys('{DOWN}')
    send_keys('{DOWN}')
    send_keys('{RIGHT}')


    time.sleep(2)
    # dlg_spec_3.print_control_identifiers()

    # Button - ''(L67, T132, R83, B148)
    # | | | | | | ['ТоварыButton', 'Button5', 'ТоварыButton0', 'ТоварыButton1']
    # | | | | | | child_window(auto_id="Expander", control_type="Button")

    # Static - 'Абеме'(L85, T132, R121, B148)
    # | | | | | | ['Абеме2', 'Static4', 'АбемеStatic']
    # | | | | | | child_window(title="Абеме", control_type="Text")

    # TreeItem - 'Абеме'(L65, T132, R356, B148)
    # | | | | | ['TreeItem2', 'АбемеTreeItem', 'Абеме', 'Абеме0', 'Абеме1']
    # | | | | | child_window(title="Абеме", control_type="TreeItem")

    # TabControl - ''(L60, T83, R1324, B764)
    # | ['TabControl', 'TabControl+', 'ТоварыTabControl']
    # | child_window(auto_id="TabControl", control_type="Tab")
    app.kill()
