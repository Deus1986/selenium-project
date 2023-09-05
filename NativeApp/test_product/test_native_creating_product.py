import pywinauto
import pyautogui as pag
from pywinauto.application import Application
from pywinauto.keyboard import send_keys
from pywinauto import keyboard as kb
import time
import allure


def test_native_creating_product_price_negative():
    app = Application(backend="uia").start('C://Users//user//Documents//FlowersBotBuildDev//FlowersCRM.exe')
    time.sleep(2)

    dlg_spec = app.window(title='Авторизация')
    dlg_spec_2 = app.window(title='Короткий пароль')
    dlg_spec_3 = app.window(title='FlowersBot')
    dlg_spec_4 = app.window(title='PriceWindow')
    dlg_spec_5 = app.window(title='Выбор альбомов')

    # dlg_spec.print_control_identifiers()
    # dlg_spec_4.print_control_identifiers()
    # dlg_spec_2.print_control_identifiers()

    with allure.step("Ввести пароль для входа в приложение"):
        if dlg_spec.exists():
            dlg_spec.wait('visible'), dlg_spec.child_window(auto_id="passwordBox").set_text("Gfhjkm"), \
            dlg_spec.OKButton.click(), dlg_spec_2.wait('visible'), \
            dlg_spec_2.child_window(auto_id="passwordBox").set_text('1234'), dlg_spec_2.OKButton.click()

        else:
            dlg_spec_2.wait('visible'), dlg_spec_2.child_window(auto_id="passwordBox").set_text('1234'), \
            dlg_spec_2.OKButton.click()

    with allure.step("Нажать товары"):
        dlg_spec_3.wait('visible')
        time.sleep(2)
        dlg_spec_3.Static2.click_input()
        dlg_spec_3.Static3.click_input()

    with allure.step("Выбрать товары абеме"):
        dlg_spec_3.АбемеListItem.click_input()

    with allure.step("Нажать создать товар"):
        dlg_spec_3.child_window(title="Создать товар", auto_id="CreateProduct", control_type="Button").click_input()

    with allure.step("Ввести имя товара Барабулька"):
        dlg_spec_3.Edit.set_edit_text("Барабулька")

    with allure.step("Открыть окно редактирования цены, ввести наименование товара и установить одинаковую цену "
                     "закупки и себестоимость"):
        dlg_spec_3.Static262.click_input()
        dlg_spec_4.НазваниеEdit.set_edit_text("Барабулька хвостатая")
        dlg_spec_4.ЦенаEdit.set_edit_text("1")
        dlg_spec_4.СебестоимостьEdit.set_edit_text("1")
        dlg_spec_4.OKButton.click()

    with allure.step("Добавить альбом и нажаьб сохранить товар"):
        # dlg_spec_3.ЦеныCustom2.click_input()
        dlg_spec_3.ИзменитьButton.click()
        dlg_spec_5.Static2.click_input()
        dlg_spec_5.Принять2.click_input()
        dlg_spec_3.СохранитьButton.click()
        dlg_spec_3.OKButton.wait("visible")
        dlg_spec_3.OKButton.click()

    with allure.step("Изменить цену на товар, себестоимость выше цены продажи и нажать сохранить товар"):
        # print(pag.position())
        # pag.moveTo(950, 250)
        pag.leftClick(950, 250)
        dlg_spec_4.ЦенаEdit.set_edit_text("1")
        dlg_spec_4.СебестоимостьEdit.set_edit_text("2")
        dlg_spec_4.OKButton.click()
        dlg_spec_3.СохранитьButton.click()
        dlg_spec_3.OKButton.wait("visible")
        dlg_spec_3.OKButton.click()

    with allure.step("Изменить цену на товар, в себестоимость ввести текст и нажать сохранить товар"):
        pag.leftClick(950, 250)
        dlg_spec_4.ЦенаEdit.set_edit_text("1")
        dlg_spec_4.СебестоимостьEdit.set_edit_text("fdss")
        dlg_spec_4.OKButton.click()
        dlg_spec_3.СохранитьButton.click()
        dlg_spec_3.OKButton.wait("visible")
        dlg_spec_3.OKButton.click()

    with allure.step("Изменить цену на товар, в цену продажи ввести текст, в себестоимость валидное значение и нажать "
                     "сохранить товар"):
        pag.leftClick(950, 250)
        dlg_spec_4.ЦенаEdit.set_edit_text("fdss")
        dlg_spec_4.СебестоимостьEdit.set_edit_text("1")
        dlg_spec_4.OKButton.click()
        dlg_spec_3.СохранитьButton.click()
        dlg_spec_3.OKButton.wait("visible")
        dlg_spec_3.OKButton.click()

    with allure.step("Изменить цену на товар, в себестоимость ввести отрицательное значение и нажать сохранить товар"):
        pag.leftClick(950, 250)
        dlg_spec_4.ЦенаEdit.set_edit_text("3")
        dlg_spec_4.СебестоимостьEdit.set_edit_text("-1")
        dlg_spec_4.OKButton.click()
        dlg_spec_3.СохранитьButton.click()
        dlg_spec_3.OKButton.wait("visible")
        dlg_spec_3.OKButton.click()

    with allure.step("Изменить цену на товар, в цена продажи ввести отрицательное значение и нажать сохранить товар"):
        pag.leftClick(950, 250)
        dlg_spec_4.ЦенаEdit.set_edit_text("-1")
        dlg_spec_4.СебестоимостьEdit.set_edit_text("3")
        dlg_spec_4.OKButton.click()
        dlg_spec_3.СохранитьButton.click()
        dlg_spec_3.OKButton.wait("visible")
        dlg_spec_3.OKButton.click()

    with allure.step("Изменить цену на товар, в цена продажи и себестоимость ввести спецсимволы и нажать сохранить "
                     "товар"):
        pag.leftClick(950, 250)
        dlg_spec_4.ЦенаEdit.set_edit_text("!№;%:::?")
        dlg_spec_4.СебестоимостьEdit.set_edit_text("!№;%:::?")
        dlg_spec_4.OKButton.click()
        dlg_spec_3.СохранитьButton.click()
        dlg_spec_3.OKButton.wait("visible")
        dlg_spec_3.OKButton.click()

    with allure.step("Изменить цену на товар, в цена продажи ввести 0 и нажать сохранить "
                     "товар"):
        pag.leftClick(950, 250)
        dlg_spec_4.ЦенаEdit.set_edit_text("0")
        dlg_spec_4.СебестоимостьEdit.set_edit_text("1")
        dlg_spec_4.OKButton.click()
        dlg_spec_3.СохранитьButton.click()
        dlg_spec_3.OKButton.wait("visible")
        dlg_spec_3.OKButton.click()

    with allure.step("Изменить цену на товар, в себестоимость ввести 0 и нажать сохранить "
                     "товар"):
        pag.leftClick(950, 250)
        dlg_spec_4.ЦенаEdit.set_edit_text("1")
        dlg_spec_4.СебестоимостьEdit.set_edit_text("0")
        dlg_spec_4.OKButton.click()
        dlg_spec_3.СохранитьButton.click()
        dlg_spec_3.OKButton.wait("visible")
        dlg_spec_3.OKButton.click()
    # dlg_spec_3.print_control_identifiers()

    # time.sleep(8)


    # time.sleep(10)
    # send_keys('{DOWN}')
    # dlg_spec_3.child_window(auto_id="Name", control_type="Edit").set_text("Gfhjkm")
    # dlg_spec_3.child_window(auto_id="Price", control_type="Edit").set_text("12")
    # dlg_spec_3.child_window(auto_id="CostPrice", control_type="Edit").set_text("65")
    # dlg_spec_3.child_window(title="Создать", auto_id="CreateButton", control_type="Button").click()
    # dlg_spec_3.ОКButton.click()


    # time.sleep(2)

    # dlg_spec_3.print_control_identifiers()
    # dlg_spec_4.print_control_identifiers()
    # app.Untitled.print_control_identifiers()

    # Dialog - ''(L713, T392, R902, B532)
    # | ['Цена продажиDialog', 'Dialog2']
    # | |
    # | | Button - 'ОК'(L804, T491, R879, B514)
    # | | ['Button', 'ОК', 'ОКButton', 'Button0', 'Button1']
    # | | child_window(title="ОК", auto_id="2", control_type="Button")
    # | |
    # | | Static - 'Некорректный ввод цен'(L732, T446, R864, B461)
    # | | ['Некорректный ввод цен', 'Некорректный ввод ценStatic', 'Static', 'Static0', 'Static1']
    # | | child_window(title="Некорректный ввод цен", auto_id="65535", control_type="Text")
    # | |
    # | | TitleBar - ''(L721, T395, R894, B423)
    # | | ['TitleBar', 'Цена продажиTitleBar', 'TitleBar0', 'TitleBar1']
    # | | |
    # | | | Button - 'Закрыть'(L861, T393, R895, B423)
    # | | | ['Button2', 'Закрыть', 'ЗакрытьButton', 'Закрыть0', 'Закрыть1', 'ЗакрытьButton0', 'ЗакрытьButton1']
    # | | | child_window(title="Закрыть", control_type="Button")
    # |

    # Pane - ''(L431, T210, R1399, B839)
    # | | | ['Pane2', 'НазваниеPane']
    # | | | |
    # | | | | Static - 'Название'(L441, T220, R1253, B236)
    # | | | | ['Название', 'НазваниеStatic', 'Static259']
    # | | | | child_window(title="Название", control_type="Text")
    # | | | |
    # | | | | Edit - ''(L441, T246, R1253, B264)
    # | | | | ['НазваниеEdit', 'Edit', 'Edit0', 'Edit1']
    # | | | | child_window(auto_id="Name", control_type="Edit")
    # | | | |
    # | | | | Static - 'Цена продажи'(L441, T284, R629, B300)
    # | | | | ['Цена продажиStatic', 'Static260', 'Цена продажи']
    # | | | | child_window(title="Цена продажи", control_type="Text")
    # | | | |
    # | | | | Edit - ''(L441, T310, R629, B328)
    # | | | | ['Edit2', 'Цена продажиEdit']
    # | | | | child_window(auto_id="Price", control_type="Edit")
    # | | | |
    # | | | | Static - 'Цена закупки'(L857, T284, R1045, B300)
    # | | | | ['Static261', 'Цена закупки', 'Цена закупкиStatic']
    # | | | | child_window(title="Цена закупки", control_type="Text")
    # | | | |
    # | | | | Edit - ''(L857, T310, R1045, B328)
    # | | | | ['Edit3', 'Цена закупкиEdit']
    # | | | | child_window(auto_id="CostPrice", control_type="Edit")
    # | | | |
    # | | | | Static - 'Альбомы'(L441, T348, R1253, B368)
    # | | | | ['Альбомы', 'Static262', 'АльбомыStatic']
    # | | | | child_window(title="Альбомы", control_type="Text")
    # | | | |
    # | | | | Button - 'Изменить'(L1268, T343, R1335, B363)
    # | | | | ['Изменить', 'Button262', 'ИзменитьButton', 'Изменить0', 'Изменить1']
    # | | | | child_window(title="Изменить", control_type="Button")
    # | | | | |
    # | | | | | Static - 'Изменить'(L1275, T345, R1328, B361)
    # | | | | | ['Изменить2', 'Static263', 'ИзменитьStatic']
    # | | | | | child_window(title="Изменить", control_type="Text")
    # | | | |
    # | | | | Static - 'КАЛЛЫ НА ВЕСНУ 2023'(L439, T376, R567, B392)
    # | | | | ['КАЛЛЫ НА ВЕСНУ 202321', 'Static264', 'КАЛЛЫ НА ВЕСНУ 2023Static11']
    # | | | | child_window(title="КАЛЛЫ НА ВЕСНУ 2023", control_type="Text")
    # | | | |
    # | | | | Static - 'Фотографии'(L441, T410, R1253, B426)
    # | | | | ['Фотографии', 'ФотографииStatic', 'Static265']
    # | | | | child_window(title="Фотографии", control_type="Text")
    # | | | |
    # | | | | Edit - ''(L441, T436, R1253, B454)
    # | | | | ['Edit4', 'ФотографииEdit']
    # | | | | child_window(auto_id="Source", control_type="Edit")
    # | | | |
    # | | | | Button - 'Применить'(L1268, T431, R1335, B459)
    # | | | | ['ПрименитьButton', 'Button263', 'Применить', 'Применить0', 'Применить1']
    # | | | | child_window(title="Применить", auto_id="ApplySource", control_type="Button")
    # | | | | |
    # | | | | | Static - 'Применить'(L1270, T437, R1333, B453)
    # | | | | | ['Static266', 'ПрименитьStatic', 'Применить2']
    # | | | | | child_window(title="Применить", control_type="Text")
    # | | | |
    # | | | | Pane - ''(L431, T464, R1263, B807)
    # | | | | ['Pane3', 'ФотографииPane']
    # | | | | |
    # | | | | | ScrollBar - ''(L1246, T464, R1263, B807)
    # | | | | | ['ФотографииScrollBar', 'ScrollBar2']
    # | | | | | child_window(auto_id="VerticalScrollBar", control_type="ScrollBar")
    # | | | | | |
    # | | | | | | Button - ''(L1246, T464, R1263, B481)
    # | | | | | | ['Button264', 'ФотографииButton', 'ФотографииButton0', 'ФотографииButton1']
    # | | | | | | child_window(auto_id="PART_LineUpButton", control_type="Button")
    # | | | | | |
    # | | | | | | Button - ''(L1246, T790, R1263, B807)
    # | | | | | | ['Button265', 'ФотографииButton2']
    # | | | | | | child_window(auto_id="PART_LineDownButton", control_type="Button")
    # | | | |
    # | | | | Button - 'Создать'(L1345, T812, R1394, B834)
    # | | | | ['Создать', 'СоздатьButton', 'Button266', 'Создать0', 'Создать1']
    # | | | | child_window(title="Создать", auto_id="CreateButton", control_type="Button")
    # | | | | |
    # | | | | | Static - 'Создать'(L1348, T815, R1391, B831)
    # | | | | | ['СоздатьStatic', 'Создать2', 'Static267']
    # | | | | | child_window(title="Создать", control_type="Text")
    # | |
    # | | TabItem - '+'(L210, T181, R237, B207)
    # | | ['+', '+TabItem', 'TabItem2', '+0', '+1']
    # | | child_window(title="+", control_type="TabItem")
    # | | |
    # | | | Static - '+'(L219, T186, R228, B202)
    # | | | ['+2', 'Static268', '+Static']
    # | | | child_window(title="+", control_type="Text")

    app.kill()
