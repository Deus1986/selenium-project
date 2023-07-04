import time
import allure
from pages.variables_for_ok_bot import OKBotData, OkBotLocators
from pages.variables_for_posts import WebAddresses as WebAddresses, Locators
from pages.form_page import FormPage


class TestOkBotAcceptAlbum:
    allure.title("Test ok bot accept album")
    allure.severity(severity_level="blocker")

    def test_ok_bot_share(self, driver):
        with allure.step("Перейти на страницу одноклассников"):
            form_page = FormPage(driver, WebAddresses.OK_LOGIN_PAGE_ADDRESS)
            form_page.openpage()

        with allure.step("Ввести логин"):
            form_page.enter_login(OKBotData.LOGIN_DJORDJ_KIM)

        with allure.step("Ввести пароль"):
            form_page.enter_password()

        with allure.step("Нажать войти"):
            form_page.click_enter_button()

        with allure.step("Нажать на 'группы' в сайд баре"):
            form_page.click_group_in_side_bar(Locators.TOP_SIDE_NAVIGATION_BAR_LOCATORS_WITHOUT_XPATH)

        with allure.step("Нажать на группу 'Тестовая группа'"):
            form_page.click_group_name_in_side_bar(OkBotLocators.TEST_GROUP)

        with allure.step("Нажать на все для отображения альбомов группы"):
            form_page.click_element_by_number(OkBotLocators.BUTTON_PHOTOS_IN_TOP_MENU_SIDE_BAR, 2)
            time.sleep(2)

        with allure.step("Нажать показать еще 3 раза"):
            form_page.click_button(OkBotLocators.BUTTON_SEE_MORE)
            form_page.click_button(OkBotLocators.BUTTON_SEE_MORE)
            form_page.click_button(OkBotLocators.BUTTON_SEE_MORE)
            time.sleep(2)

        with allure.step("Получить количество альбомов 'Пионы на весну 2023'"):
            album_quantity = form_page.get_len(OkBotLocators.PION_ALBUMS_NAME)

        with allure.step("Нажать на сообщения в хедере страницы"):
            form_page.click_element_by_number(OkBotLocators.HEADER_LINKS, 0)

        with allure.step("Нажать на тестовую группу в сообщениях"):
            time.sleep(2)
            if form_page.check_exists_by_xpath(OkBotLocators.TEST_GROUP_MESSAGES_UNREAD):
                form_page.click_button(OkBotLocators.TEST_GROUP_MESSAGES_UNREAD)
                time.sleep(1)
                form_page.click_button(OkBotLocators.SCROLL_LAST_ELEMENT_BUTTON)
                time.sleep(1)
            else:
                form_page.click_element_by_number(OkBotLocators.TEST_GROUP_MESSAGES, 0)
                time.sleep(2)

        with allure.step("Нажать альбомы в сообщениях"):
            if form_page.check_exists_by_xpath(OkBotLocators.BUTTON_MAIN_MENU):
                form_page.move_to_last_element(OkBotLocators.BUTTON_MAIN_MENU)
                time.sleep(1)
                form_page.click_message_button(OkBotLocators.BUTTON_MAIN_MENU)
                time.sleep(2)
                if form_page.check_exists_by_xpath(OkBotLocators.SCROLL_LAST_ELEMENT_BUTTON):
                    form_page.click_button(OkBotLocators.SCROLL_LAST_ELEMENT_BUTTON)
                time.sleep(1)
                form_page.click_message_button(OkBotLocators.BUTTON_GROUP)
                time.sleep(3)
            else:
                form_page.click_message_button(OkBotLocators.BUTTON_GROUP)
                time.sleep(3)

        with allure.step("Нажать принять в сообщениях"):
            form_page.click_message_button(OkBotLocators.BUTTON_ACCEPT)
            time.sleep(2)
            form_page.click_button(OkBotLocators.SCROLL_LAST_ELEMENT_BUTTON)
            time.sleep(1)

        with allure.step("Выбрать аккаунт, с которого импортировать - Джордж Ким"):
            form_page.click_message_button(OkBotLocators.BUTTON_DJORDJ_KIM_ACCOUNT)
            time.sleep(2)

        with allure.step("Выбрать альбом пионы на весну в сообщениях"):
            form_page.click_message_button(OkBotLocators.BUTTON_PIONS_ON_SPRING)
            time.sleep(5)

        with allure.step("Закрыть окно сообщений"):
            form_page.click_message_button(OkBotLocators.BUTTON_CLOSE_MESSAGES_FORM)
            form_page.driver.refresh()
            time.sleep(2)

        with allure.step("Нажать показать еще 3 раза"):
            form_page.click_button(OkBotLocators.BUTTON_SEE_MORE)
            form_page.click_button(OkBotLocators.BUTTON_SEE_MORE)
            form_page.click_button(OkBotLocators.BUTTON_SEE_MORE)
            time.sleep(2)

        with allure.step("Проверить, что альбом добавился в тестовую группу"):
            album_quantity_after_add_album = form_page.get_len(OkBotLocators.PION_ALBUMS_NAME)
            assert (album_quantity + 1) == album_quantity_after_add_album
