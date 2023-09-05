import time
import allure
from pages.variables_for_ok_bot import OKBotData, OkBotLocators
from pages.variables_for_posts import WebAddresses as WebAddresses, Locators
from pages.form_page import FormPage


class TestOkBotSendAccountGroup:
    allure.title("Test ok bot send in account group")
    allure.severity(severity_level="blocker")

    def test_ok_bot_send_album_in_account(self, driver):
        driver.implicitly_wait(10)
        with allure.step("Перейти на страницу одноклассников"):
            form_page = FormPage(driver, WebAddresses.OK_LOGIN_PAGE_ADDRESS)
            form_page.openpage()

        with allure.step("Ввести логин"):
            form_page.enter_login(OKBotData.LOGIN_DJORDJ_KIM)

        with allure.step("Ввести пароль"):
            form_page.enter_password()

        with allure.step("Нажать войти"):
            form_page.click_enter_button()

        with allure.step("Нажать на 'фото' в сайд баре"):
            form_page.click_element_by_number(Locators.TOP_SIDE_NAVIGATION_BAR_LOCATORS_WITHOUT_XPATH, 3)
            form_page.verify_page_opened(OkBotLocators.ACCOUNT_PHOTO_CONTEXT_MENUS)

        with allure.step("Проверить есть ли альбом 'Нарциссы на весну 2023' и удалить при наличии"):
            if form_page.check_exists_by_xpath(OkBotLocators.ACCOUNT_ALBUM_NARCISES):
                form_page.click_button(OkBotLocators.ACCOUNT_ALBUM_NARCISES)
                form_page.click_button(OkBotLocators.ACCOUNT_PHOTO_CONTEXT_MENUS_EDIT)
                form_page.click_button(OkBotLocators.DELETE_BUTTON_ACCOUNT_GROUP_2)
                form_page.click_button(OkBotLocators.ACCOUNT_PHOTO_DELETE_ALBUM_CONFIRM_BUTTON)

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
            form_page.click_message_button(OkBotLocators.BUTTON_SEND)
            time.sleep(2)
            form_page.click_button(OkBotLocators.SCROLL_LAST_ELEMENT_BUTTON)
            time.sleep(1)

        with allure.step("Выбрать аккаунт, с которого импортировать - Джордж Ким"):
            form_page.click_message_button(OkBotLocators.BUTTON_DJORDJ_KIM_ACCOUNT)
            time.sleep(2)

        with allure.step("Выбрать альбом нарциссы на весну в сообщениях"):
            form_page.click_message_button(OkBotLocators.BUTTON_NARCISES_ON_SPRING)
            time.sleep(50)

        with allure.step("Закрыть окно сообщений"):
            form_page.click_message_button(OkBotLocators.BUTTON_CLOSE_MESSAGES_FORM)
            form_page.driver.refresh()
            time.sleep(2)

        with allure.step("Проверить, что альбом добавился в аккаунт"):
            result = form_page.check_exists_by_xpath(OkBotLocators.ACCOUNT_ALBUM_NARCISES)
            assert result == True

        with allure.step("Удалить добавленный альбом"):
            form_page.click_button(OkBotLocators.ACCOUNT_ALBUM_NARCISES)
            form_page.click_button(OkBotLocators.ACCOUNT_PHOTO_CONTEXT_MENUS_EDIT)
            form_page.click_button(OkBotLocators.DELETE_BUTTON_ACCOUNT_GROUP_2)
            form_page.click_button(OkBotLocators.ACCOUNT_PHOTO_DELETE_ALBUM_CONFIRM_BUTTON)
