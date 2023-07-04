import time
import allure
from pages.variables_for_ok_bot import OKBotData, OkBotLocators
from pages.variables_for_posts import WebAddresses as WebAddresses
from pages.form_page import FormPage


class TestOkBotShare:
    allure.title("Test ok bot share")
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

        with allure.step("Нажать на тестовую группу в сообщениях"):
            form_page.click_element_by_number(OkBotLocators.TEST_GROUP_MESSAGES, 0)
            form_page.driver.refresh()
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

        with allure.step("Нажать поделиться в сообщениях"):
            form_page.click_message_button(OkBotLocators.BUTTON_SHARE)
            time.sleep(2)
            form_page.click_button(OkBotLocators.SCROLL_LAST_ELEMENT_BUTTON)
            time.sleep(1)

        with allure.step("Выбрать альбом пионы на весну в сообщениях"):
            form_page.click_message_button(OkBotLocators.BUTTON_PIONS_ON_SPRING)
            time.sleep(2)

        with allure.step("Нажать очистить"):
            form_page.click_message_button(OkBotLocators.BUTTON_CLEAR)
            time.sleep(2)

        with allure.step("Выбрать альбом пионы на весну в сообщениях"):
            form_page.click_message_button(OkBotLocators.BUTTON_PIONS_ON_SPRING)
            time.sleep(2)

        with allure.step("Нажать отправить"):
            elements_len = form_page.get_len(OkBotLocators.URL_ELEMENTS)
            form_page.click_message_button(OkBotLocators.BUTTON_SEND)
            time.sleep(20)
            elements_len_after = form_page.get_len(OkBotLocators.URL_ELEMENTS)
            assert (elements_len_after - elements_len) == 5
