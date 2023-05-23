from selenium.webdriver.common.by import By

from pages.base_page import BasePage
from BotTests.variablesForTests.variables_for_posts import Locators
from BotTests.variablesForTests.variables_for_posts import OKData


class FormPage(BasePage):
    def enter_login(self):
        login = OKData.LOGIN_SERIC
        self.element_is_visible(Locators.LOGIN_FIELD).send_keys(login)

    def enter_password(self):
        password = OKData.PASSWORD_SERIC
        self.element_is_visible(Locators.PASSWORD_FIELD).send_keys(password)

    def click_enter_button(self):
        self.element_is_visible(Locators.ENTER_OK_BUTTON).click()

    def click_user_name_in_side_bar(self):
        top_side_navigation_bar = self.element_are_visible(Locators.TOP_SIDE_NAVIGATION_BAR_LOCATORS)
        top_side_navigation_bar[0].click()

    def check_post_publication_in_privat_wall(self, description, time_now):
        self.element_are_visible(Locators.PRIVAT_WALL)
        last_post_text = self.driver.find_elements(By.XPATH, Locators.LAST_POST)
        a = last_post_text[0].find_element(By.XPATH, '//div[@class = "media-text_cnt_tx emoji-tx textWrap"]').text
        assert a == description + time_now

    def click_group_name_in_side_bar(self, top_side_navigation_bar_locators):
        top_side_navigation_bar = self.driver.find_elements(By.XPATH, top_side_navigation_bar_locators)
        side_bar_account_name_link = top_side_navigation_bar[4]
        side_bar_account_name_link.click()
