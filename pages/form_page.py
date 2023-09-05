from selenium.common import NoSuchElementException
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By

from pages.base_page import BasePage
from pages.variables_for_posts import Locators
from pages.variables_for_posts import OKData


class FormPage(BasePage):
    def enter_seric_login(self):
        login = OKData.LOGIN_SERIC
        self.element_is_visible(Locators.LOGIN_FIELD).send_keys(login)

    def enter_login(self, login):
        self.element_is_visible(Locators.LOGIN_FIELD).send_keys(login)

    def enter_password(self):
        password = OKData.PASSWORD_SERIC
        self.element_is_visible(Locators.PASSWORD_FIELD).send_keys(password)

    def click_enter_button(self):
        self.element_is_visible(Locators.ENTER_OK_BUTTON).click()

    def click_button(self, button_locator):
        element = (By.XPATH, button_locator)
        self.element_is_visible(element).click()

    def verify_page_opened(self, button_locator):
        element = (By.XPATH, button_locator)
        self.element_is_visible(element)

    def click_element_by_number(self, locator, number):
        element = (By.XPATH, locator)
        self.element_are_visible(element)
        elements = self.driver.find_elements(By.XPATH, locator)
        elements[number].click()

    def click_user_name_in_side_bar(self):
        top_side_navigation_bar = self.element_are_visible(Locators.TOP_SIDE_NAVIGATION_BAR_LOCATORS)
        top_side_navigation_bar[0].click()

    def check_post_publication(self, page_locator, description, time_now, last_post_text_locator):
        self.element_are_visible(page_locator)
        last_post = self.driver.find_elements(By.XPATH, Locators.LAST_POST)
        last_post_text = last_post[0].find_element(By.XPATH, last_post_text_locator).text
        print(last_post_text)
        assert last_post_text == description + time_now

    def click_group_in_side_bar(self, top_side_navigation_bar_locators):
        top_side_navigation_bar = self.driver.find_elements(By.XPATH, top_side_navigation_bar_locators)
        side_bar_account_name_link = top_side_navigation_bar[4]
        side_bar_account_name_link.click()

    def click_group_name_in_side_bar(self, group_name):
        name_group_locator = self.element_is_visible(group_name)
        name_group_locator.click()

    def click_last_post_photo(self, page_locator):
        self.element_are_visible(page_locator)
        photo_posts = self.driver.find_elements(By.XPATH, Locators.LAST_POST_PHOTO)
        photo_posts[0].click()

    def verify_last_post_text(self, description, time_now):
        self.element_is_visible(Locators.ADVERSTISING_PAGE)
        posts_text = self.driver.find_element(By.XPATH, Locators.POST_TEXT).text
        print(posts_text)
        assert posts_text == description + time_now

    def verify_comment_publication(self, comment, time_now):
        all_comments = self.driver.find_elements(By.XPATH, Locators.ALL_COMMENTS)
        print(all_comments[0].text)
        assert all_comments[0].text == comment + time_now

    def verify_comment_author(self, user_name):
        all_author_name = self.driver.find_elements(By.XPATH, Locators.ALL_COMMENTS_AUTHOR_NAMES)
        print(all_author_name[0].text)
        assert all_author_name[0].text == user_name

    def click_message_button(self, message_group_locator):
        # elements = (By.XPATH, message_group_locator)
        # self.element_are_visible(elements)
        message_group = self.driver.find_elements(By.XPATH, message_group_locator)
        message_group_len = len(message_group)
        message_group[message_group_len - 1].click()

    def get_len(self, locator):
        elements = self.driver.find_elements(By.XPATH, locator)
        elements_len = len(elements)
        return elements_len

    def move_to_last_element(self, locator):
        elements = self.driver.find_elements(By.XPATH, locator)
        elements_len = len(elements)
        # print(elements_len)
        action = ActionChains(self.driver)
        action.move_to_element(elements[elements_len - 1])
        action.perform()

    def check_exists_by_xpath(self, locator):
        try:
            self.driver.find_element(By.XPATH, locator)
        except NoSuchElementException:
            return False
        return True
