import os
import time

import keyboard
from PIL import Image
from selenium.common import NoSuchElementException
from selenium.webdriver import ActionChains, Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait as Wait



def send_keys_from_keyboard(key):
    keyboard.send(key)


class BasePage:
    def __init__(self, driver, url):
        self.driver = driver
        self.url = url

    def open_page(self):
        self.driver.get(self.url)

    def element_is_visible(self, locator, timeout=5):
        return Wait(self.driver, timeout).until(EC.visibility_of_element_located(locator))

    def element_are_visible(self, locator, timeout=10):
        return Wait(self.driver, timeout).until(EC.visibility_of_all_elements_located(locator))

    def enter_data_in_authorization_field(self, input_field_locator, input_data, button_locator, button_status):
        input_field = (By.XPATH, input_field_locator)
        button = (By.XPATH, button_locator)
        data = self.element_is_visible(input_field)
        data.clear()
        data.send_keys(input_data)
        button = self.element_is_visible(button)
        assert button.is_enabled() == button_status

    def click_button(self, button_locator, element_index=0):
        element = self.driver.find_elements(By.XPATH, button_locator)
        Wait(self.driver, 10).until(EC.element_to_be_clickable((element[element_index]))).click()

    def double_click_button(self, button_locator, element_number=0):
        # element = (By.XPATH, button_locator)
        element = self.driver.find_elements(By.XPATH, button_locator)
        # self.element_is_visible(element).double_click()
        action = ActionChains(self.driver)
        action.double_click(element[element_number]).perform()

    # def click_button(self, button_locator, button_index=0):
    #     elements = self.driver.find_elements(By.XPATH, button_locator)
    #     elements[button_index].click()

    def verify_element_is_visible(self, locator):
        element = (By.XPATH, locator)
        self.element_is_visible(element)

    def verify_elements_are_visible(self, locator):
        elements = (By.XPATH, locator)
        self.element_are_visible(elements)

    def verify_element_attribute(self, locator, attribute_name, name):
        element = (By.XPATH, locator)
        attribute = self.element_is_visible(element).get_attribute(attribute_name)
        # print(attribute)
        assert attribute == name

    def verify_some_elements_attribute(self, locator, attribute_name, name, element_number):
        element = self.driver.find_elements(By.XPATH, locator)
        attribute = element[element_number].get_attribute(attribute_name)
        # print(attribute)
        assert attribute == name

    def verify_element_text(self, text_element_locator, element_text):
        element = (By.XPATH, text_element_locator)
        locator_text = self.element_is_visible(element).text
        # print(element_text)
        # print(locator_text)
        assert locator_text == element_text

    def verify_element_number_text(self, text_element_locator, element_text, element_number=0):
        locators = self.driver.find_elements(By.XPATH, text_element_locator)
        # print(element_text)
        # print(locators[element_number].text)
        assert locators[element_number].text == element_text

    def verify_elements_text(self, text_elements_locator, element_text):
        elements = self.driver.find_elements(By.XPATH, text_elements_locator)
        for i in range(len(elements)):
            assert elements[i].text == element_text

    def get_elements_text(self, text_elements_locator, element_number):
        elements = self.driver.find_elements(By.XPATH, text_elements_locator)
        element_text = elements[element_number].text
        return element_text

    def switch_window(self, index):
        windows = self.driver.window_handles
        self.driver.switch_to.window(windows[index])

    def navigate_back_window(self):
        self.driver.back()

    def close_tab(self):
        self.driver.close()

    def verify_button_status(self, button_locator, button_status):
        element = (By.XPATH, button_locator)
        button = self.element_is_visible(element)
        # print(button.is_enabled())
        assert button.is_enabled() == button_status

    def move_mouse_to_element(self, element):
        element_locator = (By.XPATH, element)
        self.element_is_visible(element_locator)
        element_to_move = self.driver.find_element(By.XPATH, element)
        ActionChains(self.driver).move_to_element(element_to_move).perform()

    def move_mouse_some_elements(self, elements_locator, element_number=0):
        elements = (By.XPATH, elements_locator)
        self.element_is_visible(elements)
        element_to_move = self.driver.find_elements(By.XPATH, elements_locator)
        ActionChains(self.driver).move_to_element(element_to_move[element_number]).perform()

    def web_scrolled_into_view_elements(self, element_locator, element_number=0):
        locator = (By.XPATH, element_locator)
        self.element_are_visible(locator)
        elm = self.driver.find_elements(By.XPATH, element_locator)
        self.driver.execute_script("arguments[0].scrollIntoView();", elm[element_number])

    def check_exists_by_xpath(self, xpath):
        try:
            self.driver.find_element(By.XPATH, xpath)
        except NoSuchElementException:
            return False
        return True

    def verify_elements_length(self, locator, length):
        elements_length = len(self.driver.find_elements(By.XPATH, locator))
        # print(elements_length)
        assert elements_length == length

    def return_elements_length(self, locator):
        elements_length = len(self.driver.find_elements(By.XPATH, locator))
        return elements_length

    def enter_data_in_field(self, input_field_locator, input_data):
        element = (By.XPATH, input_field_locator)
        data = self.element_is_visible(element)
        data.send_keys(input_data)

    def enter_data_in_fields_without_clear(self, input_field_locator, input_data, element_number=0):
        elements = self.driver.find_elements(By.XPATH, input_field_locator)
        elements[element_number].send_keys(input_data)

    def enter_data_in_field_some_elements(self, input_field_locator, input_data, element_number=0):
        data = self.driver.find_elements(By.XPATH, input_field_locator)
        data[element_number].clear()
        data[element_number].send_keys(input_data)

    def upload_file(self, element_locator, upload_file_path):
        upload_button = self.driver.find_element(By.XPATH, element_locator)
        upload_button.send_keys(upload_file_path)

    def verify_elements_sorted(self, elements_locator, revers_sort=False):
        elements = self.driver.find_elements(By.XPATH, elements_locator)
        elements_array = []
        for i in range(0, len(elements)):
            element_text = elements[i].text
            elements_array.append(element_text)
        new_array = list(elements_array)
        new_array_sorted = sorted(new_array, reverse=revers_sort)
        # print(elements_array)
        # print(new_array_sorted)
        assert new_array_sorted == elements_array

    def verify_date_sorted(self, elements_locator, revers_sort=False):
        elements = self.driver.find_elements(By.XPATH, elements_locator)
        elements_array = []
        for i in range(0, len(elements)):
            element_text = elements[i].text
            elements_array.append(element_text)
        new_array = list(elements_array)
        new_array.sort(key=lambda x: time.mktime(time.strptime(x, "%d.%m.%Y")), reverse=revers_sort)
        # print(elements_array)
        # print(new_array)
        assert new_array == elements_array

    def verify_elements_not_sorted(self, elements_locator, revers_sort=False):
        elements = self.driver.find_elements(By.XPATH, elements_locator)
        elements_array = []
        for i in range(0, len(elements)):
            element_text = elements[i].text
            elements_array.append(element_text)
        new_array = list(elements_array)
        new_array_sorted = sorted(new_array, reverse=revers_sort)
        # print(new_array)
        # print(new_array_sorted)
        assert new_array_sorted != elements_array

    def enter_absolute_path(self, path):
        absolute_path = os.path.abspath(
            os.path.join(os.path.dirname(os.getcwd()), path))
        # print(absolute_path)
        keyboard.write(fr"{absolute_path}")
        keyboard.press('enter')

    def get_odd_and_even_elements_of_array(self, elements_locator, start_value, step_value):
        elements = self.driver.find_elements(By.XPATH, elements_locator)
        elements_array = []
        for i in range(start_value, len(elements), step_value):
            element_text = elements[i].text
            elements_array.append(element_text)
        return elements_array

    def verify_date_array_sorted(self, elements_array, revers_sort=False):
        new_array = list(elements_array)
        new_array.sort(key=lambda x: time.mktime(time.strptime(x, "%d.%m.%Y")), reverse=revers_sort)
        # print(elements_array)
        # print(new_array)
        assert len(new_array) != 0
        assert new_array == elements_array

    def drag_and_drop_element(self, source_element_locator, target_element_locator):
        source_element = self.driver.find_element(By.XPATH, source_element_locator)
        BasePage.move_mouse_to_element(self, source_element_locator)
        target_element_locator = self.driver.find_element(By.XPATH, target_element_locator)
        action = ActionChains(self.driver)
        action.drag_and_drop(source_element, target_element_locator).perform()

    # def take_screenshot_and_compare_image(self, expected_image_path, element_locator, element_number=0):
    #     screenshot_element = self.driver.find_elements(By.XPATH, element_locator)
    #     screenshot_element[element_number].screenshot(f'{TableWidgetData.SCREENSHOT_PATH}//screenshots//test_screenshot.png')
    #     image_1 = Image.open(expected_image_path)
    #     im1 = image_1.load()
    #     image_2 = Image.open(f'{TableWidgetData.SCREENSHOT_PATH}//screenshots//test_screenshot.png')
    #     im2 = image_2.load()
    #     i = 0
    #     if image_1.size == image_2.size:
    #         x1, y1 = image_1.size
    #
    #         for x in range(0, x1):
    #             for y in range(0, y1):
    #                 if im1[x, y] != im2[x, y]:
    #                     i = i + 1
    #         #             print(f'Координаты: x={x}, y={y} Изображение 1={im1[x, y]} - Изображение 2={im2[x, y]}')
    #         # print(f"Количество разных пикселей: {i}")
    #         assert i < 1000
    #     else:
    #         assert image_1.size == image_2.size

    def verify_dropdown_elements_by_attribute(self, elements_locator, attribute_name, dictionary):
        elements = self.driver.find_elements(By.XPATH, elements_locator)
        attribute = attribute_name
        for i in range(0, len(elements) - 1):
            attribute_name = elements[i].get_attribute(attribute)
            assert attribute_name == dictionary[i + 1]

    def click_and_hold(self, element_locator):
        element = self.driver.find_element(By.XPATH, element_locator)
        ActionChains(self.driver).click_and_hold(element).perform()

    def drag_and_drop_for_some_elements(self, source_element_locator, target_element_locator, source_el_number=0,
                                        target_el_number=0):
        source_elements = self.driver.find_elements(By.XPATH, source_element_locator)
        # self.move_mouse_to_element(source_element_locator[source_el_number])
        target_element_locator = self.driver.find_elements(By.XPATH, target_element_locator)
        action = ActionChains(self.driver)
        action.drag_and_drop(source_elements[source_el_number], target_element_locator[target_el_number]).perform()

    def drag_and_drop_element_hold_release(self, source_element_locator, target_element_locator,
                                           source_element_number=0, target_element_number=0):
        source = self.driver.find_elements(By.XPATH, source_element_locator)
        target = self.driver.find_elements(By.XPATH, target_element_locator)
        action = ActionChains(self.driver)
        action.click_and_hold(source[source_element_number]).move_to_element(target[target_element_number]) \
            .release(target[target_element_number]).perform()

    def drag_and_drop_changed_target_element(self, source_element_locator, target_element_locator,
                                             source_element_number=0, target_element_number=0):
        self.click_and_hold(source_element_locator)
        time.sleep(1.5)
        self.verify_element_is_visible(target_element_locator)
        self.drag_and_drop_element_hold_release(source_element_locator, target_element_locator,
                                                source_element_number, target_element_number)

    def clear_field_by_backspace(self, element_locator, symbol_quantity, element_number=0):
        element = self.driver.find_elements(By.XPATH, element_locator)
        for i in range(symbol_quantity):
            element[element_number].send_keys(Keys.BACKSPACE)
            time.sleep(0.1)

    def click_shadow_root_element(self, shadow_root_host_locator, shadow_root_element_locator_css, element_number=0):
        shadow_host = self.driver.find_element(By.XPATH, shadow_root_host_locator)
        shadow_root = shadow_host.shadow_root
        shadow_content = shadow_root.find_elements(By.CSS_SELECTOR, shadow_root_element_locator_css)
        time.sleep(0.5)
        shadow_content[element_number].click()

    def compare_all_array_elements(self, element_array, compare_mean):
        # print(element_array[0])
        for i in range(len(element_array)):
            assert element_array[i] == compare_mean

    def load_file(self, load_locator, file_path):
        file_input = self.driver.find_element(By.XPATH, load_locator)
        file_path = file_path
        file_input.send_keys(file_path)


