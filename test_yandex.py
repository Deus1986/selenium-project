import time

import allure
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
allure.title("Google search")
allure.severity(severity_level="blocker")
def test_yandex():
    driver = webdriver.Chrome("E://Selenium//chromedriver.exe")
    with allure.step("Открыть гугл"):
        driver.get("https://www.google.ru/")
        time.sleep(2)
    search_input = driver.find_element(By.XPATH, '//input[@class = "gLFyf"]')
    search_input.send_keys("market.yandex.ru")
    time.sleep(1)
    search_button = driver.find_element(By.XPATH, '//div[@class= "lJ9FBc"]//input[@value= "Поиск в Google"]')
    search_button.is_displayed()
    search_button.click()
    time.sleep(1)
    search_result = driver.find_elements(By.XPATH, '//div[@class= "MjjYud"]')
    assert len(search_result) > 6
    link = search_result[0].find_element(By.XPATH, '//h3[@class= "LC20lb MBeuO DKV0Md"]')
    link.click()
    driver.switch_to.window(driver.window_handles[1])
    time.sleep(10)
    assert driver.title == 'Интернет-магазин Яндекс Маркет — покупки с быстрой доставкой'