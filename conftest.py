import pytest
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


@pytest.fixture()
def driver():
    driver_service = Service(ChromeDriverManager(version="114.0.5735.90").install())
    driver = webdriver.Chrome(service=driver_service)
    driver.maximize_window()
    yield driver
    driver.quit()
