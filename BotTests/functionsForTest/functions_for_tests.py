import time
from selenium import webdriver
from selenium.webdriver.common.by import By
import requests
import allure
from BotTests.variablesForTests.variables_for_posts import RequestsVariables as RequestsVariables
from BotTests.variablesForTests.variables_for_posts import OKData as OKData
from BotTests.variablesForTests.variables_for_posts import WebAddresses as WebAddresses
from BotTests.variablesForTests.variables_for_posts import Pathes as Pathes
from BotTests.variablesForTests.variables_for_posts import Locators as Locators

driver = webdriver.Chrome(Pathes.webDriverChromeLocalPath)
def user_authorizaion(login, loginInputLocator, password, passwordInputLocator):
    with allure.step("Ввести логин"):
        email_field = driver.find_element(By.XPATH, loginInputLocator)
        email_field.send_keys(login)

    with allure.step("Ввести пароль"):
        password_field = driver.find_element(By.XPATH, passwordInputLocator)
        password_field.send_keys(password)

    with allure.step("Нажать войти"):
        enter_account_button = driver.find_element(By.XPATH, Locators.enterOKButton)
        enter_account_button.click()
def open_OK_page():
    with allure.step("Перейти на страницу одноклассников"):
        driver.get(WebAddresses.okLoginPageAddress)
def get_api_authorization_token(baseUrl, password):
    with allure.step("Выполнить запрос login для получения токена авторизации"):
        response = requests.get(RequestsVariables.baseUrl + "/api/Employee/login/" + RequestsVariables.password)
        assert response.status_code == 200
        response_json = response.json()
        authorization_token = response_json.get('token')
        return authorization_token
def api_create_post(token, baseUrl, flowerDescription, flowerPhoto, userId, groupId):
    headers = {"Authorization": "Bearer " + token}
    time_now = str(time.time())
    description = flowerDescription
    body = [
        {
            "post": {
                "description": description + time_now,
                "photos": [
                    flowerPhoto
                ]
            },
            "users": [],
            "groupIdsByUserId": {
                userId: [
                    groupId
                ]
            }
        }
    ]

    with allure.step("Выполнить публикацию на стену группы"):
        postWallGroup = requests.post(baseUrl + "/api/Posts/CreatePosts", headers=headers, json=body)
        assert postWallGroup.status_code == 200
        time.sleep(60)
    return time_now
def click_on_the_top_side_bar_element(topSideBarElementLocator, elementNumber):
    top_side_navigation_bar = driver.find_elements(By.XPATH, topSideBarElementLocator)
    side_bar_account_name_link = top_side_navigation_bar[elementNumber]
    side_bar_account_name_link.click()
    time.sleep(1)

def click_group_by_locator(groupLocator):
    with allure.step("Нажать на группу 'Цветочки наши цветочки'"):
        flowers_our_flowers = driver.find_element(By.XPATH, groupLocator)
        flowers_our_flowers.click()
        time.sleep(1)
def check_post_publication_in_group(lastPostLocator, lastPostContentLocator, flowerDescription, timeNow):
    with allure.step("Проверить, что пост в собственную публичную группу успешно опубликован"):
        posts_on_the_wall = driver.find_elements(By.XPATH, lastPostLocator)
        last_post_text = posts_on_the_wall[0].find_element(By.XPATH, lastPostContentLocator).text
        assert last_post_text == flowerDescription + timeNow