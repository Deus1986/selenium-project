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

allure.title("Own public group wall post")
allure.severity(severity_level="blocker")


def test_own_public_group_post():
    with allure.step("Выполнить запрос login для получения токена авторизации"):
        response = requests.get(RequestsVariables.baseUrl + "/api/Employee/login/" + RequestsVariables.password)
        assert response.status_code == 200

    response_json = response.json()
    authorization_token = response_json.get('token')
    headers = {"Authorization": "Bearer " + authorization_token}
    time_now = str(time.time())
    description = OKData.callaCaptinVenturaDescription
    body = [
        {
            "post": {
                "description": description + time_now,
                "photos": [
                    OKData.callaCaptinVenturaPhoto
                ]
            },
            "users": [],
            "groupIdsByUserId": {
                OKData.userId: [
                    OKData.groupGobiguliId
                ]
            }
        }
    ]

    with allure.step("Выполнить публикацию на стену группы"):
        postWallGroup = requests.post(RequestsVariables.baseUrl +"/api/Posts/CreatePosts", headers=headers, json=body)
        assert postWallGroup.status_code == 200
        time.sleep(60)

    driver = webdriver.Chrome(Pathes.webDriverChromeLocalPath)
    with allure.step("Перейти на страницу одноклассников"):
        driver.get(WebAddresses.okLoginPageAddress)

    with allure.step("Ввести логин"):
        email_field = driver.find_element(By.XPATH, Locators.loginField)
        email_field.send_keys(OKData.loginSeric)

    with allure.step("Ввести пароль"):
        password_field = driver.find_element(By.XPATH, Locators.passwordField)
        password_field.send_keys(OKData.passwordSeric)

    with allure.step("Нажать войти"):
        enter_account_button = driver.find_element(By.XPATH, Locators.enterOKButton)
        enter_account_button.click()

    with allure.step("Нажать на 'группы' в сайд баре"):
        top_side_navigation_bar = driver.find_elements(By.XPATH, Locators.topSideNavigationBarLocators)
        side_bar_account_name_link = top_side_navigation_bar[4]
        side_bar_account_name_link.click()
        time.sleep(1)

    with allure.step("Нажать на группу 'Гобигули'"):
        flowers_our_flowers = driver.find_element(By.XPATH, Locators.gobiguliLocator)
        flowers_our_flowers.click()
        time.sleep(1)

    with allure.step("Проверить, что пост в собственную приватную группу успешно опубликован"):
        posts_on_the_wall = driver.find_elements(By.XPATH, Locators.lastPost)
        last_post_text = posts_on_the_wall[0].find_element(By.XPATH, Locators.lastPostContent).text
        assert last_post_text == description + time_now