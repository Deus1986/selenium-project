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

allure.title("Public private wall post")
allure.severity(severity_level="blocker")
def test_post_private_wall():

    with allure.step("Выполнить запрос login для получения токена авторизации"):
        response = requests.get(RequestsVariables.baseUrl + "/api/Employee/login/" + RequestsVariables.password)
        assert response.status_code == 200


    response_json = response.json()
    authorization_token = response_json.get('token')
    headers = {"Authorization": "Bearer " + authorization_token}
    time_now = str(time.time())
    description = OKData.callaHameleonDescription
    body = [
        {
            "post": {
                "description": description + time_now,
                "photos": [
                    OKData.callaHameleonPhoto
                ]
            },
            "users": [
                {
                    "id": OKData.userId,
                    "publicationTime": None
                }
            ],
            "groupIdsByUserId": {}
        }
    ]

    with allure.step("Выполнить публикацию на стену аккаунта"):
        postWall = requests.post(RequestsVariables.baseUrl + "/api/Posts/CreatePosts", headers=headers, json=body)
        assert postWall.status_code == 200
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

    with allure.step("Нажать на имя пользователя в сайд баре"):
        top_side_navigation_bar = driver.find_elements(By.XPATH, Locators.topSideNavigationBarLocators)
        side_bar_account_name_link = top_side_navigation_bar[0]
        side_bar_account_name_link.click()
        time.sleep(1)

    with allure.step("Проверить, что пост успешно опубликован"):
        posts_on_the_wall = driver.find_elements(By.XPATH, Locators.lastPost)
        last_post_text = posts_on_the_wall[0].find_element(By.XPATH, Locators.lastPostContent).text
        assert last_post_text == description + time_now