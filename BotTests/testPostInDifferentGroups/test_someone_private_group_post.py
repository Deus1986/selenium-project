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

allure.title("Someone private group wall post")
allure.severity(severity_level="blocker")


def test_someone_private_group_post():
    with allure.step("Выполнить запрос login для получения токена авторизации"):
        response = requests.get(RequestsVariables.BASE_URL + "/api/Employee/login/" + RequestsVariables.PASSWORD)
        assert response.status_code == 200

    response_json = response.json()
    authorization_token = response_json.get('token')
    headers = {"Authorization": "Bearer " + authorization_token}
    time_now = str(time.time())
    description = OKData.callaVermeerDescription
    body = [
        {
            "post": {
                "description": description + time_now,
                "photos": [
                    OKData.callaVermeerPhoto
                ]
            },
            "users": [],
            "groupIdsByUserId": {
                OKData.USER_ID: [
                    OKData.groupTratatuliId
                ]
            }
        }
    ]

    with allure.step("Выполнить публикацию на стену группы"):
        postWallGroup = requests.post(RequestsVariables.BASE_URL + "/api/Posts/CreatePosts", headers=headers, json=body)
        assert postWallGroup.status_code == 200
        time.sleep(60)

    driver = webdriver.Chrome(Pathes.webDriverChromeLocalPath)
    with allure.step("Перейти на страницу одноклассников"):
        driver.get(WebAddresses.OK_LOGIN_PAGE_ADDRESS)

    with allure.step("Ввести логин"):
        email_field = driver.find_element(By.XPATH, Locators.loginField)
        email_field.send_keys(OKData.loginSeric)

    with allure.step("Ввести пароль"):
        password_field = driver.find_element(By.XPATH, Locators.PASSWORD_FIELD)
        password_field.send_keys(OKData.PASSWORD_SERIC)

    with allure.step("Нажать войти"):
        enter_account_button = driver.find_element(By.XPATH, Locators.ENTER_OK_BUTTON)
        enter_account_button.click()

    with allure.step("Нажать на 'группы' в сайд баре"):
        top_side_navigation_bar = driver.find_elements(By.XPATH, Locators.TOP_SIDE_NAVIGATION_BAR_LOCATORS)
        side_bar_account_name_link = top_side_navigation_bar[4]
        side_bar_account_name_link.click()
        time.sleep(0.5)

    with allure.step("Нажать на группу 'Трататули'"):
        flowers_our_flowers = driver.find_element(By.XPATH, Locators.tratatuliLocator)
        flowers_our_flowers.click()
        time.sleep(0.5)

    with allure.step("Проверить, что пост в чужую публичную группу успешно опубликован"):
        posts_on_the_wall = driver.find_elements(By.XPATH, Locators.LAST_POST)
        last_post_text = posts_on_the_wall[0].find_element(By.XPATH, Locators.LAST_POST_CONTENT).text
        assert last_post_text == description + time_now
