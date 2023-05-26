import time
import requests
import allure
from BotTests.variablesForTests.variables_for_posts import RequestsVariables as RequestsVariables
from BotTests.variablesForTests.variables_for_posts import OKData as OKData
from BotTests.variablesForTests.variables_for_posts import WebAddresses as WebAddresses
from BotTests.variablesForTests.variables_for_posts import Locators as Locators
from pages.form_page import FormPage


class TestPostInSomeonePublicGroup:
    allure.title("Someone public group wall post")
    allure.severity(severity_level="blocker")

    def test_own_public_group_post(self, driver):
        with allure.step("Выполнить запрос login для получения токена авторизации"):
            response = requests.get(RequestsVariables.BASE_URL + "/api/Employee/login/" + RequestsVariables.PASSWORD)
            assert response.status_code == 200

        response_json = response.json()
        authorization_token = response_json.get('token')

        headers = {"Authorization": "Bearer " + authorization_token}
        time_now = str(time.time())
        description = OKData.CALLA_AMETIST_DESCRIPTION
        body = [
            {
                "post": {
                    "description": description + time_now,
                    "photos": [
                        OKData.CALLA_AMETIST_PHOTO
                    ]
                },
                "users": [],
                "groupIdsByUserId": {
                    OKData.USER_ID: [
                        OKData.GROUP_TRANSISTORS_ID
                    ]
                }
            }
        ]

        with allure.step("Выполнить публикацию на стену группы"):
            post_wall = requests.post(RequestsVariables.BASE_URL + "/api/Posts/CreatePosts", headers=headers, json=body)
            assert post_wall.status_code == 200
            time.sleep(80)

        with allure.step("Перейти на страницу одноклассников"):
            form_page = FormPage(driver, WebAddresses.OK_LOGIN_PAGE_ADDRESS)
            form_page.openpage()

        with allure.step("Ввести логин"):
            form_page.enter_login()

        with allure.step("Ввести пароль"):
            form_page.enter_password()

        with allure.step("Нажать войти"):
            form_page.click_enter_button()

        with allure.step("Нажать на 'группы' в сайд баре"):
            form_page.click_group_in_side_bar(Locators.TOP_SIDE_NAVIGATION_BAR_LOCATORS_WITHOUT_XPATH)

        with allure.step("Нажать на группу 'Транзисторы'"):
            form_page.click_group_name_in_side_bar(Locators.TRANSISTORS_GROUP)

        with allure.step("Проверить, что пост в собственную публичную группу успешно опубликован"):
            form_page.check_post_publication(Locators.GROUP_PAGE, description, time_now, Locators.LAST_POST_CONTENT)