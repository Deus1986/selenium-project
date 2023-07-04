import time
import requests
import allure
from pages.variables_for_posts import RequestsVariables as RequestsVariables, Locators
from pages.variables_for_posts import OKData as OKData
from pages.variables_for_posts import WebAddresses as WebAddresses
from pages.form_page import FormPage


class TestPostPrivateWall:
    allure.title("Public private wall post")
    allure.severity(severity_level="blocker")

    def test_post_private_wall(self, driver):
        with allure.step("Выполнить запрос login для получения токена авторизации"):
            response = requests.get(RequestsVariables.BASE_URL + "/api/Employee/login/" + RequestsVariables.PASSWORD)
            assert response.status_code == 200

        response_json = response.json()
        authorization_token = response_json.get('token')
        headers = {"Authorization": "Bearer " + authorization_token}
        time_now = str(time.time())
        description = OKData.CALLA_CHAMELEON_DESCRIPTION
        body = [
            {
                "post": {
                    "description": description + time_now,
                    "photos": [
                        OKData.CALLA_CHAMELEON_PHOTO
                    ]
                },
                "users": [
                    {
                        "id": OKData.USER_ID,
                        "publicationTime": None
                    }
                ],
                "groupIdsByUserId": {}
            }
        ]

        with allure.step("Выполнить публикацию на стену аккаунта"):
            post_wall = requests.post(RequestsVariables.BASE_URL + "/api/Posts/CreatePosts", headers=headers, json=body)
            assert post_wall.status_code == 200
            time.sleep(80)

        with allure.step("Перейти на страницу одноклассников"):
            form_page = FormPage(driver, WebAddresses.OK_LOGIN_PAGE_ADDRESS)
            form_page.openpage()

        with allure.step("Ввести логин"):
            form_page.enter_seric_login()

        with allure.step("Ввести пароль"):
            form_page.enter_password()

        with allure.step("Нажать войти"):
            form_page.click_enter_button()

        with allure.step("Нажать на имя пользователя в сайд баре"):
            form_page.click_user_name_in_side_bar()

        with allure.step("Проверить, что пост успешно опубликован"):
            form_page.check_post_publication(Locators.PRIVAT_WALL, description, time_now, Locators.LAST_POST_CONTENT)
