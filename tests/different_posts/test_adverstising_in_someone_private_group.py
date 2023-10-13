import time

import pytest
import requests
import allure
from pages.variables_for_posts import RequestsVariables as RequestsVariables, Users
from pages.variables_for_posts import OKData as OKData
from pages.variables_for_posts import WebAddresses as WebAddresses
from pages.variables_for_posts import Locators as Locators
from pages.form_page import FormPage


@pytest.mark.skip
class TestAdverstisingInSomeonePrivateGroup:
    allure.title("Adverstising_in_someone_private_group")
    allure.severity(severity_level="blocker")

    def test_adverstising_in_someone_private_group(self, driver):
        driver.implicitly_wait(10)
        with allure.step("Выполнить запрос login для получения токена авторизации"):
            response = requests.get(RequestsVariables.BASE_URL + "/api/Employee/login/" + RequestsVariables.PASSWORD)
            assert response.status_code == 200

        response_json = response.json()
        authorization_token = response_json.get('token')
        headers = {"Authorization": "Bearer " + authorization_token}
        time_now = str(time.time())
        user_name = Users.SERIC_USER_NAME
        comment = Users.USER_COMMENT_3
        description = OKData.CALLA_AMETIST_DESCRIPTION
        body = {
            "accountId": OKData.USER_ID,
            "groupIds": [
                OKData.GROUP_TRATATULI_ID
            ],
            "photos": [
                {
                    "url": OKData.CALLA_AMETIST_PHOTO,
                    "description": description + time_now,
                    "comment": comment + time_now
                }
            ]
        }

        with allure.step("Выполнить публикацию на стену группы с комментарием"):
            post_wall_group = requests.post(RequestsVariables.BASE_URL + "/api/Advertising/ToAdvertiseInTheGroupAlbum",
                                            headers=headers, json=body)
            assert post_wall_group.status_code == 200
            time.sleep(60)

        with allure.step("Перейти на страницу одноклассников"):
            form_page = FormPage(driver, WebAddresses.OK_LOGIN_PAGE_ADDRESS)
            form_page.openpage()

        with allure.step("Ввести логин"):
            form_page.enter_seric_login()

        with allure.step("Ввести пароль"):
            form_page.enter_password()

        with allure.step("Нажать войти"):
            form_page.click_enter_button()

        with allure.step("Нажать на 'группы' в сайд баре"):
            form_page.click_group_in_side_bar(Locators.TOP_SIDE_NAVIGATION_BAR_LOCATORS_WITHOUT_XPATH)

        with allure.step("Нажать на группу 'Трататули'"):
            form_page.click_group_name_in_side_bar(Locators.TRATATULI_GROUP)

        with allure.step("Нажать на фото последнего поста"):
            form_page.click_last_post_photo(Locators.GROUP_PAGE)

        with allure.step("Проверить, что пост успешно опубликованн в чужую открытую группу"):
            form_page.verify_last_post_text(description, time_now)

        with allure.step("Проверить, что комментарий успешно опубликованн в чужую открытую группу"):
            form_page.verify_comment_publication(comment, time_now)

        with allure.step(f'Проверить, что комментарий оставлен {user_name}'):
            form_page.verify_comment_author(user_name)
