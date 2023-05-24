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
from BotTests.variablesForTests.variables_for_posts import Users as Users

allure.title("Adverstising_in_someone_private_group")
allure.severity(severity_level="blocker")

def test_adverstising_in_someone_private_group():
    with allure.step("Выполнить запрос login для получения токена авторизации"):
        response = requests.get(RequestsVariables.BASE_URL + "/api/Employee/login/" + RequestsVariables.PASSWORD)
        assert response.status_code == 200

    response_json = response.json()
    authorization_token = response_json.get('token')
    headers = {"Authorization": "Bearer " + authorization_token}
    time_now = str(time.time())
    userName = Users.sericUserName
    comment = Users.userComment3
    description = OKData.CALLA_AMETIST_DESCRIPTION
    body = {
            "accountId": OKData.USER_ID,
            "groupIds": [
                OKData.groupTratatuliId
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
        postWallGroup = requests.post(RequestsVariables.BASE_URL + "/api/Advertising/ToAdvertiseInTheGroupAlbum", headers=headers, json=body)
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

    with allure.step("Нажать на фото последнего поста"):
        photo_posts = driver.find_elements(By.XPATH, Locators.lastPostPhoto)
        photo_posts[0].click()
        time.sleep(0.5)

    with allure.step("Проверить, что пост успешно опубликованн в чужую закрытую группу"):
        posts_text = driver.find_element(By.XPATH, Locators.postText).text
        assert posts_text == description + time_now
        time.sleep(0.5)

    with allure.step("Проверить, что комментарий успешно опубликованн в чужую закрытую группу"):
        allComments = driver.find_elements(By.XPATH, Locators.allComments)
        commentText = allComments[0].text

        assert commentText == comment + time_now
        time.sleep(0.5)

    with allure.step('Проверить, что комментарий оставлен {userName}'):
        allAuthorName = driver.find_elements(By.XPATH, Locators.allCommentsAuthorNames)
        authorName = allAuthorName[0].text

        assert authorName == userName