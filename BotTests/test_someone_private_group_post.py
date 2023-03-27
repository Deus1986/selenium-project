import time
from selenium import webdriver
from selenium.webdriver.common.by import By
import requests
import allure

allure.title("Someone private group wall post")
allure.severity(severity_level="blocker")


def test_someone_private_group_post():
    with allure.step("Выполнить запрос login для получения токена авторизации"):
        response = requests.get("http://10.243.10.12:5000/api/Employee/login/Пароль")
        assert response.status_code == 200

    response_json = response.json()
    authorization_token = response_json.get('token')
    headers = {"Authorization": "Bearer " + authorization_token}
    time_now = str(time.time())
    description = "Калла Вермеер. Цена 410 руб.  Морозоустойчивость до -7С.Многолетнее " \
                  "травянистое растение с клубневидным корневищем. Листья крупные," \
                  " стреловидной формы, блестящие, восковые. Соцветие одиночное на длинном " \
                  "цветоносе в виде кремово-желтого початка в обрамлении воронковидного, " \
                  "слегка волнистого покрывала.  Высота растения 60-70 см. Цветение длительное, " \
                  "июнь-сентябрь. Период цветения: июнь-август. Место посадки: солнце/полутень."
    body = [
        {
            "post": {
                "description": description + time_now,
                "photos": [
                    "https://i.mycdn.me/i?r=AyH4iRPQ2q0otWIFepML2LxRAo0vbrCRl1q-RnvNbvFdHg"
                ]
            },
            "users": [],
            "groupIdsByUserId": {
                "589219845582": [
                    "70000002228915"
                ]
            }
        }
    ]

    with allure.step("Выполнить публикацию на стену группы"):
        postWallGroup = requests.post("http://10.243.10.12:5000/api/Posts/CreatePosts", headers=headers, json=body)
        assert postWallGroup.status_code == 200
        time.sleep(100)

    driver = webdriver.Chrome("E://Selenium//chromedriver.exe")
    with allure.step("Перейти на страницу одноклассников"):
        driver.get("https://ok.ru/")

    with allure.step("Ввести логин и пароль, и нажать войти"):
        email_field = driver.find_element(By.XPATH, '//input[@id = "field_email"]')
        email_field.send_keys(77712906977)

    with allure.step("Ввести пароль"):
        password_field = driver.find_element(By.XPATH, '//input[@id = "field_password"]')
        password_field.send_keys('lexusrx300')

    with allure.step("Нажать войти"):
        enter_account_button = driver.find_element(By.XPATH, '//input[@value= "Войти в Одноклассники"]')
        enter_account_button.click()

    with allure.step("Нажать на 'группы' в сайд баре"):
        top_side_navigation_bar = driver.find_elements(By.XPATH, '//div[@class = "nav-side_i-w"]')
        side_bar_account_name_link = top_side_navigation_bar[4]
        side_bar_account_name_link.click()
        time.sleep(0.5)

    with allure.step("Нажать на группу 'Трататули'"):
        flowers_our_flowers = driver.find_element(By.XPATH, '//div[@data-group-id= "70000002228915"]')
        flowers_our_flowers.click()
        time.sleep(0.5)

    with allure.step("Проверить, что пост в чужую публичную группу успешно опубликован"):
        posts_on_the_wall = driver.find_elements(By.XPATH, '//div[@class = "feed-w"]')
        last_post_text = posts_on_the_wall[0].find_element(By.XPATH,
                                                           '//div[@class = "media-text_cnt_tx emoji-tx textWrap"]').text
        assert last_post_text == description + time_now
