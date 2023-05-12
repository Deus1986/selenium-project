import time
from selenium import webdriver
from selenium.webdriver.common.by import By
import requests
import allure

allure.title("Public private wall post")
allure.severity(severity_level="blocker")
def test_post_private_wall():

    with allure.step("Выполнить запрос login для получения токена авторизации"):
        response = requests.get("http://10.243.8.118:31405/api/Employee/login/Gfhjkm")
        assert response.status_code == 200


    response_json = response.json()
    authorization_token = response_json.get('token')
    headers = {"Authorization": "Bearer " + authorization_token}
    time_now = str(time.time())
    description = "Калла Хамелеон. Цена 410 руб. Морозостойкость -6. " \
                  "Многолетнее травянистое растение с клубневидным корневищем. Листья крупные, " \
                  "стреловидной формы, блестящие, восковые. Соцветие одиночное на длинном цветоносе " \
                  "в виде кремово-желтого початка в обрамлении воронковидного, слегка волнистого покрывала. " \
                  "Окраска разнообразная: белая, желтая, розовая, карминная, лиловая. Высота прямых цветоносов " \
                  "достигает 60 см. Листья стреловидные, остроконечные или в форме сердца, темно-зеленые с белым крапом. "
    body = [
        {
            "post": {
                "description": description + time_now,
                "photos": [
                    "https://i.mycdn.me/i?r=AyH4iRPQ2q0otWIFepML2LxRpNHaRE83Idry9iysh9m3LQ"
                ]
            },
            "users": [
                {
                    "id": "589219845582",
                    "publicationTime": None
                }
            ],
            "groupIdsByUserId": {}
        }
    ]

    with allure.step("Выполнить публикацию на стену аккаунта"):
        postWall = requests.post("http://10.243.8.118:31405/api/Posts/CreatePosts", headers=headers, json=body)
        assert postWall.status_code == 200
        time.sleep(60)

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

    with allure.step("Нажать на имя пользователя в сайд баре"):
        top_side_navigation_bar = driver.find_elements(By.XPATH, '//div[@class = "nav-side_i-w"]')
        side_bar_account_name_link = top_side_navigation_bar[0]
        side_bar_account_name_link.click()
        time.sleep(1)

    with allure.step("Проверить, что пост успешно опубликован"):
        posts_on_the_wall = driver.find_elements(By.XPATH, '//div[@class = "feed-w"]')
        last_post_text = posts_on_the_wall[0].find_element(By.XPATH,
                                                           '//div[@class = "media-text_cnt_tx emoji-tx textWrap"]').text
        assert last_post_text == description + time_now