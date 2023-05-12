import time
from selenium import webdriver
from selenium.webdriver.common.by import By
import requests
import allure

allure.title("Adverstising_in_own_public_group")
allure.severity(severity_level="blocker")

def test_adverstising_in_own_public_group():
    with allure.step("Выполнить запрос login для получения токена авторизации"):
        response = requests.get("http://10.243.8.118:31405/api/Employee/login/Gfhjkm")
        assert response.status_code == 200

    response_json = response.json()
    authorization_token = response_json.get('token')
    headers = {"Authorization": "Bearer " + authorization_token}
    time_now = str(time.time())
    userName = "Серик Обуманян"
    comment = "Вот такие вот цветочки"
    description = "Калла Кэптин Морелли. Цена 410 руб. Морозоустойчивость до -7С." \
                  "Очень элегантный цветок, одетый в нежное желтое покрывало с легким " \
                  "фиолетовым тоном и со светлыми прожилками. Листья темно-зеленого цвета " \
                  "с редким, белым крапом и восковым блеском. Калла - многолетнее травянистое " \
                  "растение, которое украсит не только сад, но и дом. Представляет собой небольшой " \
                  "травянистый куст шириной 30 -35см и высотой цветоноса до 60 см."
    body = {
            "accountId": "589219845582",
            "groupIds": [
                "70000002189774"
            ],
            "photos": [
                {
                    "url": "https://i.mycdn.me/i?r=AyH4iRPQ2q0otWIFepML2LxRokwSgPrsezL9_J6y_vVGQg",
                    "description": description + time_now,
                    "comment": comment + time_now
                }
            ]
        }

    with allure.step("Выполнить публикацию на стену группы с комментарием"):
        postWallGroup = requests.post("http://10.243.8.118:31405/api/Advertising/ToAdvertiseInTheGroupAlbum", headers=headers, json=body)
        assert postWallGroup.status_code == 200
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

    with allure.step("Нажать на 'группы' в сайд баре"):
        top_side_navigation_bar = driver.find_elements(By.XPATH, '//div[@class = "nav-side_i-w"]')
        side_bar_account_name_link = top_side_navigation_bar[4]
        side_bar_account_name_link.click()
        time.sleep(0.5)

    with allure.step("Нажать на группу 'Цветочки наши цветочки'"):
        flowers_our_flowers = driver.find_element(By.XPATH, '//div[@data-group-id= "70000002189774"]')
        flowers_our_flowers.click()
        time.sleep(0.5)

    with allure.step("Нажать на фото последнего поста"):
        photo_posts = driver.find_elements(By.XPATH, '//img[@class = "collage_img"]')
        photo_posts[0].click()
        time.sleep(0.5)

    with allure.step("Проверить, что пост успешно опубликованн в свою открытую группу"):
        posts_text = driver.find_element(By.XPATH, '//div[@class = "h-mod photo-layer_descr photo-layer_bottom_block"]'
                                                   '//div[@tsid= "TextFieldText"]').text
        assert posts_text == description + time_now
        time.sleep(0.5)

    with allure.step("Проверить, что комментарий успешно опубликованн в свою открытую группу"):
        allComments = driver.find_elements(By.XPATH, '// span[ @class = "js-text-full"]')
        commentText = allComments[0].text

        assert commentText == comment + time_now
        time.sleep(0.5)

    with allure.step('Проверить, что комментарий оставлен {userName}'):
        allAuthorName = driver.find_elements(By.XPATH, '// a[ @class = "comments_author-name o"]')
        authorName = allAuthorName[0].text

        assert authorName == userName