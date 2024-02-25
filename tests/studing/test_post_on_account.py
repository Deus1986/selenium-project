import os
import random
import emoji
import requests
import allure
from emoji import EMOJI_DATA

from pages.variables_for_posts import RequestsVariables as RequestsVariables, Users
from pages.variables_for_posts import OKData as OKData
from pages.variables_for_posts import WebAddresses as WebAddresses
from pages.variables_for_posts import Locators as Locators
from pages.form_page import FormPage


class TestReklama:
    allure.title("Reklama")
    allure.severity(severity_level="blocker")

    def test_Reklama(self, driver):
        smiles = {1: "🐰", 2: "🌹", 3: "🌷", 4: "🌺", 5: "❤", 6: "😃", 7: "☀️", 8: "⭐", 9: "😋",
                  10: "🙉", 11: "😻", 12: "😸", 13: "❤️", 14: "😺", 15: "😍", 16: "🙂", 17: "🐶",
                  18: "🍃", 19: "✨", 20: "🔥"}

        def wait_post(time_post, user_id, description, photo_url):
            response = requests.get(
                RequestsVariables.BASE_URL_2 + "/api/Employee/login/" + RequestsVariables.PASSWORD_PROD_2)
            assert response.status_code == 200
            response_json = response.json()
            authorization_token = response_json.get('token')

            headers = {"Authorization": "Bearer " + authorization_token}

            body = [
                {
                    "post": {
                        "description": description,

                        "photos": [
                            photo_url
                        ]
                    },
                    "users": [
                        {
                            "id": user_id,
                            "publicationTime": time_post
                        }
                    ],
                    "groupIdsByUserId": {}
                }
            ]

            post_wall = requests.post(RequestsVariables.BASE_URL_2 + "/api/Posts/CreatePosts", headers=headers,
                                      json=body)
            assert post_wall.status_code == 200
            print(body)

        test_user_id = "589219845582"
        # user ZOYA
        user_id_zoya = "599137408266"

        # user Irina
        user_id_irina = "590897640384"

        # user Elena
        user_id_elena = "597571355159"

        # user Ekaterina
        user_id_ekaterina = "584498530677"

        # user Igor Ulibka
        user_id_igor = "586349628133"

        # user Yurii Garden
        user_id_yurii = "594515200564"

        month = "02"
        day = 26

        description_1 = ""
        description_2 = "Так красиво просыпается весна! 😃"
        description_3 = ""
        description_4 = "Весна в бриллиантах! 😍"
        description_5 = "От природы с любовью! ❤"
        description_6 = "Псковский кремль ❤️"

        photo_url_1 = "https://i.mycdn.me/i?r=BDHElZJBPNKGuFyY-akIDfgn_PDOuAo2VMjLeRyetCRnT9UZ-HEKjXJHO2x1QIFP6bw&fn=h_768"
        photo_url_2 = "https://i.mycdn.me/i?r=BDHElZJBPNKGuFyY-akIDfgnhcvCCndOXXIrT4fF2mI3QS6ta2_JbQbUuvnZpDAoIq8"
        photo_url_3 = "https://i.mycdn.me/i?r=BDHElZJBPNKGuFyY-akIDfgnVxYeFRF1dD4W1wsnediv6v6hEuxZJRFUOb5B7LBGaes&fn=h_768"
        photo_url_4 = "https://i.mycdn.me/i?r=BDHElZJBPNKGuFyY-akIDfgneYwWMKyenEZdH5fmfoqioSNMiXdUfR2_1QQMrd2y3_I"
        photo_url_5 = "https://i.mycdn.me/i?r=BDHElZJBPNKGuFyY-akIDfgnU4mLzCUAB3EhDxcw5kh-LOXJ7beAtvffZIj66fXusmU"
        photo_url_6 = "https://i.mycdn.me/i?r=BDHElZJBPNKGuFyY-akIDfgnG18U2vNfVRqxpW-Pi1R9bfo_DedMgBVyX-nDOGd3ymw&fn=w_790"

        # wait_post("2024-" + f"{month}-" + f"0{day - 1}" + "T21:00:00.767Z", user_id_elena, description_1, photo_url_1)
        # wait_post("2024-" + f"{month}-" + f"0{day}" + "T01:00:00.767Z", user_id_irina, description_1, photo_url_1)
        # wait_post("2024-" + f"{month}-" + f"0{day}" + "T05:00:00.767Z", user_id_zoya, description_1, photo_url_1)
        # wait_post("2024-" + f"{month}-" + f"0{day}" + "T09:00:00.767Z", user_id_ekaterina, description_1, photo_url_1)
        # wait_post("2024-" + f"{month}-" + f"0{day}" + "T13:00:00.767Z", user_id_igor, description_1, photo_url_1)
        # wait_post("2024-" + f"{month}-" + f"0{day}" + "T17:00:00.767Z", user_id_yurii, description_1, photo_url_1)
        #
        # wait_post("2024-" + f"{month}-" + f"0{day}" + "T01:00:00.767Z", user_id_elena, description_2, photo_url_2)
        # wait_post("2024-" + f"{month}-" + f"0{day}" + "T05:00:00.767Z", user_id_irina, description_2, photo_url_2)
        # wait_post("2024-" + f"{month}-" + f"0{day}" + "T09:00:00.767Z", user_id_zoya, description_2, photo_url_2)
        # wait_post("2024-" + f"{month}-" + f"0{day}" + "T13:00:00.767Z", user_id_ekaterina, description_2, photo_url_2)
        # wait_post("2024-" + f"{month}-" + f"0{day}" + "T17:00:00.767Z", user_id_igor, description_2, photo_url_2)
        # wait_post("2024-" + f"{month}-" + f"0{day - 1}" + "T21:00:00.767Z", user_id_yurii, description_2, photo_url_2)
        #
        # wait_post("2024-" + f"{month}-" + f"0{day}" + "T05:00:00.767Z", user_id_elena, description_3, photo_url_3)
        # wait_post("2024-" + f"{month}-" + f"0{day}" + "T09:00:00.767Z", user_id_irina, description_3, photo_url_3)
        # wait_post("2024-" + f"{month}-" + f"0{day}" + "T13:00:00.767Z", user_id_zoya, description_3, photo_url_3)
        # wait_post("2024-" + f"{month}-" + f"0{day}" + "T17:00:00.767Z", user_id_ekaterina, description_3, photo_url_3)
        # wait_post("2024-" + f"{month}-" + f"0{day - 1}" + "T21:00:00.767Z", user_id_igor, description_3, photo_url_3)
        # wait_post("2024-" + f"{month}-" + f"0{day}" + "T01:00:00.767Z", user_id_yurii, description_3, photo_url_3)
        #
        # wait_post("2024-" + f"{month}-" + f"0{day}" + "T09:00:00.767Z", user_id_elena, description_4, photo_url_4)
        # wait_post("2024-" + f"{month}-" + f"0{day}" + "T13:00:00.767Z", user_id_irina, description_4, photo_url_4)
        # wait_post("2024-" + f"{month}-" + f"0{day}" + "T17:00:00.767Z", user_id_zoya, description_4, photo_url_4)
        # wait_post("2024-" + f"{month}-" + f"0{day - 1}" + "T21:00:00.767Z", user_id_ekaterina, description_4, photo_url_4)
        # wait_post("2024-" + f"{month}-" + f"0{day}" + "T01:00:00.767Z", user_id_igor, description_4, photo_url_4)
        # wait_post("2024-" + f"{month}-" + f"0{day}" + "T05:00:00.767Z", user_id_yurii, description_4, photo_url_4)
        #
        # wait_post("2024-" + f"{month}-" + f"0{day}" + "T13:00:00.767Z", user_id_elena, description_5, photo_url_5)
        # wait_post("2024-" + f"{month}-" + f"0{day}" + "T17:00:00.767Z", user_id_irina, description_5, photo_url_5)
        # wait_post("2024-" + f"{month}-" + f"0{day - 1}" + "T21:00:00.767Z", user_id_zoya, description_5, photo_url_5)
        # wait_post("2024-" + f"{month}-" + f"0{day}" + "T01:00:00.767Z", user_id_ekaterina, description_5, photo_url_5)
        # wait_post("2024-" + f"{month}-" + f"0{day}" + "T05:00:00.767Z", user_id_igor, description_5, photo_url_5)
        # wait_post("2024-" + f"{month}-" + f"0{day}" + "T09:00:00.767Z", user_id_yurii, description_5, photo_url_5)
        #
        # wait_post("2024-" + f"{month}-" + f"0{day}" + "T17:00:00.767Z", user_id_elena, description_6, photo_url_6)
        # wait_post("2024-" + f"{month}-" + f"0{day - 1}" + "T21:00:00.767Z", user_id_irina, description_6, photo_url_6)
        # wait_post("2024-" + f"{month}-" + f"0{day}" + "T01:00:00.767Z", user_id_zoya, description_6, photo_url_6)
        # wait_post("2024-" + f"{month}-" + f"0{day}" + "T05:00:00.767Z", user_id_ekaterina, description_6, photo_url_6)
        # wait_post("2024-" + f"{month}-" + f"0{day}" + "T09:00:00.767Z", user_id_igor, description_6, photo_url_6)
        # wait_post("2024-" + f"{month}-" + f"0{day}" + "T13:00:00.767Z", user_id_yurii, description_6, photo_url_6)

        # 2023-" + f"{month}-" + f"{day - 1}"
        # 2024-" + f"01-" + "31" + "T21:00:00.767Z"

        wait_post("2024-" + f"{month}-" + f"{day}" + "T01:00:00.767Z", user_id_irina, description_1, photo_url_1)
        wait_post("2024-" + f"{month}-" + f"{day}" + "T05:00:00.767Z", user_id_zoya, description_1, photo_url_1)
        wait_post("2024-" + f"{month}-" + f"{day}" + "T13:00:00.767Z", user_id_igor, description_1, photo_url_1)
        wait_post("2024-" + f"{month}-" + f"{day}" + "T17:00:00.767Z", user_id_yurii, description_1, photo_url_1)

        wait_post("2024-" + f"{month}-" + f"{day}" + "T05:00:00.767Z", user_id_irina, description_2, photo_url_2)
        wait_post("2024-" + f"{month}-" + f"{day}" + "T09:00:00.767Z", user_id_zoya, description_2, photo_url_2)
        wait_post("2024-" + f"{month}-" + f"{day}" + "T17:00:00.767Z", user_id_igor, description_2, photo_url_2)
        wait_post("2024-" + f"{month}-" + f"{day-1}" + "T21:00:00.767Z", user_id_yurii, description_2, photo_url_2)

        wait_post("2024-" + f"{month}-" + f"{day}" + "T09:00:00.767Z", user_id_irina, description_3, photo_url_3)
        wait_post("2024-" + f"{month}-" + f"{day}" + "T13:00:00.767Z", user_id_zoya, description_3, photo_url_3)
        wait_post("2024-" + f"{month}-" + f"{day - 1}" + "T21:00:00.767Z", user_id_igor, description_3, photo_url_3)
        wait_post("2024-" + f"{month}-" + f"{day}" + "T01:00:00.767Z", user_id_yurii, description_3, photo_url_3)

        wait_post("2024-" + f"{month}-" + f"{day}" + "T13:00:00.767Z", user_id_irina, description_4, photo_url_4)
        wait_post("2024-" + f"{month}-" + f"{day}" + "T17:00:00.767Z", user_id_zoya, description_4, photo_url_4)
        wait_post("2024-" + f"{month}-" + f"{day}" + "T01:00:00.767Z", user_id_igor, description_4, photo_url_4)
        wait_post("2024-" + f"{month}-" + f"{day}" + "T05:00:00.767Z", user_id_yurii, description_4, photo_url_4)

        wait_post("2024-" + f"{month}-" + f"{day}" + "T17:00:00.767Z", user_id_irina, description_5, photo_url_5)
        wait_post("2024-" + f"{month}-" + f"{day - 1}" + "T21:00:00.767Z", user_id_zoya, description_5, photo_url_5)
        wait_post("2024-" + f"{month}-" + f"{day}" + "T05:00:00.767Z", user_id_igor, description_5, photo_url_5)
        wait_post("2024-" + f"{month}-" + f"{day}" + "T09:00:00.767Z", user_id_yurii, description_5, photo_url_5)

        wait_post("2024-" + f"{month}-" + f"{day - 1}" + "T21:00:00.767Z", user_id_irina, description_6, photo_url_6)
        wait_post("2024-" + f"{month}-" + f"{day}" + "T01:00:00.767Z", user_id_zoya, description_6, photo_url_6)
        wait_post("2024-" + f"{month}-" + f"{day}" + "T09:00:00.767Z", user_id_igor, description_6, photo_url_6)
        wait_post("2024-" + f"{month}-" + f"{day}" + "T13:00:00.767Z", user_id_yurii, description_6, photo_url_6)











        # wait_post("2024-" + f"{month}-" + f"0{day - 1}" + "T21:00:00.767Z", user_id_elena, description_1, photo_url_1)
        # wait_post("2024-" + f"{month}-" + f"{day}" + "T01:00:00.767Z", user_id_irina, description_1, photo_url_1)
        # wait_post("2024-" + f"{month}-" + f"{day}" + "T05:00:00.767Z", user_id_zoya, description_1, photo_url_1)
        # wait_post("2024-" + f"{month}-" + f"{day}" + "T09:00:00.767Z", user_id_ekaterina, description_1, photo_url_1)
        # wait_post("2024-" + f"{month}-" + f"{day}" + "T13:00:00.767Z", user_id_igor, description_1, photo_url_1)
        # wait_post("2024-" + f"{month}-" + f"{day}" + "T17:00:00.767Z", user_id_yurii, description_1, photo_url_1)
        #
        # wait_post("2024-" + f"{month}-" + f"{day}" + "T01:00:00.767Z", user_id_elena, description_2, photo_url_2)
        # wait_post("2024-" + f"{month}-" + f"{day}" + "T05:00:00.767Z", user_id_irina, description_2, photo_url_2)
        # wait_post("2024-" + f"{month}-" + f"{day}" + "T09:00:00.767Z", user_id_zoya, description_2, photo_url_2)
        # wait_post("2024-" + f"{month}-" + f"{day}" + "T13:00:00.767Z", user_id_ekaterina, description_2, photo_url_2)
        # wait_post("2024-" + f"{month}-" + f"{day}" + "T17:00:00.767Z", user_id_igor, description_2, photo_url_2)
        # wait_post("2024-" + f"{month}-" + f"0{day - 1}" + "T21:00:00.767Z", user_id_yurii, description_2, photo_url_2)
        #
        # wait_post("2024-" + f"{month}-" + f"{day}" + "T05:00:00.767Z", user_id_elena, description_3, photo_url_3)
        # wait_post("2024-" + f"{month}-" + f"{day}" + "T09:00:00.767Z", user_id_irina, description_3, photo_url_3)
        # wait_post("2024-" + f"{month}-" + f"{day}" + "T13:00:00.767Z", user_id_zoya, description_3, photo_url_3)
        # wait_post("2024-" + f"{month}-" + f"{day}" + "T17:00:00.767Z", user_id_ekaterina, description_3, photo_url_3)
        # wait_post("2024-" + f"{month}-" + f"0{day - 1}" + "T21:00:00.767Z", user_id_igor, description_3, photo_url_3)
        # wait_post("2024-" + f"{month}-" + f"{day}" + "T01:00:00.767Z", user_id_yurii, description_3, photo_url_3)
        #
        # wait_post("2024-" + f"{month}-" + f"{day}" + "T09:00:00.767Z", user_id_elena, description_4, photo_url_4)
        # wait_post("2024-" + f"{month}-" + f"{day}" + "T13:00:00.767Z", user_id_irina, description_4, photo_url_4)
        # wait_post("2024-" + f"{month}-" + f"{day}" + "T17:00:00.767Z", user_id_zoya, description_4, photo_url_4)
        # wait_post("2024-" + f"{month}-" + f"0{day - 1}" + "T21:00:00.767Z", user_id_ekaterina, description_4,
        #           photo_url_4)
        # wait_post("2024-" + f"{month}-" + f"{day}" + "T01:00:00.767Z", user_id_igor, description_4, photo_url_4)
        # wait_post("2024-" + f"{month}-" + f"{day}" + "T05:00:00.767Z", user_id_yurii, description_4, photo_url_4)
        #
        # wait_post("2024-" + f"{month}-" + f"{day}" + "T13:00:00.767Z", user_id_elena, description_5, photo_url_5)
        # wait_post("2024-" + f"{month}-" + f"{day}" + "T17:00:00.767Z", user_id_irina, description_5, photo_url_5)
        # wait_post("2024-" + f"{month}-" + f"0{day - 1}" + "T21:00:00.767Z", user_id_zoya, description_5, photo_url_5)
        # wait_post("2024-" + f"{month}-" + f"{day}" + "T01:00:00.767Z", user_id_ekaterina, description_5, photo_url_5)
        # wait_post("2024-" + f"{month}-" + f"{day}" + "T05:00:00.767Z", user_id_igor, description_5, photo_url_5)
        # wait_post("2024-" + f"{month}-" + f"{day}" + "T09:00:00.767Z", user_id_yurii, description_5, photo_url_5)
        #
        # wait_post("2024-" + f"{month}-" + f"{day}" + "T17:00:00.767Z", user_id_elena, description_6, photo_url_6)
        # wait_post("2024-" + f"{month}-" + f"0{day - 1}" + "T21:00:00.767Z", user_id_irina, description_6, photo_url_6)
        # wait_post("2024-" + f"{month}-" + f"{day}" + "T01:00:00.767Z", user_id_zoya, description_6, photo_url_6)
        # wait_post("2024-" + f"{month}-" + f"{day}" + "T05:00:00.767Z", user_id_ekaterina, description_6, photo_url_6)
        # wait_post("2024-" + f"{month}-" + f"{day}" + "T09:00:00.767Z", user_id_igor, description_6, photo_url_6)
        # wait_post("2024-" + f"{month}-" + f"{day}" + "T13:00:00.767Z", user_id_yurii, description_6, photo_url_6)
