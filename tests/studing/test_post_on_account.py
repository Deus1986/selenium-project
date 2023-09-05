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
        #user ZOYA
        user_id_zoya = "599137408266"
        #user Irina
        user_id_irina = "590897640384"
        # user Elena
        user_id_elena = "590114919871"

        month = "09"
        day = 7
        description = "собака сутулая" + smiles[6]
        photo_url = "https://i.mycdn.me/i?r=AyH4iRPQ2q0otWIFepML2LxRPyNBNXl1zaWeT0xG0JkIew&fn=w_548"

        # wait_post("2023-" + f"{month}-" + f"0{day - 1}" + "T21:00:00.767Z", user_id_elena, description, photo_url)
        # wait_post("2023-" + f"{month}-" + f"0{day}" + "T01:00:00.767Z", user_id_irina, description, photo_url)
        # wait_post("2023-" + f"{month}-" + f"0{day}" + "T05:00:00.767Z", user_id_zoya, description, photo_url)

        # wait_post("2023-" + f"{month}-" + f"0{day}" + "T01:00:00.767Z", user_id_elena, description, photo_url)
        # wait_post("2023-" + f"{month}-" + f"0{day}" + "T05:00:00.767Z", user_id_irina, description, photo_url)
        # wait_post("2023-" + f"{month}-" + f"0{day}" + "T09:00:00.767Z", user_id_zoya, description, photo_url)

        # wait_post("2023-" + f"{month}-" + f"0{day}" + "T05:00:00.767Z", user_id_elena, description, photo_url)
        # wait_post("2023-" + f"{month}-" + f"0{day}" + "T09:00:00.767Z", user_id_irina, description, photo_url)
        # wait_post("2023-" + f"{month}-" + f"0{day}" + "T13:00:00.767Z", user_id_zoya, description, photo_url)

        # wait_post("2023-" + f"{month}-" + f"0{day}" + "T09:00:00.767Z", user_id_elena, description, photo_url)
        # wait_post("2023-" + f"{month}-" + f"0{day}" + "T13:00:00.767Z", user_id_irina, description, photo_url)
        # wait_post("2023-" + f"{month}-" + f"0{day}" + "T17:00:00.767Z", user_id_zoya, description, photo_url)

        # wait_post("2023-" + f"{month}-" + f"0{day}" + "T13:00:00.767Z", user_id_elena, description, photo_url)
        # wait_post("2023-" + f"{month}-" + f"0{day}" + "T17:00:00.767Z", user_id_irina, description, photo_url)
        # wait_post("2023-" + f"{month}-" + f"0{day-1}" + "T21:00:00.767Z", user_id_zoya, description, photo_url)

        # wait_post("2023-" + f"{month}-" + f"0{day}" + "T17:00:00.767Z", user_id_elena, description, photo_url)
        # wait_post("2023-" + f"{month}-" + f"0{day-1}" + "T21:00:00.767Z", user_id_irina, description, photo_url)
        # wait_post("2023-" + f"{month}-" + f"0{day}" + "T01:00:00.767Z", user_id_zoya, description, photo_url)

