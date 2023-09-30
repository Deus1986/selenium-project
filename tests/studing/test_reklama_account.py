import random
import requests
import allure

from pages.variables_for_posts import RequestsVariables as RequestsVariables, Users



class TestReklama:
    allure.title("Reklama")
    allure.severity(severity_level="blocker")

    def test_Reklama(self, driver):
        def wait_post(time_post, user_id):
            smiles = {1: "🐰", 2: "🌹", 3: "🌷", 4: "🌺", 5: "❤", 6: "😃", 7: "☀️", 8: "⭐", 9: "😋",
                      10: "🙉", 11: "😻", 12: "😸", 13: "❤️", 14: "😺", 15: "😍", 16: "🙂", 17: "🐶",
                      18: "🍃", 19: "✨", 20: "🔥"}
            response = requests.get(
                RequestsVariables.BASE_URL_2 + "/api/Employee/login/" + RequestsVariables.PASSWORD_PROD_2)
            assert response.status_code == 200
            response_json = response.json()
            authorization_token = response_json.get('token')

            headers = {"Authorization": "Bearer " + authorization_token}

            response_albums = requests.get(RequestsVariables.BASE_URL_2 + "/api/Album/GetAlbums", headers=headers)
            response_albums_json = response_albums.json()
            response_albums_json_random_album_id = response_albums_json[
                random.randint(0, len(response_albums_json) - 1)]
            album_id = response_albums_json_random_album_id["id"]
            # print(len(response_albums_json))

            response_products = requests.get(
                RequestsVariables.BASE_URL_2 + f"/api/Product/GetNameProducts?albumId={album_id}", headers=headers)
            response_products_json = response_products.json()
            # print(response_albums_json_random_album_id)

            while len(response_products_json) == 0:
                response_albums = requests.get(RequestsVariables.BASE_URL_2 + "/api/Album/GetAlbums", headers=headers)
                response_albums_json = response_albums.json()
                response_albums_json_random_album_id = response_albums_json[
                    random.randint(0, len(response_albums_json) - 1)]
                album_id = response_albums_json_random_album_id["id"]
                response_products = requests.get(
                    RequestsVariables.BASE_URL_2 + f"/api/Product/GetNameProducts?albumId={album_id}", headers=headers)
                response_products_json = response_products.json()

            # print(album_id)

            response_products_json_random_product_id = response_products_json[
                random.randint(0, len(response_products_json) - 1)]
            product_id = response_products_json_random_product_id["id"]
            # print(response_products_json)
            # print(product_id)

            response_product_info = requests.get(RequestsVariables.BASE_URL_2 + f"/api/Product?Id={product_id}",
                                                 headers=headers)
            response_product_info_json = response_product_info.json()
            # print(response_product_info_json)
            # print(str(response_product_info_json["productPrices"][0]["price"]))

            body = [
                {
                    "post": {
                        "description": response_product_info_json["name"] + " Цена: " +
                                       str(response_product_info_json["productPrices"][0]["price"]) +
                                       f' {response_product_info_json["images"][0]["description"]}' +
                                       smiles[random.randint(1, len(smiles))] +
                                       smiles[random.randint(1, len(smiles))] +
                                       smiles[random.randint(1, len(smiles))],

                        "photos": [
                            response_product_info_json["images"][0]["url"]
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
            # print(post_wall.url)

        #user ZOYA
        user_id_zoya = "599137408266"

        #user Irina
        user_id_irina = "590897640384"

        # user Elena
        user_id_elena = "590114919871"

        # user Ekaterina
        user_id_ekaterina = "584498530677"

        # user Igor Ulibka
        user_id_igor = "586349628133"

        month = "10"
        day = 1

        # wait_post("2023-" + f"{month}-" + f"{day - 1}" + "T23:00:00.767Z", user_id_zoya)
        # wait_post("2023-" + f"{month}-" + f"{day}" + "T03:00:00.767Z", user_id_zoya)
        # wait_post("2023-" + f"{month}-" + f"{day}" + "T07:00:00.767Z", user_id_zoya)
        # wait_post("2023-" + f"{month}-" + f"{day}" + "T11:00:00.767Z", user_id_zoya)
        # wait_post("2023-" + f"{month}-" + f"{day}" + "T15:00:00.767Z", user_id_zoya)
        # wait_post("2023-" + f"{month}-" + f"{day}" + "T19:00:00.767Z", user_id_zoya)
        #
        # wait_post("2023-" + f"{month}-" + f"{day - 1}" + "T23:00:00.767Z", user_id_irina)
        # wait_post("2023-" + f"{month}-" + f"{day}" + "T03:00:00.767Z", user_id_irina)
        # wait_post("2023-" + f"{month}-" + f"{day}" + "T07:00:00.767Z", user_id_irina)
        # wait_post("2023-" + f"{month}-" + f"{day}" + "T11:00:00.767Z", user_id_irina)
        # wait_post("2023-" + f"{month}-" + f"{day}" + "T15:00:00.767Z", user_id_irina)
        # wait_post("2023-" + f"{month}-" + f"{day}" + "T19:00:00.767Z", user_id_irina)
        #
        # wait_post("2023-" + f"{month}-" + f"{day - 1}" + "T23:00:00.767Z", user_id_elena)
        # wait_post("2023-" + f"{month}-" + f"{day}" + "T03:00:00.767Z", user_id_elena)
        # wait_post("2023-" + f"{month}-" + f"{day}" + "T07:00:00.767Z", user_id_elena)
        # wait_post("2023-" + f"{month}-" + f"{day}" + "T11:00:00.767Z", user_id_elena)
        # wait_post("2023-" + f"{month}-" + f"{day}" + "T15:00:00.767Z", user_id_elena)
        # wait_post("2023-" + f"{month}-" + f"{day}" + "T19:00:00.767Z", user_id_elena)
        #
        # wait_post("2023-" + f"{month}-" + f"{day - 1}" + "T23:00:00.767Z", user_id_ekaterina)
        # wait_post("2023-" + f"{month}-" + f"{day}" + "T03:00:00.767Z", user_id_ekaterina)
        # wait_post("2023-" + f"{month}-" + f"{day}" + "T07:00:00.767Z", user_id_ekaterina)
        # wait_post("2023-" + f"{month}-" + f"{day}" + "T11:00:00.767Z", user_id_ekaterina)
        # wait_post("2023-" + f"{month}-" + f"{day}" + "T15:00:00.767Z", user_id_ekaterina)
        # wait_post("2023-" + f"{month}-" + f"{day}" + "T19:00:00.767Z", user_id_ekaterina)
        #
        # wait_post("2023-" + f"{month}-" + f"{day - 1}" + "T23:00:00.767Z", user_id_igor)
        # wait_post("2023-" + f"{month}-" + f"{day}" + "T03:00:00.767Z", user_id_igor)
        # wait_post("2023-" + f"{month}-" + f"{day}" + "T07:00:00.767Z", user_id_igor)
        # wait_post("2023-" + f"{month}-" + f"{day}" + "T11:00:00.767Z", user_id_igor)
        # wait_post("2023-" + f"{month}-" + f"{day}" + "T15:00:00.767Z", user_id_igor)
        # wait_post("2023-" + f"{month}-" + f"{day}" + "T19:00:00.767Z", user_id_igor)

#         weetetete

        wait_post("2023-" + f"{month}-" + f"0{day - 1}" + "T23:00:00.767Z", user_id_zoya)
        wait_post("2023-" + f"{month}-" + f"0{day}" + "T03:00:00.767Z", user_id_zoya)
        wait_post("2023-" + f"{month}-" + f"0{day}" + "T07:00:00.767Z", user_id_zoya)
        wait_post("2023-" + f"{month}-" + f"0{day}" + "T11:00:00.767Z", user_id_zoya)
        wait_post("2023-" + f"{month}-" + f"0{day}" + "T15:00:00.767Z", user_id_zoya)
        wait_post("2023-" + f"{month}-" + f"0{day}" + "T19:00:00.767Z", user_id_zoya)

        wait_post("2023-" + f"{month}-" + f"0{day - 1}" + "T23:00:00.767Z", user_id_irina)
        wait_post("2023-" + f"{month}-" + f"0{day}" + "T03:00:00.767Z", user_id_irina)
        wait_post("2023-" + f"{month}-" + f"0{day}" + "T07:00:00.767Z", user_id_irina)
        wait_post("2023-" + f"{month}-" + f"0{day}" + "T11:00:00.767Z", user_id_irina)
        wait_post("2023-" + f"{month}-" + f"0{day}" + "T15:00:00.767Z", user_id_irina)
        wait_post("2023-" + f"{month}-" + f"0{day}" + "T19:00:00.767Z", user_id_irina)

        wait_post("2023-" + f"{month}-" + f"0{day - 1}" + "T23:00:00.767Z", user_id_elena)
        wait_post("2023-" + f"{month}-" + f"0{day}" + "T03:00:00.767Z", user_id_elena)
        wait_post("2023-" + f"{month}-" + f"0{day}" + "T07:00:00.767Z", user_id_elena)
        wait_post("2023-" + f"{month}-" + f"0{day}" + "T11:00:00.767Z", user_id_elena)
        wait_post("2023-" + f"{month}-" + f"0{day}" + "T15:00:00.767Z", user_id_elena)
        wait_post("2023-" + f"{month}-" + f"0{day}" + "T19:00:00.767Z", user_id_elena)

        wait_post("2023-" + f"{month}-" + f"0{day - 1}" + "T23:00:00.767Z", user_id_ekaterina)
        wait_post("2023-" + f"{month}-" + f"0{day}" + "T03:00:00.767Z", user_id_ekaterina)
        wait_post("2023-" + f"{month}-" + f"0{day}" + "T07:00:00.767Z", user_id_ekaterina)
        wait_post("2023-" + f"{month}-" + f"0{day}" + "T11:00:00.767Z", user_id_ekaterina)
        wait_post("2023-" + f"{month}-" + f"0{day}" + "T15:00:00.767Z", user_id_ekaterina)
        wait_post("2023-" + f"{month}-" + f"0{day}" + "T19:00:00.767Z", user_id_ekaterina)

        wait_post("2023-" + f"{month}-" + f"0{day - 1}" + "T23:00:00.767Z", user_id_igor)
        wait_post("2023-" + f"{month}-" + f"0{day}" + "T03:00:00.767Z", user_id_igor)
        wait_post("2023-" + f"{month}-" + f"0{day}" + "T07:00:00.767Z", user_id_igor)
        wait_post("2023-" + f"{month}-" + f"0{day}" + "T11:00:00.767Z", user_id_igor)
        wait_post("2023-" + f"{month}-" + f"0{day}" + "T15:00:00.767Z", user_id_igor)
        wait_post("2023-" + f"{month}-" + f"0{day}" + "T19:00:00.767Z", user_id_igor)




