import datetime
import time

import allure
import pytest

from base_page import BasePage


class TestOpenCloseByRate:
    allure.title("test_open_close_by_rate")
    allure.severity(severity_level="blocker")

    @pytest.mark.skip()
    def test_open_close_by_rate(self, driver):
        form_page = BasePage(driver, "https://futures.mexc.com/ru-RU/exchange/FIDA_USDT?type=linear_swap")
        form_page.open_page()
        date = datetime.datetime.now()
        current_year = date.year
        current_month = date.month
        current_day = date.day

        # target_time = datetime.datetime(current_year, current_month, current_day, 2, 59, 58, 800000)
        # target_time_2 = datetime.datetime(current_year, current_month, current_day, 3, 0, 00, 250001)

        target_time = datetime.datetime(current_year, current_month, current_day, 10, 59, 58, 500000)
        target_time_2 = datetime.datetime(current_year, current_month, current_day, 11, 00, 00, 250001)

        # target_time = datetime.datetime(current_year, current_month, current_day, 18, 59, 59)
        # target_time_2 = datetime.datetime(current_year, current_month, current_day, 19, 00, 00, 250001)

        # target_time = datetime.datetime(current_year, current_month, current_day, 22, 59, 59)
        # target_time_2 = datetime.datetime(current_year, current_month, current_day, 23, 00, 00, 250001)

        # target_time = datetime.datetime(current_year, current_month, current_day, 6, 59, 59)
        # target_time_2 = datetime.datetime(current_year, current_month, current_day, 7, 0, 00, 250001)

        # target_time = datetime.datetime(current_year, current_month, current_day, 14, 59, 58, 800000)
        # target_time_2 = datetime.datetime(current_year, current_month, current_day, 15, 00, 00, 300001)

        date = datetime.datetime.now()
        while date <= target_time:
            date = datetime.datetime.now()
        else:
            print(f"время срабатывания таргет 1 {date}")

        confirm_button = '//button[@class= "ant-btn ant-btn-primary"]'

        form_page.click_button(confirm_button)

        while date <= target_time_2:
            date = datetime.datetime.now()
        else:
            close_deal = '//button[@class= "ant-btn ant-btn-default ant-btn-sm FastClose_flashCloseBtn__mz2O7 FastClose_closeBtn__ze4z7 FastClose_background__ew1j2"]'
            form_page.web_scrolled_into_view_elements(close_deal)
            print(datetime.datetime.now())
            form_page.click_button(close_deal)
            print(datetime.datetime.now())
            time.sleep(20)
