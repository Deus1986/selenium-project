from selenium.webdriver.common.by import By


# class RequestsVariables:
#     BASE_URL = "http://10.243.8.118:31405"
#     PASSWORD = "Gfhjkm"

class OKBotData:
    LOGIN_DJORDJ_KIM = 77767850360
    PASSWORD_SERIC = 'lexusrx300'


#     CALLA_CHAMELEON_DESCRIPTION = "Калла Хамелеон. Цена 410 руб. Морозостойкость -6. " \
#                   "Многолетнее травянистое растение с клубневидным корневищем. Листья крупные, " \
#                   "стреловидной формы, блестящие, восковые. Соцветие одиночное на длинном цветоносе " \
#                   "в виде кремово-желтого початка в обрамлении воронковидного, слегка волнистого покрывала. " \
#                   "Окраска разнообразная: белая, желтая, розовая, карминная, лиловая. Высота прямых цветоносов " \
#                   "достигает 60 см. Листья стреловидные, остроконечные или в форме сердца, темно-зеленые с белым крапом. "
#
#     CALLA_CAPTIN_MORELLI_DESCRIPTION = "Калла Кэптин Морелли. Цена 410 руб. Морозоустойчивость до -7С." \
#                                     "Очень элегантный цветок, одетый в нежное желтое покрывало с легким " \
#                                     "фиолетовым тоном и со светлыми прожилками. Листья темно-зеленого цвета с " \
#                                     "редким, белым крапом и восковым блеском. Калла - многолетнее травянистое " \
#                                     "растение, которое украсит не только сад, но и дом. Представляет собой небольшой " \
#                                     "травянистый куст шириной 30 -35см и высотой цветоноса до 60 см."
#
#     CALLA_AMETIST_DESCRIPTION = "Калла Аметист. Цена 410 руб. Морозоустойчивость до -7С." \
#                   "Многолетний, крупноцветковый сорт. Высота растения 60-70 сантиметров. " \
#                   "Период цветения: июнь-июль-август. Цветок крупного размера, в форме свечи, " \
#                   "тёмно-фиолетового цвета. Высота растения 60-70 сантиметров. " \
#                   "Период цветения: июнь-июль-август."
#
#     CALLA_CAPTIN_SAFARI_DESCRIPTION = "Калла Кэптин Сафари. Цена 410 руб. Морозоустойчивость до -7С." \
#                   "Растение представляет собой травянистое многолетнее клубневое " \
#                   "красивоцветущее растение. Высота взрослого растения достигает 50-80 " \
#                   "сантиметров в высоту. Листья вытянутые, зеленого цвета с белым вкраплением. " \
#                   "Цветки в форме свечи, желто-оранжево-розовые."
#
#     CALLA_CAPTIN_VENTURA_DESCRIPTION = "Калла Кэптин Вентура .Цена 410 руб. Морозоустойчивость до -6С. " \
#                   "Многолетнее травянистое растение с клубневидным корневищем. Высота растения 40-50 см. " \
#                   "Соцветие каллы одиночное, белого цвета с восковым отливом, в форме початка, " \
#                   "окруженного крупным воронковидным листом."
#
#     CALLA_VERMEER_DESCRIPTION = "Калла Вермеер. Цена 410 руб.  Морозоустойчивость до -7С.Многолетнее " \
#                   "травянистое растение с клубневидным корневищем. Листья крупные," \
#                   " стреловидной формы, блестящие, восковые. Соцветие одиночное на длинном " \
#                   "цветоносе в виде кремово-желтого початка в обрамлении воронковидного, " \
#                   "слегка волнистого покрывала.  Высота растения 60-70 см. Цветение длительное, " \
#                   "июнь-сентябрь. Период цветения: июнь-август. Место посадки: солнце/полутень."
#
#     CALLA_CHAMELEON_PHOTO = "https://i.mycdn.me/i?r=AyH4iRPQ2q0otWIFepML2LxRpNHaRE83Idry9iysh9m3LQ"
#     CALLA_CAPTIN_MORELLI_PHOTO = "https://i.mycdn.me/i?r=AyH4iRPQ2q0otWIFepML2LxRokwSgPrsezL9_J6y_vVGQg"
#     CALLA_AMETIST_PHOTO = "https://i.mycdn.me/i?r=AyH4iRPQ2q0otWIFepML2LxRrHlEJMCSuJh5kk_tzAIpZw"
#     CALLA_CAPTIN_SAFARI_PHOTO = "https://i.mycdn.me/i?r=AyH4iRPQ2q0otWIFepML2LxRokwSgPrsezL9_J6y_vVGQg"
#     CALLA_CAPTIN_VENTURA_PHOTO = "https://i.mycdn.me/i?r=AyH4iRPQ2q0otWIFepML2LxRdD5PjNjorIDtPYCelRtrQg"
#     CALLA_VERMEER_PHOTO = "https://i.mycdn.me/i?r=AyH4iRPQ2q0otWIFepML2LxRAo0vbrCRl1q-RnvNbvFdHg"
#
#     USER_ID = "589219845582"
#
#     GROUP_GOBIGULI_ID = "70000002213582"
#     GROUP_FLOWERS_OUR_FLOWERS_ID = "70000002189774"
#     GROUP_TRATATULI_ID = "70000002228915"
#     GROUP_TRANSISTORS_ID = "70000001979493"
#
# class WebAddresses:
#     OK_LOGIN_PAGE_ADDRESS = "https://ok.ru/"
#
# class Pathes:
#     webDriverChromeLocalPath = "E://Selenium//chromedriver.exe"
#
class OkBotLocators:
    # header
    HEADER_LINKS = '//div[@class= "toolbar_nav_i_ic"]'
    # PASSWORD_FIELD = (By.XPATH, '//input[@id = "field_password"]')
    # ENTER_OK_BUTTON = (By.XPATH, '//input[@value= "Войти в Одноклассники"]')

    # messages
    TEST_GROUP_MESSAGES = '//div[@class= "title-okmsg"]//span[text()= "Тестовая группа"]'
    TEST_GROUP_MESSAGES_UNREAD = '//div[@class= "title-okmsg unread-okmsg"]//span[text()= "Тестовая группа"]'
    BUTTON_GROUP = '//div[@class= "group-okmsg"]//msg-keyboard-button'
    BUTTON_ADD = '//msg-keyboard-button'
    SCROLL_LAST_ELEMENT_BUTTON = '//msg-button[@data-l= "t,scrollToBottom"]'
    BUTTON_MAIN_MENU = '//div[@class= "group-okmsg"]//msg-parsed-text//span[text()="Главное меню"]'
    BUTTON_SHARE = '//div[@class= "group-okmsg"]//msg-parsed-text//span[text()="Поделиться"]'
    BUTTON_PIONS_ON_SPRING = '//div[@class= "group-okmsg"]//msg-parsed-text//span[text()="ПИОНЫ НА ВЕСНУ 2079"]'
    BUTTON_NARCISES_ON_SPRING = '//div[@class= "group-okmsg"]//msg-parsed-text//span[text()="НАРЦИССЫ НА ВЕСНУ 2023"]'
    BUTTON_CLEAR = '//div[@class= "group-okmsg"]//msg-parsed-text//span[text()="Очистить"]'
    BUTTON_SEND = '//div[@class= "group-okmsg"]//msg-parsed-text//span[text()="Отправить"]'
    BUTTON_ACCEPT = '//div[@class= "group-okmsg"]//msg-parsed-text//span[text()="Принять"]'
    BUTTON_DJORDJ_KIM_ACCOUNT = '//div[@class= "group-okmsg"]//msg-parsed-text//span[text()="джордж ким:584397600334"]'
    URL_ELEMENTS = '//a[@class= "link-okmsg url-okmsg"]'
    BUTTON_CLOSE_MESSAGES_FORM = '//div[@class= "toolbar-layer_controls_item js-close-layer"]'

    # test group locators
    BUTTON_ALL_ALBUMS = '//div[@class= "portlet group-new-photos-rb"]//div[@class= "portlet_h_inf"]'
    BUTTON_PHOTOS_IN_TOP_MENU_SIDE_BAR = '//a[@class= "mctc_navMenuSec ellip-menu-item "]'
    PION_ALBUMS_NAME = '//a[@title= "ПИОНЫ НА ВЕСНУ 2079"]'
    BUTTON_SEE_MORE = '//button[@class= "button-clean__0wfyv loader-btn__85lk6"]'

    # account locators
    ACCOUNT_PHOTO_CONTEXT_MENUS = '//div[@class= "actions-menu-wrapper__u3693"]'
    ACCOUNT_PHOTO_CONTEXT_MENUS_EDIT = '//menu[@class="actions-menu__u3693"]//*[text()= "Редактировать"]'
    ACCOUNT_PHOTO_DELETE_ALBUM = '//div[@class="portlet_controls"]'
    ACCOUNT_PHOTO_DELETE_ALBUM_CONFIRM_BUTTON = '//input[@data-l="t,confirm"]'
    ACCOUNT_ALBUM_NARCISES = '//a[@class = "title__x4tyv"][text()= "НАРЦИССЫ НА ВЕСНУ 2023"]'


    # wall
    PRIVAT_WALL = (By.XPATH, '//div[@id= "hook_Block_UserFeed"]')
    LAST_POST = '//div[@class = "feed-w"]'
    LAST_POST_CONTENT = '//div[@class = "media-text_cnt_tx emoji-tx textWrap"]'
    LAST_POST_PHOTO = '//img[@class = "collage_img"]'
    POST_TEXT = '//div[@class = "h-mod photo-layer_descr photo-layer_bottom_block"]//div[@tsid= "TextFieldText"]'
    ALL_COMMENTS = '//span[ @class = "js-text-full"]'
    ALL_COMMENTS_AUTHOR_NAMES = '// a[ @class = "comments_author-name o"]'
    ADVERSTISING_PAGE = (By.XPATH, '//*[@id="hook_Block_PhotoLayerDescription"]/div/div/div[1]')

    # group locators
    TEST_GROUP = (By.XPATH, '//div[@data-group-id= "70000000729023"]')
# class Users:
#     SERIC_USER_NAME = "Серик Обуманян"
#     USER_COMMENT_1 = "Вот такие вот гобигули"
#     USER_COMMENT_2 = "Вот такие вот цветочки"
#     USER_COMMENT_3 = "Вот такие вот Трататули"
#     USER_COMMENT_4 = "Вот такие вот транзисторы"
