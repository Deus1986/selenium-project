from selenium.webdriver.common.by import By


class RequestsVariables:
    BASE_URL = "http://10.243.8.118:31405"
    PASSWORD = "Gfhjkm"

class OKData:

    LOGIN_SERIC = 77712906977
    PASSWORD_SERIC = 'lexusrx300'
    CALLA_CHAMELEON_DESCRIPTION = "Калла Хамелеон. Цена 410 руб. Морозостойкость -6. " \
                  "Многолетнее травянистое растение с клубневидным корневищем. Листья крупные, " \
                  "стреловидной формы, блестящие, восковые. Соцветие одиночное на длинном цветоносе " \
                  "в виде кремово-желтого початка в обрамлении воронковидного, слегка волнистого покрывала. " \
                  "Окраска разнообразная: белая, желтая, розовая, карминная, лиловая. Высота прямых цветоносов " \
                  "достигает 60 см. Листья стреловидные, остроконечные или в форме сердца, темно-зеленые с белым крапом. "

    callaCaptinMorelliDescription = "Калла Кэптин Морелли. Цена 410 руб. Морозоустойчивость до -7С." \
                                    "Очень элегантный цветок, одетый в нежное желтое покрывало с легким " \
                                    "фиолетовым тоном и со светлыми прожилками. Листья темно-зеленого цвета с " \
                                    "редким, белым крапом и восковым блеском. Калла - многолетнее травянистое " \
                                    "растение, которое украсит не только сад, но и дом. Представляет собой небольшой " \
                                    "травянистый куст шириной 30 -35см и высотой цветоноса до 60 см."

    callaAmetistDescription = "Калла Аметист. Цена 410 руб. Морозоустойчивость до -7С." \
                  "Многолетний, крупноцветковый сорт. Высота растения 60-70 сантиметров. " \
                  "Период цветения: июнь-июль-август. Цветок крупного размера, в форме свечи, " \
                  "тёмно-фиолетового цвета. Высота растения 60-70 сантиметров. " \
                  "Период цветения: июнь-июль-август."

    callaCaptinSafariDescription = "Калла Кэптин Сафари. Цена 410 руб. Морозоустойчивость до -7С." \
                  "Растение представляет собой травянистое многолетнее клубневое " \
                  "красивоцветущее растение. Высота взрослого растения достигает 50-80 " \
                  "сантиметров в высоту. Листья вытянутые, зеленого цвета с белым вкраплением. " \
                  "Цветки в форме свечи, желто-оранжево-розовые."

    callaCaptinVenturaDescription = "Калла Кэптин Вентура .Цена 410 руб. Морозоустойчивость до -6С. " \
                  "Многолетнее травянистое растение с клубневидным корневищем. Высота растения 40-50 см. " \
                  "Соцветие каллы одиночное, белого цвета с восковым отливом, в форме початка, " \
                  "окруженного крупным воронковидным листом."

    callaVermeerDescription = "Калла Вермеер. Цена 410 руб.  Морозоустойчивость до -7С.Многолетнее " \
                  "травянистое растение с клубневидным корневищем. Листья крупные," \
                  " стреловидной формы, блестящие, восковые. Соцветие одиночное на длинном " \
                  "цветоносе в виде кремово-желтого початка в обрамлении воронковидного, " \
                  "слегка волнистого покрывала.  Высота растения 60-70 см. Цветение длительное, " \
                  "июнь-сентябрь. Период цветения: июнь-август. Место посадки: солнце/полутень."

    CALLA_CHAMELEON_PHOTO = "https://i.mycdn.me/i?r=AyH4iRPQ2q0otWIFepML2LxRpNHaRE83Idry9iysh9m3LQ"
    callaCaptinMorelliPhoto = "https://i.mycdn.me/i?r=AyH4iRPQ2q0otWIFepML2LxRokwSgPrsezL9_J6y_vVGQg"
    callaAmetistPhoto = "https://i.mycdn.me/i?r=AyH4iRPQ2q0otWIFepML2LxRrHlEJMCSuJh5kk_tzAIpZw"
    callaCaptinSafariPhoto = "https://i.mycdn.me/i?r=AyH4iRPQ2q0otWIFepML2LxRokwSgPrsezL9_J6y_vVGQg"
    callaCaptinVenturaPhoto = "https://i.mycdn.me/i?r=AyH4iRPQ2q0otWIFepML2LxRdD5PjNjorIDtPYCelRtrQg"
    callaVermeerPhoto = "https://i.mycdn.me/i?r=AyH4iRPQ2q0otWIFepML2LxRAo0vbrCRl1q-RnvNbvFdHg"

    USER_ID = "589219845582"

    groupGobiguliId = "70000002213582"
    groupFlowersOurFlowersId = "70000002189774"
    groupTratatuliId = "70000002228915"
    groupTransistorsiId = "70000001979493"

class WebAddresses:
    OK_LOGIN_PAGE_ADDRESS = "https://ok.ru/"

class Pathes:
    webDriverChromeLocalPath = "E://Selenium//chromedriver.exe"

class Locators:

    #authorization page
    LOGIN_FIELD = (By.XPATH, '//input[@id = "field_email"]')
    PASSWORD_FIELD = (By.XPATH, '//input[@id = "field_password"]')
    ENTER_OK_BUTTON = (By.XPATH, '//input[@value= "Войти в Одноклассники"]')

    #top side bar
    TOP_SIDE_NAVIGATION_BAR_LOCATORS = (By.XPATH, '//div[@class = "nav-side_i-w"]')
    TOP_SIDE_NAVIGATION_BAR_LOCATORS_WITHOUT_XPATH = '//div[@class = "nav-side_i-w"]'

    #wall
    PRIVAT_WALL = (By.XPATH, '//div[@id= "hook_Block_UserFeed"]')
    LAST_POST = '//div[@class = "feed-w"]'
    LAST_POST_CONTENT = (By.XPATH, '//div[@class = "media-text_cnt_tx emoji-tx textWrap"]')
    lastPostPhoto = '//img[@class = "collage_img"]'
    postText = '//div[@class = "h-mod photo-layer_descr photo-layer_bottom_block"]//div[@tsid= "TextFieldText"]'
    allComments = '// span[ @class = "js-text-full"]'
    allCommentsAuthorNames = '// a[ @class = "comments_author-name o"]'

    #group locators
    gobiguliLocator = '//div[@data-group-id= "70000002213582"]'
    flowersOurFlowersLocator = '//div[@data-group-id= "70000002189774"]'
    tratatuliLocator = '//div[@data-group-id= "70000002228915"]'
    transistorsLocator = '//div[@data-group-id= "70000001979493"]'
class Users:
    sericUserName = "Серик Обуманян"
    userComment1 = "Вот такие вот гобигули"
    userComment2 = "Вот такие вот цветочки"
    userComment3 = "Вот такие вот Трататули"
    userComment4 = "Вот такие вот транзисторы"