class RequestsVariables:
    baseUrl = "http://10.243.8.118:31405"
    password = "Gfhjkm"

class OKData:

    loginSeric = 77712906977
    passwordSeric = 'lexusrx300'
    callaHameleonDescription = "Калла Хамелеон. Цена 410 руб. Морозостойкость -6. " \
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

    callaHameleonPhoto = "https://i.mycdn.me/i?r=AyH4iRPQ2q0otWIFepML2LxRpNHaRE83Idry9iysh9m3LQ"
    callaCaptinMorelliPhoto = "https://i.mycdn.me/i?r=AyH4iRPQ2q0otWIFepML2LxRokwSgPrsezL9_J6y_vVGQg"
    callaAmetistPhoto = "https://i.mycdn.me/i?r=AyH4iRPQ2q0otWIFepML2LxRrHlEJMCSuJh5kk_tzAIpZw"
    callaCaptinSafariPhoto = "https://i.mycdn.me/i?r=AyH4iRPQ2q0otWIFepML2LxRokwSgPrsezL9_J6y_vVGQg"

    userId = "589219845582"

    groupGobiguliId = "70000002213582"
    groupFlowersOurFlowersId = "70000002189774"
    groupTratatuliId = "70000002228915"
    groupTransistorsiId = "70000001979493"

class WebAddresses:
    okLoginPageAddress = "https://ok.ru/"

class Pathes:
    webDriverChromeLocalPath = "E://Selenium//chromedriver.exe"

class Locators:

    #authorization page
    loginField = '//input[@id = "field_email"]'
    passwordField = '//input[@id = "field_password"]'
    enterOKButton = '//input[@value= "Войти в Одноклассники"]'

    #top side bar
    topSideNavigationBarLocators = '//div[@class = "nav-side_i-w"]'

    #wall
    lastPost = '//div[@class = "feed-w"]'
    lastPostContent = '//div[@class = "media-text_cnt_tx emoji-tx textWrap"]'
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