2.1
<br>
Для того, чтобы запустить проект tracking необходимо в командной строке перейти в эту папку и при помощи команды 
`docker-compose up --build -d` запустить контейнеры. Проект состоит из PHP и веб-сервера nginx. 
Чтобы проверить работоспособность сервера по получению данных о местоположении болида необходимо отправить запрос `127.0.0.1?x=A&y=B`, где вместо A и В необходимо вставить свои числа, после этого можно убедиться, что в файл car_coord.txt, в последнюю строку, записалась информация в формате `x, y`. 
Для того, чтобы данные могло отправлять только одно устройство реализованно следующее: в файл id_cli.txt записывается ip устройста, с которово пришел запрос, если это файл пуст или дата его последнего изменения более 30 секунд, то он перезаписывается. Если же в нем уже есть данные, то ip адрес с файла проверяется с ip адресом пользователя, если совпадает, то в файл car_coord.txt дописывается информация о местопложении болида. Также сервер был залит на VPS хостинг и доступен по адресу `80.78.246.71?x=A&y=B`.
Для запуска мобильного приложения необходимо его установить, он называется `application.apk`. После установки необходимо его открыть и появится экран с гоночным треком и с графическим отображением болида. Если в течении 10 секунд данные с сервера так и не пришли, то выводится сообщение о том, что потеряно соединение с сервером. Для запуска API необходимо запустить Open Server. Также необходимо поместить папку с проектом в директорию `OSPanel/domain/localhost` и запустить при помощи Open Server.
<br>
2.2
<br>
Для корректной работы нужно установить ряд зависимостей, а именно:
  1. Язык программирования Python 3.8.7;
  2. Менеджер пакетов Pip;
  3. Cистема автоматизации сборки программного обеспечения из исходного кода CMake (а также добавленная папка `../CMake/bin в PATH`);
  4. Пакеты для языка Python из файла requirements.txt (`pip3 install -r requirements.txt`).
<br>
После установки всех зависимостей можно запускать файл main.py.
Файл попросит вас выбрать видеофайл, который нужно обработать и произведёт его обработку.
Реализована возможность загрузки фотографий людей в формате (id, фото) для последующего их распознавания, для этого фотографии людей нужно поместить в папку faces, где название файла - id человека.
Далее открывается окно потокового просмотра видеоролика, где программа будет отмечать найденные лица. Если есть необходимость прервать выполнение, достаточно нажать клавишу q.
После обнаружения знакомого человека на видео, программа запишет сведения об этом в файл discovered.txt, а также отправит данные на сервер.
После работы программы в каталоге проекта можно найти файл, заканчивающийся на ...outputVideo.mp4, который содержит в себе видео с отмеченными лицами.
