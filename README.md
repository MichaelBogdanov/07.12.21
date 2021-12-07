2.2
<br>
Для корректной работы нужно установить ряд зависимостей, а именно:
  1. Язык программирования Python 3.8.7;
  2. Менеджер пакетов Pip;
  3. Cистема автоматизации сборки программного обеспечения из исходного кода CMake (а также добавить папку ../CMake/bin в PATH);
  4. Пакеты для языка Python из файла requirements.txt (pip3 install -r requirements.txt).
<br>
После установки всех зависимостей можно запускать файл main.py.
Файл попросит вас выбрать видеофайл, который нужно обработать и произведёт его обработку.
Реализована возможность загрузки фотографий людей в формате (id, фото) для последующего их распознавания, для этого фотографии людей нужно поместить в папку faces, где название файла - id человека.
После обнаружения знакомого человека на видео, программа запишет сведения об этом в файл discovered.txt, а также отправит данные на сервер.
