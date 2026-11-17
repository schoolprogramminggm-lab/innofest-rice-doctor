# Тестовые фото

Два маленьких фото (по 256 пикселей по длинной стороне) для `tests/test_real_model.py`.

| Файл | Что на фото | Источник | Автор | Лицензия |
|---|---|---|---|---|
| `rice_leaf.jpg` | Лист риса с пирикуляриозом (blast) | Paddy Doctor, папка `blast` ([Kaggle](https://www.kaggle.com/competitions/paddy-disease-classification)) | Petchiammal A., Briskline Kiruba S., Murugan D., Pandarasamy A. | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `not_rice.jpg` | Две кошки (не лист риса) | Open Images V7, фото [Flickr](https://www.flickr.com/photos/johnleach/3296181822) | John Leach | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) |

Фото `not_rice.jpg` не входит в обучающую и проверочную выборки класса `not_rice`.
Фото `rice_leaf.jpg` взято из проверочной части (`data/val`), сеть его при обучении не видела.
