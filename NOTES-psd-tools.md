# Нода comfyui-psd-export в образе runpod-baked — решение и обоснование

Дата: 18.08.2026. Контекст: нужно было добавить ноду экспорта PSD в baked-образ
и решить, куда прописать её зависимость `psd-tools`.

## Короткий ответ

**Во freeze образа ничего не добавляем.** `psd-tools==1.14.2` уже там
(строка 308 `requirements.freeze.txt`) — тянется ради
`ComfyUI_LayerStyle_Advance` (у него в requirements `psd-tools` без пина).
Вместо этого починили саму ноду под актуальный API psd-tools.

## Почему не «просто дописать зависимость»

1. **Пакет уже в образе.** Дописывать во freeze нечего.
2. **Новых транзитивных зависимостей нет.** Метаданные колёс 1.14.2 и 1.18.0
   идентичны: `typing-extensions`, `attrs`, `Pillow`, `numpy` — всё уже во
   freeze. `aggdraw / scipy / scikit-image` только в extra `[composite]`,
   нода их не использует.
3. **`psd-tools>=1.18` в requirements ноды было неверно.** Код был написан под
   API **1.10.x**. Сигнатуры сломались в 1.11.0 и с тех пор стабильны
   (проверено на 1.10.13 / 1.11.1 / 1.12.2 / 1.13.1 / 1.14.2/3 / 1.15 / 1.16 /
   1.17.4 / 1.18.0):

   | Вызов | 1.10.x | 1.11+ |
   |---|---|---|
   | `PixelLayer.frompil(img, psd, layer_name=…)` | OK | параметр называется `name`; `layer_name` **молча** уходит в `**kwargs` → **все слои называются "Layer"** |
   | `Group.new(name, open_folder, parent=…)` | OK | сигнатура `(parent, name, open_folder)` → `TypeError: got multiple values for 'parent'` при `group_per_prompt=True` |
   | `ChannelData / MaskData / ChannelInfo` (ручная маска слоя) | OK | OK — этот кусок API не менялся |

   То есть апгрейд до 1.18 сломал бы ноду тихо: PSD сохранился бы, но без имён
   слоёв — а имена слоёв и есть весь смысл ноды.

## Отвергнутый вариант: пин `psd-tools==1.10.13` во freeze

- Полная пересборка venv (~520 пакетов) — Сценарий 3 из PLAYBOOK, самый долгий.
- `ComfyUI_LayerStyle_Advance` сейчас живёт на 1.14.2; его requirements не
  пинят версию, значит pip не вернул бы её обратно — пак вслепую съезжает на
  четыре минорных версии назад.
- Расхождение с прод-venv ради одной ноды.

## Что сделано в ноде

- `PixelLayer.frompil(..., layer_name=)` → `name=` / создание без имени.
- `Group.new(label, open_folder=True, parent=psd)` → `Group.new(psd, open_folder=True)`.
- Guard на импорте: на psd-tools < 1.11 нода отказывается грузиться с внятной
  ошибкой вместо тихого «все слои Layer».
- `requirements.txt`, `pyproject.toml`, README: `>=1.18` → `>=1.11`.
- **Побочно найден и починен второй баг: кириллические имена слоёв роняли
  сохранение.** `psd.save()` пишет legacy-имя слоя в macroman → на «джинсы»
  падал `UnicodeEncodeError`. Добавлен `_set_layer_name()`: юникодное имя
  пишется в тегированный блок `luni` (его и читает Photoshop), а legacy-поле
  получает macroman-safe версию. Проверено — имена читаются обратно корректно.

## Проверка

Реальный код ноды прогнан без ComfyUI (заглушки `folder_paths` и `torch`) в
venv с psd-tools 1.14.2 и 1.18.0: кириллические имена, группа на промт с
несколькими детекциями, маски слоёв, `only_first_visible`, `crop_to_mask`,
`invert_masks`, вариант без имён. Всё сходится на обеих версиях; на 1.10.13
guard срабатывает как задумано.

## Доставка в образ

- Репозиторий: `https://github.com/wikinikiwings/psd_nodes.git`,
  папка в `custom_nodes` — `comfyui-psd-export`.
- Строка добавлена в `nodes.list` (после собственных паков).
- Папка добавлена в массив `OWN_NODES` в `start.sh` → `UPDATE_OWN_NODES=1`
  обновляет ноду на поде без пересборки base.
- Пересборка: правился `nodes.list`, а не freeze/секции 1-7 → **Сценарий 2**,
  venv берётся из кэша, пересобирается хвост с шага 8.
- Обкатка до пересборки: у ноды нет неудовлетворённых зависимостей, поэтому её
  можно гонять через env `EXTRA_NODES` (clone без pip) на текущем образе.
