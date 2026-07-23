# Оптимизация маркетинговых кампаний для страховой компании

Дано большое множество клиентов (в общем случае — *субъектов*) и набор
кандидатных предложений для каждого из них. Нужно выбрать не более одного
предложения на клиента так, чтобы максимизировать суммарный ожидаемый доход
(EV) при соблюдении множества бизнес-ограничений (бюджеты, лимиты на число
контактов, ограничения по сегментам и т.д.), точная форма которых заранее не
известна. Пакет `offer_opt/` генерически разбирает ограничения, решает
получившуюся масштабную задачу распределения на GPU и сам же проверяет
собственный ответ — полное обоснование того, *почему* всё устроено именно
так, см. в `system_design_overview.md`.

## Требования

- Python 3.13 — рабочее окружение `.venv/` уже лежит в репозитории со всеми
  установленными зависимостями (torch, pandas, numpy, jsonschema, requests,
  pytest).
- Больше ничего не нужно, чтобы запустить 3 примера кейсов или тесты.
- Опционально: доступный OpenAI-совместимый LLM-эндпоинт (например, сервер
  vLLM) — нужен только для той части конвейера, которая обобщается на
  действительно новые/незнакомые датасеты — см. раздел «Использование
  настоящей LLM» ниже. Всё остальное работает и без него.

Все команды ниже подразумевают, что вы находитесь в корне репозитория.

## Быстрый старт

```bash
.venv/bin/python3 -m pytest -q      # запустить весь набор тестов (~10 минут)
```

```python
from offer_opt.pipeline import run_case
from offer_opt.device import get_device

result = run_case("hard", get_device(prefer_gpu=True))
print(result.verification)          # PASS/FAIL, суммарный EV, нарушения (если есть)
```

## Запуск тестов

- `pytest` — все тесты сразу (быстрые проверки + полные, "с полным бюджетом
  итераций", проверки качества решения — по умолчанию разделение по маркерам
  не настроено).
- `pytest -m slow` — только строгие проверки качества решения с полным
  бюджетом итераций.
- `pytest -m llm_integration` — только тесты, которым нужен реальный
  LLM-эндпоинт; они автоматически пропускаются, если переменная окружения
  `LLM_BASE_URL` не задана.

## Решение одного из 3 примеров кейсов (low / med / hard)

```python
from offer_opt.pipeline import run_case
from offer_opt.device import get_device

device = get_device(prefer_gpu=True)              # CUDA > MPS > CPU, что доступно
result = run_case("med", device, max_iters=400, repair_every=20)

print(result.verification)   # PASS/FAIL, суммарный EV, нарушения (если есть)
print(result.reference_ev)   # EV эталонного решения от заказчика — для сравнения
```

`result` — это `CaseResult`: `offer_table`, `constraint_set`, `solve_result`,
`verification`, `reference_ev`.

## Решение произвольного нового датасета

Это и есть настоящая точка входа для обобщённого решения — она вообще не
знает о «low/med/hard», только о путях к сырым файлам предложений и
ограничений:

```python
from offer_opt.pipeline import run_dataset
from offer_opt.device import get_device

result = run_dataset(
    "path/to/offers.csv",
    "path/to/constraints.csv",
    get_device(prefer_gpu=True),
    llm_client=None,       # по умолчанию NullClient -- см. «Использование настоящей LLM» ниже
    max_iters=300,
)

print(result.dims)           # обнаруженные названия измерений, например ("product", "channel", "segment")
print(result.trees)          # выведенная иерархия (DimensionTree) для каждого измерения
print(result.conflicts)      # найденные противоречия между ограничениями "предок/потомок", если есть
print(result.verification)
print(result.codegen_agrees) # согласился ли сгенерированный код проверки с verify.py
```

`result` — это `DatasetResult`: `offer_table`, `constraint_set`, `dims`,
`trees`, `conflicts`, `solve_result`, `verification`, `generated_checks`,
`codegen_agrees`.

Без `llm_client` всё, что символьный парсер/эвристики не могут уверенно
разобрать сами — принципиально новая строка типа ограничения, неоднозначная
колонка, иерархия измерения без каких-либо признаков в именовании — вызывает
громкую ошибку (`UnresolvedConstraintError` или аналогичную), а не тихое
угадывание. Именно это и решает подключение настоящей LLM.

## Использование настоящей LLM

Укажите любой OpenAI-совместимый chat-completions эндпоинт (например, сервер
vLLM с Qwen или аналогичной моделью):

```bash
export LLM_BASE_URL="http://your-server:8000"
export LLM_API_KEY="..."          # только если эндпоинт требует ключ
```

```python
from offer_opt.llm.client import VLLMOpenAIClient

client = VLLMOpenAIClient()        # читает LLM_BASE_URL / LLM_API_KEY из окружения
assert client.health_check()       # быстрая проверка связи (GET /v1/models)

result = run_dataset(offers_path, constraints_path, device, llm_client=client)
```

Сначала имеет смысл проверить сам эндпоинт отдельно: `pytest -m llm_integration -v`.

Для разработки/тестирования без реального эндпоинта используйте
`offer_opt.llm.client.NullClient` (принудительно включает полностью
символьный путь — это поведение по умолчанию) или `FakeLLMClient`
(сценарный тестовый дублёр, возвращающий заранее заданные ответы — примеры
см. в любом тестовом файле под `tests/`, например
`tests/test_generalization.py`).

Также есть небольшой инструмент для оценки (`offer_opt.llm.evaluate`),
позволяющий измерить точность/задержку настоящего клиента на задаче
классификации ограничений на фиксированном наборе кейсов
(`tests/fixtures/constraint_classification_cases.json`), в том же стиле, что
и `vendor-examples/examples/prompt_lab.py`.

## Замер производительности (бенчмарк)

```python
from offer_opt import metrics
from offer_opt.device import get_device

report = metrics.benchmark("hard", get_device(prefer_gpu=True))
print(report)
```

В `baselines/phase0_baseline.md` зафиксированы эталонные показатели
собственного решателя этого репозитория (CPU и MPS, все 3 кейса), снятые
*до* начала работы по обобщению — точка сравнения для вопроса «не стало ли
хуже?».

## Структура проекта

```
offer_opt/                    сам пакет
  schema.py, scope.py           каноническая модель данных + сопоставление scope с учётом иерархии
  constraints.py                 разбор строки ограничения -> ConstraintSpec / ParameterSpec
  features.py                     построение канонической таблицы предложений (по кейсам + генерически)
  discovery/                       разрешение схемы, вывод иерархии измерений, поиск конфликтов
  llm/                              заменяемый LLM-клиент (Null/Fake/настоящий vLLM), промпты, кэш, бюджет
  codegen/                          генерация + песочница для кода проверки по каждому ограничению
  solver/                           сам оптимизатор (релаксация Лагранжа + локальный поиск + repair)
  verify.py                        эталонный модуль проверки ограничений
  pipeline.py                      run_case() / run_dataset() -- точки входа, описанные выше
  metrics.py                       инструмент бенчмарка + восстановление эталонных решений
tests/                        ~150 тестов; fixtures/ содержит синтетические датасеты для проверки обобщения
case_1_low/, case_2_med/, case_3_hard/   3 примера кейсов от заказчика
vendor-examples/              собственные материалы школы ИИ по онбордингу (соглашения по LLM-инфраструктуре)
baselines/                    снимок производительности до начала работы по обобщению
notebooks/solution.ipynb      исходный черновой ноутбук
system_design_overview.md     целевая архитектура этого пакета простым языком
Ingosstrakh_task_v20260428.pdf   исходная постановка задачи
```

## Что почитать дальше

`system_design_overview.md` простым языком объясняет, какую задачу это
решение на самом деле решает и почему оно устроено именно так — начните с
этого документа, если кода под `offer_opt/` окажется недостаточно для
понимания.
