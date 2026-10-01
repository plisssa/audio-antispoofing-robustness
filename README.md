# audio-antispoofing-robustness

Оценка устойчивости детекторов аудиодипфейков к сдвигу распределения, искажениям сигнала и
состязательным атакам. Детектор рассматривается как индикатор: он запускается на «родных»,
соседних и искажённых данных, и измеряется, где и при какой силе воздействия он начинает ошибаться.

## Состав

- `src/adfd/` — фреймворк (CLI `adfd`): адаптеры детекторов, искажения, атаки, метрики, конвейер.
- `tests/` — тесты фреймворка (`pytest`, без GPU и без скачивания данных).
- `configs/` — реестр моделей, сетки искажений и атак, конвейеры прогонов.
- `scripts/` — скрипты кластера (Slurm): окружение, загрузка корпусов, запуск конвейеров.
- `results/report/` — сводные таблицы первой волны (июль 2026).
- `results/degradation_grid_ci.csv` — сетка деградации ΔEER первой волны с 95 % ДИ.
- `results/wave2/` — сводные таблицы второй волны (сентябрь 2026), см. ниже.
- `analysis/` — воспроизведение ключевых чисел статьи из сводных таблиц.
- `figures/` — генерация рисунков статьи.
- `docs/` — блок-схемы методики и конвейера.

## Установка

```
python3.12 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/pip install -r requirements-gpu.txt
make test
```

Вторая строка достаточна для тестов и анализа; третья нужна для нейросетевых детекторов.
Подготовка кластера и тяжёлых моделей — `RUNBOOK.md` и `HEAVY_MODELS.md`.

## Модели

reference (LFCC + логистическая регрессия), RawNet2, AASIST, AASIST (базовая система ASVspoof 5),
AASIST3 (KAN), Spectra-AASIST3, SSL-AASIST (XLS-R 300M), XLSR-Mamba (XLS-R 300M),
Nes2Net (WavLM), Nes2Net-X (XLS-R 300M), линейные зонды на замороженных энкодерах
XLS-R 300M, WavLM-large, HuBERT-large, wav2vec2-large-lv60, WavLM-base-plus.
Реестр и источники чекпойнтов — `configs/models.yaml`, сильные и слабые стороны — `ARCHITECTURES.md`.

## Данные

ASVspoof 2019 LA, ASVspoof 2021 LA, In-the-Wild, ASVspoof 5 (оценочная часть, подвыборка
16 атак × 1500 + 6000 подлинных), корпуса шумов MUSAN, RIRS, ESC-50. В репозиторий не входят,
загружаются скриптами `scripts/download_*.sh` и `scripts/fetch_asvspoof5.py`.

## Эксперименты второй волны

Каждый эксперимент — конвейер в `configs/`, запускается одной задачей:
`sbatch scripts/slurm_pipeline.sh configs/<конвейер>.yaml`.

| Что | Конвейеры |
|---|---|
| Перенос атак, матрица 10 × 9 на ASVspoof 2019 и тонкая сетка на ITW и ASVspoof 2021 | `pipeline_matrix_*`, `pipeline_fine`, `pipeline_spectra_fix` |
| Согласованность градиентов источника и жертвы | `pipeline_align` |
| Зонды на других энкодерах | `pipeline_probes` |
| Чистое качество на 4000 записей на класс | `pipeline_large` |
| Сила атаки и пороги устойчивости | `pipeline_strength*`, `pipeline_thr_*` |
| Проверки маскировки градиента: SPSA, Square, диагностика градиента, функция потерь по разности логитов, старт от суррогата | `pipeline_blackbox`, `pipeline_square`, `pipeline_nes2net_diag`, `pipeline_margin_control`, `pipeline_warm*` |
| Защиты на входе и адаптивная атака (EOT) | `pipeline_defence*`, `pipeline_eot*` |
| ASVspoof 5, включая состязательные атаки Malafide и Malacopula | `pipeline_asv5*` |
| Перцептивное качество возмущений (PESQ, STOI) | `pipeline_thr_xlsr_linear_perc` |
| Разброс по случайным зёрнам у зондов | `pipeline_seeds_*` |

## Результаты второй волны

`results/wave2/` собирается из каталогов прогонов скриптом `analysis/collect_wave2.py`.
Каждая строка помечена группой и именем прогона (`run_group`, `run`), поэтому таблицы
можно фильтровать по эксперименту.

| Файл | Содержимое |
|---|---|
| `clean.csv` | чистые метрики: EER, AUC, minDCF, Cllr, min-Cllr |
| `sweep.csv` | деградация по искажениям, ΔEER с 95 % ДИ парного бутстрэпа |
| `attack.csv` | атаки: EER, ΔEER с ДИ, доля успеха атаки, фактический SNR |
| `transfer.csv` | перенос атак: источник, жертва, EER, ΔEER, AUC, Cllr |
| `defence.csv` | защиты: доля успеха до и после защиты, цена защиты на чистых данных |
| `alignment.csv` | косинус и знаковая согласованность градиентов по парам моделей |
| `gradcheck.csv` | диагностика градиента: детерминированность, точность линейного прогноза, сводка траекторий PGD |
| `perceptual.csv` | PESQ-WB и STOI атаки и шума той же энергии по бюджетам |
| `asv5_per_attack.csv` | EER на ASVspoof 5 по каждой атаке |

## Воспроизведение

```
python analysis/verify_stats.py --report results/report
python analysis/verify_wave2.py --results results/wave2
python figures/make_figures.py --results results/report --out figures/out
```

`verify_wave2.py` выводит: перенос при общем и разном энкодере по трём доменам (знаковый
критерий по источникам, регрессия с поправкой на чистый EER жертвы, перестановочный критерий),
связь переноса с согласованностью градиентов, доли успеха слабой и сильнейшей атаки по бюджетам,
проверки маскировки, PESQ, разброс по зёрнам и сравнение A17/A18 на ASVspoof 5.

Пересборка `results/wave2` из своих прогонов:
`python analysis/collect_wave2.py --root <каталог с runs*> --out results/wave2`.

## Метрики

EER, ΔEER с 95 % ДИ парного бутстрэпа, AUC, Cllr и min-Cllr, minDCF, доля успеха атаки (ASR),
перенос атак, знаковая согласованность градиентов, PESQ-WB, STOI.
