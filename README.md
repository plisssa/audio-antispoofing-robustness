# audio-antispoofing-robustness

Оценка устойчивости детекторов аудиодипфейков к сдвигу распределения, искажениям сигнала и
состязательным атакам. 

## Состав

- `configs/` — конфигурации прогонов: реестр моделей, сетки искажений и атак, пайплайн.
- `scripts/` — скрипты кластера (Slurm): подготовка окружения, загрузка корпусов, прогоны sweep и attack.
- `results/report/` — сводные таблицы финальной выгрузки: чистое качество, деградация по искажениям,
  атаки, перенос атак.
- `results/degradation_grid_ci.csv` — полная сетка деградации ΔEER (модель × домен × искажение ×
  уровень) с 95 % доверительными интервалами парного бутстрэпа, 1093 строки.
- `figures/` — генерация всех рисунков статьи из `results/report`.
- `analysis/verify_stats.py` — воспроизведение ключевых чисел статьи из сводных таблиц
  (ранговая корреляция Спирмена, знаковый критерий «атака против шума», перенос PGD, калибровка).
- `analysis/prisma_diagram.py` — построение диаграммы PRISMA по заданным числам отбора.

## Данные

Корпуса ASVspoof 2019 LA, ASVspoof 2021 LA и In-the-Wild загружаются скриптами из `scripts/`
и в репозиторий не входят. Оценка проводится на сбалансированных подвыборках (800 записей на класс).

## Модели

reference (LFCC + логистическая регрессия), RawNet2, AASIST, AASIST3 (KAN), Spectra-AASIST3,
SSL-AASIST (XLS-R 300M), XLSR-Mamba (XLS-R 300M), Nes2Net (WavLM). Реестр и источники чекпойнтов —
в `configs/models.yaml`.

## Воспроизведение

```
pip install -r requirements.txt
python analysis/verify_stats.py --report results/report
python figures/make_figures.py --results results/report --out figures/out
python figures/make_spectrogram.py --out figures/out
```

Прогоны экспериментов выполняются на кластере: `scripts/slurm_sweep.sh`, `scripts/slurm_attack.sh`.

## Метрики

EER, ΔEER с 95 % ДИ парного бутстрэпа, AUC, Cllr, minDCF, доля успеха атаки (ASR), перенос атак.
