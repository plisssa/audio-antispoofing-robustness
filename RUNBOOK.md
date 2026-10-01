# RUNBOOK — команды запуска и проверки

Все команды выполняются из корня проекта (`adfd-robustness/`).

## 1. Установка окружения

```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install -e .
./.venv/bin/adfd list                       # детекторы, искажения, атаки
```

На кластере cHARISMa для тяжёлых детекторов:

```bash
./.venv/bin/python -m pip install -r requirements-gpu.txt
```

Проверка окружения:

```bash
./.venv/bin/python -c "import numpy,scipy,soundfile,sklearn,pandas; print('core ok')"
./.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

## 2. Дымовой прогон без данных (CPU)

```bash
./.venv/bin/adfd bootstrap --n 120
./.venv/bin/adfd verify-dataset --manifest data/bootstrap/manifest.csv --expect-total 240
./.venv/bin/adfd run --detector reference
./.venv/bin/adfd sweep --detector reference melspec
./.venv/bin/adfd attack --detector torch_reference
./.venv/bin/python -m pytest -q
```

## 3. Датасеты: скачивание и проверка размера

```bash
bash scripts/download_asvspoof2019_la.sh
bash scripts/download_in_the_wild.sh
```

Построить манифест:

```bash
./.venv/bin/adfd manifest --dataset asvspoof2019_la --root data/raw/asvspoof2019_la/LA --split eval --out data/asvspoof2019_la/eval.csv
./.venv/bin/adfd manifest --dataset in_the_wild --root data/raw/in_the_wild/release_in_the_wild --out data/in_the_wild/manifest.csv
```

Проверить правильность размера (счётчики, битые/короткие файлы, длительность):

```bash
./.venv/bin/adfd verify-dataset --manifest data/asvspoof2019_la/eval.csv --expect-total 71237 --expect-bona 7355 --expect-spoof 63882
./.venv/bin/adfd verify-dataset --manifest data/in_the_wild/manifest.csv --expect-total 31779 --expect-bona 19963 --expect-spoof 11816
```

Эталонные размеры:

| датасет | split | total | bona_fide | spoof |
|---|---|---|---|---|
| ASVspoof 2019 LA | train | 25380 | 2580 | 22800 |
| ASVspoof 2019 LA | dev | 24844 | 2548 | 22296 |
| ASVspoof 2019 LA | eval | 71237 | 7355 | 63882 |
| In-the-Wild | eval | 31779 | 19963 | 11816 |

`status: OK` и нулевые `missing_files`/`unreadable_files` означают, что набор скачан и сопоставлен правильно.

## 4. Проверка правильности модели

`verify-model` грузит детектор, считает EER/AUC/Cllr на eval-сплите и сверяет с заявленным.

```bash
./.venv/bin/adfd verify-model --detector reference --manifest data/bootstrap/manifest.csv --expected-eer 0.0 --tolerance 0.05
./.venv/bin/adfd verify-model --detector aasist3 --manifest data/asvspoof2019_la/eval.csv --expected-eer 0.012 --tolerance 0.01
```

Что проверяется:
- `available: True` — зависимости и веса на месте;
- `eer/auc/cllr` близки к заявленным в статье/README → пайплайн воспроизведён;
- **ориентация score**: если `eer>0.5`, печатается предупреждение и `1-eer` — значит `spoof_index` перевёрнут, поправить в `configs/models.yaml`;
- `metadata` фиксирует repo/checkpoint/spoof_index для отчёта.

Если EER далеко от ожидаемого: проверить sample rate (16 кГц моно), нормализацию, crop, версию весов и протокол (см. README).

## 5. Эксперимент: искажения, сила, кросс-модельное сравнение

```bash
./.venv/bin/adfd sweep --detector reference melspec --manifest data/in_the_wild/manifest.csv
./.venv/bin/adfd sweep --detector reference --only noise codec transmission --clusters 8
```

Артефакты в `runs/<ts>_*/`: `aggregate.csv` (EER/ΔEER/Cllr по уровню силы), `response_per_distortion.csv` (монотонность отклика), `model_distortion_pivot.csv` и `model_cluster_pivot.csv` (какая модель на каком шуме/кластере ошибается), `error_table.csv` (пер-сэмпл).

## 6. Adversarial (FGSM/PGD)

```bash
./.venv/bin/adfd attack --detector torch_reference
./.venv/bin/adfd attack --detector aasist --only pgd          # на кластере
```

`attack_aggregate.csv`: EER, ΔEER и attack success rate по сетке epsilon.

## 7. Кластер cHARISMa (SLURM) — полный цикл

Логин-сервер только для сборки/загрузки/лёгких тестов; все расчёты идут через Slurm. Студентам недоступны узлы type_e/f (A100/H100), поэтому GPU-задачи идут на V100 (`--constraint=type_a|type_b`) в очереди `normal`.

### 7.1. Загрузка проекта (с Mac, локально)
```bash
rsync -avz --exclude .venv --exclude data --exclude runs --exclude .git \
  -e "ssh -p 2222 -i <ключ>.pem" \
  ./ <логин>@cluster.hpc.hse.ru:adfd-robustness/
```
Альтернатива: `scp -P 2222 -i <ключ>.pem adfd-robustness.zip <логин>@cluster.hpc.hse.ru:` и затем `unzip` на кластере.

### 7.2. Проверки на кластере
```bash
mp                 # проекты: если их несколько, добавлять -A <ID> к sbatch
checkquota         # хватает ли места (датасеты большие)
module avail Python
freenodes          # свободные ресурсы
```

### 7.3. Окружение (логин-сервер; numpy/scipy/torch берём из модуля)
Системный gcc на cHARISMa старый, поэтому numpy/scipy НЕ собираются из исходников. Используем модуль `Python/PyTorch_GPU_v2.4` (Python 3.12 + numpy/scipy/torch уже собраны) и venv поверх него:
```bash
cd adfd-robustness
bash scripts/setup_cluster.sh
```
Скрипт делает: `module load Python/PyTorch_GPU_v2.4` → `python -m venv --system-site-packages .venv` → ставит только `soundfile`/`scikit-learn` готовыми колёсами (`--only-binary=:all:`) → `pip install -e . --no-deps`. torch/transformers берутся из модуля.

**Вычислительные узлы без интернета.** Датасеты и веса HuggingFace качаем только на логин-сервере (в tmux); в GPU-задаче стоит `HF_HUB_OFFLINE=1`.

Быстрая проверка всего конвейера на кластере БЕЗ скачивания (синтетика — гарантированно работает):
```bash
module purge; module load Python/PyTorch_GPU_v2.4; source .venv/bin/activate
adfd bootstrap --n 200
sbatch scripts/slurm_sweep_cpu.sh reference data/bootstrap/manifest.csv
mj                       # дождаться завершения
cat runs/reference_*_sweep/aggregate.csv | head    # реальные артефакты
```

### 7.4. Скачивание датасетов (в tmux, чтобы пережить дисконнект)
```bash
tmux new -s dl
bash scripts/download_in_the_wild.sh
bash scripts/download_asvspoof2019_la.sh
# Ctrl+b d — отсоединиться; tmux attach -t dl — вернуться
```

### 7.5. Манифест + проверка размера
```bash
./.venv/bin/adfd manifest --dataset in_the_wild --root data/raw/in_the_wild/release_in_the_wild --out data/in_the_wild/manifest.csv
./.venv/bin/adfd verify-dataset --manifest data/in_the_wild/manifest.csv --expect-total 31779 --expect-bona 19963 --expect-spoof 11816
```

### 7.6. Быстрая интерактивная проверка GPU и модели (перед длинной задачей)
```bash
srun -p normal --constraint="type_a|type_b" --gpus=1 --cpus-per-task=4 --pty bash
source .venv/bin/activate
python -c "import torch; print('cuda', torch.cuda.is_available())"
./.venv/bin/adfd verify-model --detector aasist3 --manifest data/in_the_wild/manifest.csv
exit
```
Если `eer>0.5` — поправить `spoof_index` в `configs/models.yaml` и проверить снова.

### 7.7. Запуск длинной задачи
```bash
sbatch scripts/slurm_sweep_cpu.sh reference data/in_the_wild/manifest.csv     # сначала валидация пайплайна (CPU)
sbatch scripts/slurm_sweep.sh    aasist3   data/in_the_wild/manifest.csv     # затем реальная модель (GPU V100)
# если mp показал несколько проектов:
sbatch -A <proj_id> scripts/slurm_sweep.sh aasist3 data/in_the_wild/manifest.csv
```

### 7.8. Мониторинг
```bash
mj                 # свои задачи
mj --start         # когда ориентировочно стартуют
squeue --me
tail -f runs/slurm-adfd-sweep-*.log
scancel <job_id>   # остановить
```
Также статус и загрузка GPU — в HPC TaskMaster: https://lk.hpc.hse.ru

### 7.9. Устойчивость к дисконнекту и продолжение с места остановки
- **Дисконнект не важен**: `sbatch`-задача выполняется на узле независимо от вашей SSH-сессии, можно выйти.
- **Продолжение с места**: все длинные команды используют фиксированный `--run-dir` (скрипты задают его автоматически по модели и датасету). Результаты пишутся инкрементально в `clean.partial.*.csv` / `sweep.partial.*.csv`. Если задача упала или была снята — просто запустите `sbatch` тот же скрипт ещё раз: уже посчитанные файлы пропускаются, прогон продолжается.
- **Профилактика 30 июня**: в скриптах стоит `#SBATCH --requeue`, поэтому задача автоматически перезапустится после профилактики и продолжит с чекпойнта.
- Важно: при продолжении не менять `configs/distortions.yaml` и состав `--detector`, иначе частичные результаты не совпадут.

### 7.10. AASIST3 (репозиторий mtuciru/AASIST3, веса с HuggingFace)
Адаптер `aasist3` импортирует класс из склонированного репозитория (`from model import aasist3`, `from_pretrained("MTUCI/AASIST3")`), вход — сырая волна 64600 семплов, выход — логиты (B,2) без softmax. Узлы офлайн → веса (AASIST3 + внутренний wav2vec2) кэшируем на логине.

```bash
mkdir -p third_party && ln -s ~/AASIST3 third_party/AASIST3
curl -sI https://huggingface.co | head -1        # есть ли HF-интернет на логине?
bash scripts/precache_aasist3.sh                 # если есть — качает и кэширует веса
```
Если логин не видит HF — скачать на Mac `huggingface-cli download MTUCI/AASIST3 --local-dir aasist3_weights` и `rsync` папку на кластер, а в `configs/models.yaml` указать `repo:` = путь к ней.

Проверка перед длинным прогоном (в интерактивной GPU-сессии) — обязательно подтвердить `spoof_index` (выход без softmax, ориентация неизвестна):
```bash
srun -A <проект> -p rocky --constraint="type_a|type_b" --gpus=1 -c 4 --pty bash
module purge; module load Python/PyTorch_GPU_v2.4; source .venv/bin/activate
export HF_HUB_OFFLINE=1 PYTHONPATH=third_party/AASIST3
adfd verify-model --detector aasist3 --manifest data/in_the_wild/manifest_small.csv
```
Если `eer>0.5` — поменять `spoof_index` в `configs/models.yaml` на `0`. Затем: `sbatch scripts/slurm_sweep.sh aasist3 data/in_the_wild/manifest_small.csv`.

Если на этом окружении (torch 2.4) импорт/запуск AASIST3 падает по версиям — пересобрать venv на модуле `Python/Google_Colab_GPU_2025` (torch 2.6 + transformers 4.52, совпадает с requirements репозитория).

### 7.11. Прочие тяжёлые модели
`ssl_aasist`, `xlsr_mamba`, `nes2net` гонять отдельными задачами. Заранее положить репозитории в `third_party/<repo>`, скачать веса, для XLS-R-моделей — `xlsr2_300m.pt` в каталог репозитория, поставить fairseq/mamba-ssm/s3prl, и обязательно `verify-model` для подтверждения `spoof_index`.

## 8. Обратная задача (предсказание ошибки по параметрам)

Последний пункт плана: по таблице «параметры семпла + параметры шума → ошибка детектора» обучается классификатор, предсказывающий ошибку, и показывается, какие параметры её определяют.
```bash
adfd inverse --error-table runs/aasist3_itw_light/error_table.csv runs/reference_manifest_sweep/error_table.csv --out runs/inverse
```
Артефакты: `inverse_importance.csv` (важность признаков), `failure_by_distortion.csv` (error rate по модель×искажение), `inverse_report.json` (AUC предсказания ошибки, доля ошибок, топ-признаки). Высокий AUC = ошибки систематичны и предсказуемы по параметрам; топ-признаки = что именно ломает детектор (сила искажения, длительность, тишина, спектр и т.д.).

## 9. Воспроизводимость

Каждый прогон пишет `config.yaml`, `environment.txt` (pip freeze + платформа), `meta.json` (архитектура, repo, checkpoint, commit hash, дата). Перед запуском зафиксировать версию:

```bash
git rev-parse HEAD
./.venv/bin/python -m pip freeze > environment.lock.txt
```

## 10. Полный конвейер одной командой (`adfd pipeline`)

Весь эксперимент описан в `configs/pipeline.yaml` как последовательность этапов
(`subset → verify → sweep по моделям на родном домене и ITW → реальные искажения/телефония/EnCodec →
атаки FGSM/PGD с переносом → опциональные тяжёлые модели → combine/inverse/eda`).

### 10.1. Подготовка на login-узле (интернет есть)

```bash
tmux new -s prep
bash scripts/prepare_login.sh
```

Скрипт скачивает датасеты (In-the-Wild, ASVspoof2019 LA, ASVspoof2021 LA),
корпуса реальных шумов/ИХ (MUSAN, RIRS_NOISES, ESC-50 → `data/corpora/`),
кэширует веса моделей, строит манифесты и сбалансированные подвыборки (800/класс),
и прогоняет офлайн-smoke моделей. Отдельные части:

```bash
bash scripts/download_corpora.sh data/raw/corpora data/corpora
bash scripts/download_asvspoof2021.sh LA data/raw/asvspoof2021_la
adfd subset --manifest data/in_the_wild/manifest.csv --out data/in_the_wild/manifest_small.csv --n 800
```

### 10.2. Запуск конвейера в фоне (вычислительный узел, офлайн)

```bash
sbatch scripts/slurm_pipeline.sh
```

- Этапы идут строго по порядку; завершённый этап помечается `runs/pipeline/<name>.done` и при
  перезапуске (requeue/обрыв/повторный `sbatch`) пропускается — прогон возобновляется с места остановки.
- Обязательный этап при падении **останавливает** весь конвейер (по требованию: fail-fast).
- Этапы с `optional: true` (SSL-AASIST, XLSR-Mamba, RawNet2, реальные корпуса, телефония, EnCodec) при
  падении лишь **пропускаются** — например, если не собрался fairseq/mamba-ssm или отсутствует корпус.
- Сводка по всем этапам — `runs/pipeline/status.json` (`done/cached/skipped/failed/disabled`).

Полезное:

```bash
adfd pipeline --only sweep_aasist3_asv19 attack_aasist3_asv19   # только эти этапы
adfd pipeline --force                                          # перепрогнать всё, игнорируя .done
sbatch scripts/slurm_attack.sh spectra_aasist3 data/asvspoof2019_la/eval_small.csv aasist
```

### 10.3. Реальные искажения вместо синтетики

`configs/distortions_full.yaml` добавляет к синтетическим условиям парные реальные:
`real_noise` (MUSAN), `real_bird` (ESC-50), `real_rir` (openSLR-28), `telephony_g711a/g711u/g722`,
`telephony_amrnb`, `neural_encodec`. Пути к корпусам — в `options.corpus/rir_dir`.
Прогнать вручную по одной модели:

```bash
adfd sweep --detector aasist3 --manifest data/in_the_wild/manifest_small.csv \
  --distortions configs/distortions_full.yaml --only real_noise real_bird real_rir \
  --run-dir runs/aasist3_real
```

### 10.4. Атаки: бюджет по SNR и перенос

```bash
adfd attack --detector aasist3 --manifest data/asvspoof2019_la/eval_small.csv \
  --budget snr --transfer aasist --run-dir runs/aasist3_attack
```

Уровни атак в `configs/attacks.yaml` — это целевой **SNR возмущения** (40/30/20/10 дБ), общий с шумом;
ε считается на каждый файл от его RMS, в `attack_table.csv` пишется достигнутый SNR (`strength_value`),
а `transfer_aggregate.csv` — EER атакованных примеров на других моделях (перенос).

### 10.5. Кросс-доменный reference (обучение на ASVspoof2019, оценка zero-shot)

`prepare_login.sh` строит два манифеста для reference через `adfd merge`:
`data/reference/native.csv` (train ASVspoof2019 + eval ASVspoof2019) и
`data/reference/crossdomain.csv` (train ASVspoof2019 + eval In-the-Wild).
Так reference обучается на ASVspoof2019, а на ITW работает честно zero-shot — как нейросетевые модели.
Метка `dataset` в артефактах берётся из eval-сплита (не из train-строк).

```bash
adfd merge --train-from data/asvspoof2019_la/train_small.csv \
  --eval-from data/in_the_wild/manifest_small.csv --out data/reference/crossdomain.csv
```

### 10.6. Мониторинг прогресса и отчёт

```bash
# на каком этапе конвейер (обновляется по мере выполнения):
./.venv/bin/adfd progress
watch -n 30 ./.venv/bin/adfd progress          # автообновление раз в 30 с
squeue -u "$USER"                              # статус Slurm-джоба
tail -f runs/slurm-adfd-pipeline-*.log         # живой лог
cat runs/pipeline/status.json                  # done/cached/skipped/failed по этапам

# сводный отчёт + графики (можно вызвать в любой момент, не дожидаясь конца):
./.venv/bin/adfd report
```

`adfd report` собирает из всех прогонов: `clean_summary.csv` (EER/AUC/Cllr/minDCF),
`dataset_shift.csv` (EER по доменам), `degradation_summary.csv` (ΔEER±CI по каждому
модель×искажение×уровень), `worst_degradation.csv`, `attack_summary.csv` (ASR по SNR),
`transfer_summary.csv`, плюс `report.md` и графики (`clean_calibration.png`,
`degradation_<dataset>.png`, `attack_asr.png`) — если установлен matplotlib.

### 10.7. Полная шпаргалка команд (весь цикл)

```bash
# --- один раз: окружение (login-узел) ---
bash scripts/setup_cluster.sh
./.venv/bin/python -m pip install -r requirements-gpu.txt

# --- скачивание + подготовка (login-узел, tmux, есть интернет) ---
bash scripts/prepare_login.sh                  # датасеты, корпуса, веса, манифесты, подвыборки, merge, smoke

# --- запуск всего эксперимента в фоне (вычислительный узел, офлайн) ---
sbatch scripts/slurm_pipeline.sh               # весь конвейер, resume при обрыве

# --- наблюдение ---
./.venv/bin/adfd progress
squeue -u "$USER"
tail -f runs/slurm-adfd-pipeline-*.log

# --- отчёт и графики (в любой момент) ---
./.venv/bin/adfd report

# --- точечные перезапуски ---
./.venv/bin/adfd pipeline --only sweep_aasist3_asv19 attack_aasist3_asv19
sbatch scripts/slurm_attack.sh spectra_aasist3 data/asvspoof2019_la/eval_small.csv aasist
./.venv/bin/adfd pipeline --force              # перепрогнать всё
```
