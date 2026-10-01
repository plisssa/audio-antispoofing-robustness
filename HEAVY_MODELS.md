# Тяжёлые модели: установка и проверка

## Итог (2026-07)
- **RawNet2 — интегрирована** (verify EER 0.046), 5-я модель пула.
- **Nes2Net — не интегрирована**: s3prl-зависимость чинится (omegaconf + предкэш WavLM), но конструктор `WavLM_Nes2Net_X` требует спец-аргументов (`args.agg` и др.), а релизный чекпойнт обучен на CtrSVDD (пение) → off-domain для речевого спуфинга. Стадии отключены (`enabled: false`).
- **SSL-AASIST / XLSR-Mamba — не интегрированы**: fairseq не собирается на cHARISMa/Rocky (C++/CUDA-расширения падают при компиляции), mamba-ssm без wheel под torch2.4/cp312. Ограничение тулчейна кластера.

Финальный пул: reference · AASIST · AASIST3 · Spectra · RawNet2 (5 моделей × 3 домена).



Источники проверены на живость (2026-07). Login-узел = интернет есть, но убивает тяжёлые
процессы (компиляция/большие загрузки — используй `curl -C -` resume). Compute = offline, `sbatch`,
`HF_HUB_OFFLINE=1`. Скачивание — на login, проверка — только в job'е.

Быстрый путь:
```bash
bash scripts/setup_heavy_models.sh          # login: клон + чекпойнты + зависимости
sbatch scripts/slurm_verify_heavy.sh        # compute: verify-model по каждой модели
```
Какие модели покажут вменяемый EER — те включатся в пайплайне (стадии уже есть, optional);
повторный `sbatch scripts/slurm_pipeline.sh` их досчитает.

## RawNet2 — РИСК НИЗКИЙ (ставить первой)
- Репо: `github.com/asvspoof-challenge/2021`, файлы в `LA/Baseline-RawNet2/` (в `DF/` — симлинки, копировать нельзя `cp -r`, только из `LA/` или `cp -rL`).
- Чекпойнт: `https://www.asvspoof.org/asvspoof2021/pre_trained_DF_RawNet2.zip` (~66 МБ, HTTP 200), в архиве ровно `pre_trained_DF_RawNet2.pth`.
- Зависимости: только torch (в модуле). НЕ ставить `numpy==1.17` из их requirements (ломает py3.12).
- Рантайм-загрузок нет — полностью self-contained. Единственный надёжный из четырёх.

## Nes2Net — РИСК НИЗКИЙ
- Репо: `github.com/Liu-Tianchi/Nes2Net` (класс `WavLM_Nes2Net_noRes_w_allT`, совпадает с адаптером).
- Чекпойнт: HF `nielsr/nes2net-checkpoints/…/WavLM_Nes2Net_X_e54_seed42_valid0.033…​.pt` (или gdrive `1EynkhacBVdUvami7pWX8fyUQnyZaRnvV`), переименовать в `WavLM_Nes2Net_X.pth`.
- Зависимость: `s3prl` (torch из модуля, `--no-deps`). Компиляции нет.
- OFFLINE-гоча: фронтенд WavLM-Large тянется в рантайме (~1.2 ГБ) → **предкэшировать на login** (`python -c "import s3prl.hub as hub; hub.wavlm_large()"`), иначе на compute упадёт.
- Оговорка: релизный чекпойнт обучен на CtrSVDD (пение), домен отличается — грузится/работает, но цифры это отражают.

## SSL-AASIST — РИСК СРЕДНИЙ
- Репо: `github.com/TakHemlata/SSL_Anti-spoofing`. Обучена с RawBoost → ценна для гипотезы про аугментации.
- Фронтенд `xlsr2_300m.pt` (публичный fbaipublicfiles) → симлинк в каталог репо (fairseq грузит с диска).
- Чекпойнт `Best_SSL_model_LA.pth`: Google Drive из README (точный ID не подтверждён — скачать вручную или задать `SSL_AASIST_CKPT_ID=<id>`).
- Блокер: `fairseq` под torch2.4/py3.12 — editable из исходников @`a54021305d6b3c` (сборка Python+cython, обычно НЕ убивается login). При конфликте pins: сначала `pip install omegaconf==2.0.6 hydra-core==1.0.7`.

## XLSR-Mamba — РИСК ВЫСОКИЙ
- Репо: `github.com/swagshaw/XLSR-Mamba`.
- Чекпойнт: HF `AustinXiao/XLSR-Mamba-LA` = `model.safetensors` (1.28 ГБ, НЕ .pth) → конвертировать в `xlsr_mamba_LA.pth` (`safetensors.torch.load_file` → `torch.save`). Скрипт делает это на login.
- Блокер: `mamba-ssm==1.1.4` + `causal-conv1d==1.1.3.post1` **не имеют wheel под torch2.4/cp312** → сборка nvcc (login убивает, compute без интернета для build-deps). Апгрейд до mamba-ssm 2.2.x ломает импорт в репо (`Block` переехал). Реалистично: либо готовые wheels с hpc.hse.ru/software, либо собрать на compute-узле через `srun` (если build-deps уже в venv), либо забить — модель опциональна.

## Проверка
`sbatch scripts/slurm_verify_heavy.sh [manifest]` гоняет `verify-model` для rawnet2/ssl_aasist/xlsr_mamba/nes2net на compute-узле. EER ~0.0–0.5 = загрузилась; EER>0.5 = проверить spoof_index; ошибка = чекпойнт/зависимость не на месте.
