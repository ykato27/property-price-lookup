# 精度実験 結果（粒度 × 手法）

「どの粒度・どの方法が精度を上げるか」を、固定の評価プロトコルで検証した記録。
実験設計とリーク防止ルールは `src/market/experiments.py` の冒頭に定義してある。

- 固定プロトコル: 学習 ≤2024Q4 / 検証 2025Q1–2026Q1、対象区の行で評価。
  データが満たせない場合は最後の5四半期を検証にするフォールバック（結果に明記）。
- 指標: MAPE（中央値・取引総額ベース）、50%区間被覆率、区間幅（q75/q25平均）。
- ターゲットエンコーディング・時点補正・調整・ブレンド重み・区間較正は、
  すべて学習期間のデータだけで決める（検証期間の情報は使わない）。

> **注意**: 本番データ（23区・複数年）での本実験はデータ取り込み後に再実行が必要。
> それ以前の実行は、フィクスチャや小規模な実データによる予備結果である。

## 実行: 2026-10-23T11:53:01

- データセット: フィクスチャ1500件 + DB実データ(trade)753件（全2253件）
- 対象区: 港区
- プロトコル: fixed（学習 2021Q1–2024Q4 / 検証 2025Q1–2026Q1）

| 構成 | A範囲 | B特徴量 | C手法 | 調整 | D目的 | プール | 学習件数 | 検証件数 | MAPE(%) | 被覆率(%) | 区間幅 | 較正後被覆率(%) | 較正後区間幅 | 状態 | 備考 |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| A:all/B:base/C:gbm/D:unit | all | base | gbm | - | unit | - | 1224 | 767 | 46.00 | 4.69 | 1.177 | 7.82 | 1.288 | ok |  |
| A:ward/B:base/C:gbm/D:unit | ward | base | gbm | - | unit | - | 47 | 767 | - | - | - | - | - | skipped | 学習プールが100件未満のためGBMをスキップ |
| A:all_3y/B:base/C:gbm/D:unit | all_3y | base | gbm | - | unit | - | 878 | 767 | 46.42 | 6.52 | 1.220 | 9.78 | 1.340 | ok |  |
| A:all_2y/B:base/C:gbm/D:unit | all_2y | base | gbm | - | unit | - | 601 | 767 | 48.09 | 4.43 | 1.220 | 6.39 | 1.315 | ok |  |
| A:all/B:te/C:gbm/D:unit | all | te | gbm | - | unit | - | 1224 | 767 | 66.29 | 0.78 | 1.162 | 1.17 | 1.260 | ok |  |
| A:all/B:te_extra/C:gbm/D:unit | all | te_extra | gbm | - | unit | - | 1224 | 767 | 66.49 | 0.39 | 1.170 | 1.04 | 1.266 | ok | coverage_ratio は学習側の有効行が少なく特徴量から除外 / floor_area_ratio は学習側の有効行が少なく特徴量から除外 |
| A:all/B:base/C:gbm+tuned/D:unit | all | base | gbm | 有 | unit | - | 1224 | 767 | 45.86 | 4.56 | 1.200 | 7.82 | 1.303 | ok | 調整時OOF MAPE=13.83% |
| A:all/B:base/C:layer1[city]/D:unit | all | base | layer1 | - | unit | city | 1224 | 767 | 42.73 | 13.82 | 1.353 | - | - | ok | フル帯一致率=0.07 |
| A:all/B:base/C:layer1[district]/D:unit | all | base | layer1 | - | unit | district | 1224 | 767 | 43.20 | 22.16 | 1.706 | - | - | ok | フル帯一致率=0.00 |
| A:all/B:base/C:layer1[station]/D:unit | all | base | layer1 | - | unit | station | 1224 | 767 | 43.68 | 22.56 | 1.750 | - | - | ok | フル帯一致率=0.00 |
| A:all/B:base/C:gbm/D:total | all | base | gbm | - | total | - | 1224 | 767 | 49.29 | 5.08 | 1.358 | 7.95 | 1.483 | ok |  |
| combined_blend | all | base | blend | 有 | unit | city | 1224 | 767 | 45.86 | 4.56 | 1.200 | 7.82 | 1.303 | ok | 調整時OOF MAPE=13.83% |
## 実行: 2026-10-23T11:53:16

- データセット: DB実データ(trade)753件（全753件）
- 対象区: 港区
- プロトコル: fallback_last4q（学習 2025Q1–2025Q1 / 検証 2025Q2–2026Q1）
- 分割に関する注意: 固定プロトコル（学習≤2024Q4／検証2025Q1–2026Q1）を満たす行数が不足したため、最後の4四半期を検証期間とするフォールバックを使用。

| 構成 | A範囲 | B特徴量 | C手法 | 調整 | D目的 | プール | 学習件数 | 検証件数 | MAPE(%) | 被覆率(%) | 区間幅 | 較正後被覆率(%) | 較正後区間幅 | 状態 | 備考 |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| A:all/B:base/C:gbm/D:unit | all | base | gbm | - | unit | - | 175 | 578 | 25.90 | 42.04 | 1.419 | - | - | ok |  |
| A:ward/B:base/C:gbm/D:unit | ward | base | gbm | - | unit | - | 175 | 578 | 25.90 | 42.04 | 1.419 | - | - | ok |  |
| A:all_3y/B:base/C:gbm/D:unit | all_3y | base | gbm | - | unit | - | 175 | 578 | 25.90 | 42.04 | 1.419 | - | - | ok |  |
| A:all_2y/B:base/C:gbm/D:unit | all_2y | base | gbm | - | unit | - | 175 | 578 | 25.90 | 42.04 | 1.419 | - | - | ok |  |
| A:all/B:te/C:gbm/D:unit | all | te | gbm | - | unit | - | 175 | 578 | 26.91 | 37.89 | 1.356 | - | - | ok |  |
| A:all/B:te_extra/C:gbm/D:unit | all | te_extra | gbm | - | unit | - | 175 | 578 | 26.64 | 38.93 | 1.363 | - | - | ok | coverage_ratio は学習側の有効行が少なく特徴量から除外 / floor_area_ratio は学習側の有効行が少なく特徴量から除外 |
| A:all/B:base/C:gbm+tuned/D:unit | all | base | gbm | 有 | unit | - | 175 | 578 | 25.90 | 42.04 | 1.419 | - | - | ok |  |
| A:all/B:base/C:layer1[city]/D:unit | all | base | layer1 | - | unit | city | 175 | 578 | 30.91 | 42.73 | 1.523 | - | - | ok | フル帯一致率=0.72 |
| A:all/B:base/C:layer1[district]/D:unit | all | base | layer1 | - | unit | district | 175 | 578 | 42.98 | 35.64 | 1.688 | - | - | ok | フル帯一致率=0.04 |
| A:all/B:base/C:layer1[station]/D:unit | all | base | layer1 | - | unit | station | 175 | 578 | 42.48 | 41.87 | 1.855 | - | - | ok | フル帯一致率=0.00 |
| A:all/B:base/C:gbm/D:total | all | base | gbm | - | total | - | 175 | 578 | 28.32 | 42.04 | 1.493 | - | - | ok |  |
| combined_blend | all | base | blend | 有 | unit | city | 175 | 578 | 26.68 | 46.02 | 1.465 | - | - | ok |  |
## 実行: 2026-10-23T12:24:57

- データセット: DB実データ(trade)82575件（全82575件）
- 対象区: 港区
- プロトコル: fixed（学習 2021Q1–2024Q4 / 検証 2025Q1–2026Q1）

| 構成 | A範囲 | B特徴量 | C手法 | 調整 | D目的 | プール | 学習件数 | 検証件数 | MAPE(%) | 被覆率(%) | 区間幅 | 較正後被覆率(%) | 較正後区間幅 | 状態 | 備考 |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| A:all/B:base/C:gbm/D:unit | all | base | gbm | - | unit | - | 59026 | 753 | 26.18 | 33.47 | 1.397 | 39.31 | 1.447 | ok |  |
| A:ward/B:base/C:gbm/D:unit | ward | base | gbm | - | unit | - | 3303 | 753 | 23.66 | 33.33 | 1.289 | 50.20 | 1.437 | ok |  |
| A:all_3y/B:base/C:gbm/D:unit | all_3y | base | gbm | - | unit | - | 43549 | 753 | 26.54 | 32.27 | 1.387 | 35.59 | 1.433 | ok |  |
| A:all_2y/B:base/C:gbm/D:unit | all_2y | base | gbm | - | unit | - | 28806 | 753 | 25.52 | 33.60 | 1.389 | 39.04 | 1.444 | ok |  |
| A:all/B:te/C:gbm/D:unit | all | te | gbm | - | unit | - | 59026 | 753 | 24.54 | 37.32 | 1.345 | 42.63 | 1.399 | ok |  |
| A:all/B:te_extra/C:gbm/D:unit | all | te_extra | gbm | - | unit | - | 59026 | 753 | 24.38 | 37.32 | 1.342 | 40.64 | 1.397 | ok |  |
| A:all/B:base/C:gbm+tuned/D:unit | all | base | gbm | 有 | unit | - | 59026 | 753 | 24.79 | 31.47 | 1.314 | 39.18 | 1.383 | ok | 調整時OOF MAPE=17.64% |
| A:all/B:base/C:layer1[city]/D:unit | all | base | layer1 | - | unit | city | 59026 | 753 | 30.88 | 31.08 | 1.408 | - | - | ok | フル帯一致率=0.97 |
| A:all/B:base/C:layer1[district]/D:unit | all | base | layer1 | - | unit | district | 59026 | 753 | 31.38 | 24.17 | 1.364 | - | - | ok | フル帯一致率=0.73 |
| A:all/B:base/C:layer1[station]/D:unit | all | base | layer1 | - | unit | station | 59026 | 753 | 31.06 | 24.44 | 1.337 | - | - | ok | フル帯一致率=0.71 |
| A:all/B:base/C:gbm/D:total | all | base | gbm | - | total | - | 59026 | 753 | 27.06 | 32.40 | 1.435 | 36.25 | 1.487 | ok |  |
| combined_blend | ward | base | blend | - | unit | city | 3303 | 753 | 23.66 | 33.33 | 1.289 | 50.20 | 1.437 | ok |  |

## 実行: 2026-10-23T12:43:46

- データセット: DB実データ(trade)82575件（全82575件）
- 対象区: 世田谷区
- プロトコル: fixed（学習 2021Q1–2024Q4 / 検証 2025Q1–2026Q1）

| 構成 | A範囲 | B特徴量 | C手法 | 調整 | D目的 | プール | 学習件数 | 検証件数 | MAPE(%) | 被覆率(%) | 区間幅 | 較正後被覆率(%) | 較正後区間幅 | 状態 | 備考 |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| A:all/B:base/C:gbm/D:unit | all | base | gbm | - | unit | - | 59026 | 824 | 20.74 | 43.08 | 1.328 | 48.91 | 1.376 | ok |  |
| A:ward/B:base/C:gbm/D:unit | ward | base | gbm | - | unit | - | 3594 | 824 | 18.92 | 36.04 | 1.238 | 49.39 | 1.367 | ok |  |
| A:all_3y/B:base/C:gbm/D:unit | all_3y | base | gbm | - | unit | - | 43549 | 824 | 20.47 | 41.02 | 1.312 | 46.84 | 1.355 | ok |  |
| A:all_2y/B:base/C:gbm/D:unit | all_2y | base | gbm | - | unit | - | 28806 | 824 | 20.60 | 38.11 | 1.295 | 45.15 | 1.347 | ok |  |
| A:all/B:te/C:gbm/D:unit | all | te | gbm | - | unit | - | 59026 | 824 | 18.99 | 40.90 | 1.299 | 46.97 | 1.352 | ok |  |
| A:all/B:te_extra/C:gbm/D:unit | all | te_extra | gbm | - | unit | - | 59026 | 824 | 17.01 | 39.20 | 1.252 | 46.84 | 1.304 | ok |  |
| A:all/B:base/C:gbm+tuned/D:unit | all | base | gbm | 有 | unit | - | 59026 | 824 | 19.52 | 37.74 | 1.262 | 44.42 | 1.328 | ok | 調整時OOF MAPE=17.64% |
| A:all/B:base/C:layer1[city]/D:unit | all | base | layer1 | - | unit | city | 59026 | 824 | 25.33 | 39.32 | 1.383 | - | - | ok | フル帯一致率=0.98 |
| A:all/B:base/C:layer1[district]/D:unit | all | base | layer1 | - | unit | district | 59026 | 824 | 26.14 | 34.47 | 1.331 | - | - | ok | フル帯一致率=0.48 |
| A:all/B:base/C:layer1[station]/D:unit | all | base | layer1 | - | unit | station | 59026 | 824 | 26.94 | 37.62 | 1.335 | - | - | ok | フル帯一致率=0.62 |
| A:all/B:base/C:gbm/D:total | all | base | gbm | - | total | - | 59026 | 824 | 20.59 | 43.08 | 1.347 | 50.49 | 1.396 | ok |  |
| combined_blend | all | te_extra | blend | - | unit | city | 59026 | 824 | 17.01 | 39.20 | 1.252 | 46.84 | 1.304 | ok |  |

## 実行: 2026-10-23T12:54:24

- データセット: DB実データ(trade)82575件（全82575件）
- 対象区: 港区
- プロトコル: fixed（学習 2021Q1–2024Q4 / 検証 2025Q1–2026Q1）

| 構成 | A範囲 | B特徴量 | C手法 | 調整 | D目的 | プール | 学習件数 | 検証件数 | MAPE(%) | 被覆率(%) | 区間幅 | 較正後被覆率(%) | 較正後区間幅 | 状態 | 備考 |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| supp:all/te_extra/gbm+tuned | all | te_extra | gbm | 有 | unit | - | 59026 | 753 | 23.21 | 33.33 | 1.285 | 41.43 | 1.359 | ok | 調整時OOF MAPE=15.44% |
| supp:all/te_extra/blend[city] | all | te_extra | blend | - | unit | city | 59026 | 753 | 24.38 | 37.32 | 1.342 | 40.64 | 1.397 | ok |  |
| supp:ward/te_extra/gbm | ward | te_extra | gbm | - | unit | - | 3303 | 753 | 22.61 | 31.21 | 1.263 | 48.87 | 1.424 | ok |  |
| supp:ward/te/gbm | ward | te | gbm | - | unit | - | 3303 | 753 | 23.29 | 31.08 | 1.264 | 48.87 | 1.430 | ok |  |

## 実行: 2026-10-23T13:06:07

- データセット: DB実データ(trade)82575件（全82575件）
- 対象区: 世田谷区
- プロトコル: fixed（学習 2021Q1–2024Q4 / 検証 2025Q1–2026Q1）

| 構成 | A範囲 | B特徴量 | C手法 | 調整 | D目的 | プール | 学習件数 | 検証件数 | MAPE(%) | 被覆率(%) | 区間幅 | 較正後被覆率(%) | 較正後区間幅 | 状態 | 備考 |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| supp:all/te_extra/gbm+tuned | all | te_extra | gbm | 有 | unit | - | 59026 | 824 | 16.23 | 34.95 | 1.204 | 44.90 | 1.274 | ok | 調整時OOF MAPE=15.44% |
| supp:all/te_extra/blend[city] | all | te_extra | blend | - | unit | city | 59026 | 824 | 17.01 | 39.20 | 1.252 | 46.84 | 1.304 | ok |  |
| supp:ward/te_extra/gbm | ward | te_extra | gbm | - | unit | - | 3594 | 824 | 16.63 | 31.92 | 1.185 | 49.76 | 1.308 | ok |  |
| supp:ward/te/gbm | ward | te | gbm | - | unit | - | 3594 | 824 | 18.46 | 31.92 | 1.214 | 50.61 | 1.342 | ok |  |

## 実行: 2026-10-23T14:11:42

- データセット: DB実データ(trade・全国)269202件（全269202件）
- 対象区: 大阪市中央区
- プロトコル: fixed（学習 2021Q1–2024Q4 / 検証 2025Q1–2026Q1）

| 構成 | A範囲 | B特徴量 | C手法 | 調整 | D目的 | プール | 学習件数 | 検証件数 | MAPE(%) | 被覆率(%) | 区間幅 | 較正後被覆率(%) | 較正後区間幅 | 状態 | 備考 |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| old_all_base | all | base | gbm | - | unit | - | 211324 | 770 | 20.42 | 31.43 | 1.296 | 35.58 | 1.323 | ok |  |
| adopted_all_te_extra_tuned | all | te_extra | gbm | 有 | unit | - | 211324 | 770 | 17.15 | 34.42 | 1.229 | 43.38 | 1.286 | ok | 調整時OOF MAPE=28.59% |
| adopted_ward_te_extra_tuned | ward | te_extra | gbm | 有 | unit | - | 2457 | 770 | 17.30 | 24.68 | 1.172 | 42.60 | 1.289 | ok | 調整時OOF MAPE=16.35% |

## 実行: 2026-10-23T14:20:21

- データセット: DB実データ(trade・全国)269202件（全269202件）
- 対象区: 福岡市中央区
- プロトコル: fixed（学習 2021Q1–2024Q4 / 検証 2025Q1–2026Q1）

| 構成 | A範囲 | B特徴量 | C手法 | 調整 | D目的 | プール | 学習件数 | 検証件数 | MAPE(%) | 被覆率(%) | 区間幅 | 較正後被覆率(%) | 較正後区間幅 | 状態 | 備考 |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| old_all_base | all | base | gbm | - | unit | - | 211324 | 654 | 26.38 | 32.26 | 1.464 | 36.09 | 1.494 | ok |  |
| adopted_all_te_extra_tuned | all | te_extra | gbm | 有 | unit | - | 211324 | 654 | 23.67 | 36.70 | 1.353 | 45.57 | 1.416 | ok | 調整時OOF MAPE=28.59% |
| adopted_ward_te_extra_tuned | ward | te_extra | gbm | 有 | unit | - | 2445 | 654 | 22.84 | 33.94 | 1.268 | 48.17 | 1.435 | ok | 調整時OOF MAPE=18.99% |

## 実行: 2026-10-23T14:29:23

- データセット: DB実データ(trade・全国)269202件（全269202件）
- 対象区: 横浜市港北区
- プロトコル: fixed（学習 2021Q1–2024Q4 / 検証 2025Q1–2026Q1）

| 構成 | A範囲 | B特徴量 | C手法 | 調整 | D目的 | プール | 学習件数 | 検証件数 | MAPE(%) | 被覆率(%) | 区間幅 | 較正後被覆率(%) | 較正後区間幅 | 状態 | 備考 |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| old_all_base | all | base | gbm | - | unit | - | 211324 | 475 | 24.02 | 44.00 | 1.539 | 46.74 | 1.572 | ok |  |
| adopted_all_te_extra_tuned | all | te_extra | gbm | 有 | unit | - | 211324 | 475 | 17.28 | 41.26 | 1.334 | 48.21 | 1.396 | ok | 調整時OOF MAPE=28.59% |
| adopted_ward_te_extra_tuned | ward | te_extra | gbm | 有 | unit | - | 1752 | 475 | 16.80 | 27.37 | 1.197 | 50.74 | 1.372 | ok | 調整時OOF MAPE=19.77% |

## 実行: 2026-10-23T14:45:33

- データセット: DB実データ(trade・全国)269202件（全269202件）
- 対象区: 名古屋市中区
- プロトコル: fixed（学習 2021Q1–2024Q4 / 検証 2025Q1–2026Q1）

| 構成 | A範囲 | B特徴量 | C手法 | 調整 | D目的 | プール | 学習件数 | 検証件数 | MAPE(%) | 被覆率(%) | 区間幅 | 較正後被覆率(%) | 較正後区間幅 | 状態 | 備考 |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| old_all_base | all | base | gbm | - | unit | - | 211324 | 399 | 21.28 | 42.86 | 1.351 | 45.61 | 1.379 | ok |  |
| adopted_all_te_extra_tuned | all | te_extra | gbm | 有 | unit | - | 211324 | 399 | 20.15 | 43.61 | 1.278 | 52.13 | 1.338 | ok | 調整時OOF MAPE=28.59% |
| adopted_ward_te_extra_tuned | ward | te_extra | gbm | 有 | unit | - | 1267 | 399 | 19.02 | 31.58 | 1.189 | 50.38 | 1.315 | ok | 調整時OOF MAPE=19.74% |

