[![English](https://img.shields.io/badge/lang-English-blue.svg)](README.md) [![한국어](https://img.shields.io/badge/lang-한국어-red.svg)](README.ko.md)

# 약물감시 ADE 분류기

이진(binary) 약물 이상 반응(ADE) 문장 분류 과제에서 바이오메디컬 도메인
사전학습의 효과를 페어드 부트스트랩(paired bootstrap) 기반으로 검증한
ablation 연구입니다.

## 프로젝트 개요

본 프로젝트에서는 세 개의 transformer 인코더 — `bert-base-uncased`,
BioBERT, PubMedBERT — 를 ADE Corpus v2의 분류 split에 대해 fine-tuning하고,
다음 두 가지 좁은 범위의 질문을 검증합니다. 첫째, 바이오메디컬 도메인
사전학습이 일반 도메인 베이스라인 대비 다운스트림 성능을 측정 가능한
수준으로 향상시키는가? 둘째, 처음부터(from-scratch) 수행한 바이오메디컬
사전학습(PubMedBERT)이 일반 도메인 가중치에서 이어서 사전학습한
(continued pretraining) BioBERT보다 더 나은 성능을 보이는가? 본 연구는
사전학습 레시피에 대한 일반화된 주장이 아니라, 소규모 코퍼스에서의
정직한 유의성 검정 방법론을 보여 주는 예시로 이해되기를 권장합니다.

## 주요 결과

argmax 임계값 0.5 기준으로, PubMedBERT는 base BERT 대비 macro-F1
(gap -0.012, 95% CI [-0.024, -0.003])과 PR-AUC (-0.018, CI [-0.031, -0.007])
모두에서 우위를 보입니다. BioBERT는 base BERT 대비 확률 순위(PR-AUC gap
-0.012, CI [-0.020, -0.005])가 개선되지만, macro-F1 개선치는 0과 구분되지
않습니다. 효과 크기는 크지 않으며 — macro-F1 기준 약 1%p 수준 — 평가는
단일 코퍼스, 단일 test split에 한정됩니다. 전체 기술 보고서는
[`docs/paper.md`](docs/paper.md)를, 원본 수치는
[`results/pairwise_gaps.csv`](results/pairwise_gaps.csv)를 참고하시기
바랍니다.

## 빠른 시작

결과를 재현하는 가장 빠른 방법은 커밋된 run 출력물을 대상으로 분석
노트북을 실행하는 것입니다. GPU가 필요하지 않으며, 전체 실행에 약
10초 정도 소요됩니다.

```bash
git clone https://github.com/tjohns94/pharmacovigilance-ade-classifier.git
cd pharmacovigilance-ade-classifier
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements-colab.txt
pip install -e .
jupyter lab notebooks/analysis.ipynb
```

Python 3.10 이상이 필요합니다. `paper.md`의 모든 figure와 표는 재학습
없이 `results/` 내 커밋된 artifact로부터 재생성됩니다. 30-run ablation
(모델 3종 × seed 10개)을 Colab에서 재학습하려면
[`docs/reproduce.md`](docs/reproduce.md)를 참고하시기 바랍니다.

## 저장소 구조

```
configs/ablation.yaml     실험의 단일 설정 파일(single source of truth)
data/splits/              커밋된 train/val/test 인덱스 + data card
notebooks/
  colab_train.ipynb       GPU 학습 run (Colab)
  analysis.ipynb          로컬 분석 + figure, 수 초 내 실행
scripts/                  헤드리스(bootstrap) 재생성 스크립트
src/pv_ade/               패키지: data, model, train, evaluate, analysis
results/                  seed별 metric, raw 예측값, pairwise gap
figures/                  analysis.ipynb에서 렌더링된 PNG
tests/                    pytest smoke test
docs/                     프로젝트 문서
```

## 추가 문서 안내

| 목적 | 참고 문서 |
|---|---|
| 빠른 개요 파악 | [`docs/overview.md`](docs/overview.md) |
| 과학적 결과의 이해 | [`docs/paper.md`](docs/paper.md) |
| 분석을 직접 실행 | [`docs/reproduce.md`](docs/reproduce.md) |
| 데이터 및 중복 제거 정책 이해 | [`docs/dataset.md`](docs/dataset.md) |
| Colab 학습과 로컬 분석의 연결 구조 확인 | [`docs/pipeline.md`](docs/pipeline.md) |
| `src/` 내 특정 함수 검색 | [`docs/code-map.md`](docs/code-map.md) |
| figure가 포함된 분석 열람 | [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb) |

## 저자

Tyson Johnson, 조지 메이슨 대학교 전산 및 데이터 과학과
(Department of Computational and Data Sciences, George Mason University).

## 라이선스

MIT 라이선스. [`LICENSE`](LICENSE) 파일을 참고하시기 바랍니다.

## 감사의 글

본 프로젝트의 기획, 구현, 초안 작성 과정에서 Anthropic's Claude의
도움을 받았습니다. 모든 분석, 해석, 결론은 저자 본인의 것입니다.
