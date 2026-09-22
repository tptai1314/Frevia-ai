# Research — FREVIA (NCKH)

Đọc/ghi dataset và các thí nghiệm của đề tài NCKH. Dùng chung với repo `Frevia-ai`.

```bash
uvicorn app.main:app --reload
```

## Cấu trúc repo

```
frevia-ai/
├── app/                     # Demo service (FastAPI + Qwen extraction)
│   ├── api/v1/              # endpoints
│   ├── core/
│   ├── schemas/             # CV / job schemas (Pydantic)
│   └── services/            # qwen_service (Ollama, qwen3:4b), cv_parser
├── research/                # Thí nghiệm NCKH
│   └── config.py            # path dữ liệu (data/raw/...)
├── data/                    # Dữ liệu đề tài
│   ├── raw/
│   │   ├── jobs/
│   │   │   ├── Cleaned_DS_Jobs.csv        # 660 JD (Data Science) — paper 2026
│   │   │   ├── Uncleaned_DS_jobs.csv      # 672 JD (raw variant)
│   │   │   └── glassdoor_jobs.csv         # 1500 JD (extra)
│   │   └── resumes/
│   │       └── UpdatedResumeDataSet.csv   # 962 CV, 25 domains — paper 2026
│   └── processed/           # (trống) dữ liệu đã tiền xử lý
├── docs/
│   ├── FullText.pdf         # Paper baseline 2026
│   └── FullText.txt
├── DE_CUONG.md              # Đề cương NCKH
└── requirements.txt
```

## Ghi chú dữ liệu (khớp paper 2026)

- **JD**: Glassdoor "Data Science" (Kaggle) — 660 bài.
- **CV**: Kaggle "Resume Dataset" — 962 hồ sơ, 25 domain.
- **Protocol đánh giá**: P@10 / R@10 trên domain Data Science (40 CV) và Hadoop (42 CV).

## Run locally

```bash
uvicorn app.main:app --reload
```

The API health endpoint is available at `GET /health`.