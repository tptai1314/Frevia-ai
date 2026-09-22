from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
JOBS_DIR = RAW_DIR / 'jobs'
RESUMES_DIR = RAW_DIR / 'resumes'

# Datasets (paper 2026)
CLEANED_JOBS_CSV = JOBS_DIR / 'Cleaned_DS_Jobs.csv'  # 660 JD (Data Science)
UNCLEANED_JOBS_CSV = JOBS_DIR / 'Uncleaned_DS_jobs.csv'  # raw JD variant
GLASSDOOR_JOBS_CSV = JOBS_DIR / 'glassdoor_jobs.csv'  # 1500 JD (extra)
RESUMES_CSV = RESUMES_DIR / 'UpdatedResumeDataSet.csv'  # 962 CV, 25 domains

# Paper 2026 evaluation protocol
TARGET_DOMAINS = ['Data Science', 'Hadoop']
TOP_K = 10