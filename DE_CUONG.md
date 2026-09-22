# ĐỀ CƯƠNG NCKH — FREVIA

**Chủ đề:** Cải thiện chất lượng ranking CV–JD bằng structured multi-view representation, cross-view interaction và adaptive fusion
(so với baseline semantic similarity-based job matching).

**Vai trò tài liệu:**
| Nguồn | Vai trò |
|-------|---------|
| Paper 2026 (Ajjam & Al-Raweshidy, *Information Sciences 724*) | Baseline — reproduce + so sánh |
| Matrix (multi-view/cross-view interaction) | Related work — tham khảo cách xử lý multi-view |
| FREVIA (nhóm đề xuất) | Phần cải tiến (ĐÓNG GÓP) |

```
NCKH
│
├── Bài toán
├── Baseline
├── Phương pháp đề xuất ← ĐÓNG GÓP
├── Dataset + Ground Truth
├── Experiment
└── Kết quả + phân tích
```

---

## 0. Tóm tắt đề tài (Abstract)

Các hệ thống đối sánh CV–JD (CV–Job Description matching) dựa trên keyword matching chỉ nhận diện được sự trùng khớp chuỗi ký tự chính xác, dẫn đến bỏ sót các kỹ năng tương đương (transferable skills) và từ đồng nghĩa. Các phương pháp semantic similarity gần đây (ví dụ TF‑IDF/embedding kết hợp cosine similarity) đã cải thiện đáng kể so với keyword matching, nhưng vẫn biểu diễn toàn bộ CV/JD dưới dạng một vector văn bản duy nhất (whole-document embedding), qua đó bỏ qua **cấu trúc ngữ nghĩa nội tại** của văn bản (Title, Skills, Experience, Projects) và **mối quan hệ giữa các khía cạnh** này.

Đề tài đề xuất **FREVIA** — một framework đối sánh CV–JD dựa trên ba thành phần cải tiến:

1. **Structured multi-view representation**: sử dụng LLM (Qwen) để trích xuất CV và JD thành các semantic view có cấu trúc, thay vì một vector văn bản đơn nhất.
2. **Cross-view interaction**: mở rộng đối sánh không chỉ trong cùng loại view (Skill↔Skill) mà còn giữa các view khác loại có liên quan ngữ nghĩa (Skill↔Experience, Experience↔Projects, ...).
3. **Adaptive fusion**: tổng hợp các tín hiệu đối sánh bằng trọng số học được (learned weighting) thay vì trọng số tĩnh/đồng đều, nhằm thích nghi theo đặc thù từng JD.

Đề tài được đánh giá thực nghiệm trên dữ liệu real-world (domain Data Science, Hadoop) với các baseline: keyword matching, TF‑IDF + cosine similarity, và SBERT, sử dụng các metric ranking chuẩn (Precision@10, Recall@10).

---

## 1. Bài toán

### 1.1 Bối cảnh

* Khối lượng hồ sơ ứng tuyển trực tuyến tăng nhanh, gây ra **screening bottleneck** cho nhà tuyển dụng và hệ thống ATS (Applicant Tracking System).
* Keyword-based matching chỉ nhận diện đối khớp chuỗi chính xác, không xử lý được từ đồng nghĩa và kỹ năng tương đương (transferable skills).
* Semantic similarity (TF‑IDF/embedding + cosine similarity) là baseline cải tiến, nhưng có hạn chế về mặt biểu diễn cấu trúc — được phân tích cụ thể ở mục 1.2.

### 1.2 Khoảng trống nghiên cứu (Research Gap) và động lực

| Hạn chế của baseline | Hệ quả |
|---|---|
| Whole-document embedding: gộp toàn bộ CV/JD thành một vector | Mất cấu trúc ngữ nghĩa (title/skills/experience/projects); tín hiệu quan trọng bị pha loãng trong văn bản dài |
| Đối sánh đơn chiều, không tách theo khía cạnh | Không phân biệt "yêu cầu kỹ năng" và "kinh nghiệm thực tế"; không phản ánh trọng số quan trọng khác nhau giữa các khía cạnh theo từng JD |
| Không có tương tác chéo giữa các view | Bỏ sót tín hiệu liên khía cạnh có giá trị (VD: "5 năm kinh nghiệm" + "dự án AI" liên quan yêu cầu "AI Engineer" nhưng không được liên kết) |
| Fusion tĩnh (trọng số cố định/trung bình cộng) | Điểm cuối không thích nghi theo đặc thù JD, dẫn đến ranking kém tối ưu trên các domain khác nhau |

Khoảng trống mà FREVIA hướng đến: **chuyển từ biểu diễn phẳng (flat) sang biểu diễn có cấu trúc, có tương tác chéo, và có cơ chế tổng hợp thích nghi.**

### 1.3 Câu hỏi nghiên cứu (Research Questions)

* **RQ1**: Structured multi-view representation (trích xuất bằng LLM Qwen) có cải thiện chất lượng ranking CV–JD so với whole-document embedding (TF‑IDF) không?
* **RQ2**: Cross-view interaction có bổ sung tín hiệu đối sánh có ý nghĩa thống kê so với chỉ đối sánh view cùng loại (view-to-view thuần túy) không?
* **RQ3**: Adaptive fusion (trọng số học được) có vượt trội so với fusion tĩnh (trọng số cố định hoặc trung bình cộng) không?
* **RQ4**: FREVIA có cải thiện Precision@10 và Recall@10 trên dữ liệu thực tế (domain Data Science, Hadoop) so với các baseline (keyword matching, TF‑IDF + cosine, SBERT) không?

### 1.4 Giả thuyết nghiên cứu (Hypotheses)

* **H1**: FREVIA (full) đạt Precision@10/Recall@10 cao hơn baseline TF‑IDF với khác biệt có ý nghĩa thống kê (p < 0.05, Wilcoxon signed-rank test).
* **H2**: Biến thể có cross-view interaction vượt trội biến thể chỉ same-view matching (ablation RQ2).
* **H3**: Adaptive fusion vượt trội fusion tĩnh (weight cố định / trung bình cộng) (ablation RQ3).
* **H4**: FREVIA vượt baseline SBERT về Precision@10/Recall@10 trên cùng domain (RQ4).

Các giả thuyết này gắn trực tiếp RQ với tiêu chí kiểm định rõ ràng ở phần Experiment, tránh việc chỉ so sánh số liệu tuyệt đối.

### 1.5 Mục tiêu và phạm vi nghiên cứu

**Mục tiêu:**
* Xây dựng pipeline hoàn chỉnh: trích xuất multi-view (LLM) → đối sánh cross-view → adaptive fusion → ranking.
* Thiết lập benchmark thực nghiệm so sánh với các baseline đã công bố (paper 2026) và baseline cổ điển (keyword, TF-IDF, SBERT).

**Phạm vi:**
* Dữ liệu: văn bản tiếng Anh, giới hạn trong các domain có sẵn trong bộ dữ liệu tham chiếu (Data Science, Hadoop, ...).
* Không xử lý bài toán **cold-start domain** (domain hoàn toàn mới không có dữ liệu huấn luyện/tham chiếu) — đây là **giới hạn (limitation)** được nêu rõ, định hướng phần "Hướng phát triển" (future work).
* Không điều chỉnh (fine-tune) LLM Qwen — sử dụng ở chế độ trích xuất (extraction/prompting); việc học tham số chỉ nằm ở tầng fusion.

### 1.6 Đóng góp chính (Contributions)

1. Đề xuất framework FREVIA kết hợp structured multi-view representation, cross-view interaction, và adaptive fusion cho bài toán CV–JD matching.
2. Cung cấp **ablation study** đánh giá đóng góp riêng lẻ của từng thành phần (multi-view / cross-view / adaptive fusion) thông qua so sánh với các biến thể rút gọn (variant ablation), trả lời trực tiếp RQ1–RQ3.
3. Benchmark FREVIA với các baseline chuẩn trên dữ liệu real-world, sử dụng Precision@10/Recall@10, có kiểm định thống kê để bảo đảm tính tin cậy của kết luận.

---

## 2. Baseline

### 2.1 Semantic similarity-based job matching (paper 2026 — reproduce)

- Preprocessing (tokenize, stopword, stemming, lemmatize) → TF‑IDF (unigram, L2-norm) → cosine similarity → greedy one-to-one matching.
- Ưu điểm: interpretable, lightweight, không fine-tune, tránh bias ngầm của pre-trained LLM.
- Nhược điểm (cơ sở đề xuất FREVIA): whole-document representation, không tận dụng cấu trúc CV, fusion tĩnh, nhạy với văn bản thưa/thiếu cấu trúc.

### 2.2 Các baseline đối chứng trong thí nghiệm

| Baseline | Mô tả |
|---|---|
| Keyword-based | Overlap ratio sau preprocessing (không TF-IDF, không semantic) |
| TF-IDF + cosine | Pipeline paper 2026: whole-document TF-IDF → cosine → greedy matching |
| SBERT (`all-MiniLM-L6-v2`) | Sentence embedding 384 chiều; baseline mạnh hơn keyword, có ngữ cảnh |

### 2.3 Tiêu chí reproduce

Code chạy lại được trên đúng dataset & bộ metric với FREVIA (cùng bộ chia dữ liệu, cùng khoá seed, cùng thứ tự đánh giá).

---

## 3. Related Work

### 3.1 Semantic similarity-based job matching
- Paper 2026 đại diện cho hướng TF-IDF + cosine; nhấn mạnh transparency, chi phí thấp, thích ứng domain-vocabulary, và cảnh báo bias/interpretability của LLM.
- Định vị: đây là **baseline** của đề tài, không phải hướng FREVIA theo đuổi.

### 3.2 Multi-aspect / multi-view job matching
- Các hướng neural field-level: InEXIT (nội bộ field interaction + cross-document interaction), DPGNN (dual-perspective graph), APJFNN/PJFNN, Bian et al. (multi-view co-teaching trên sparse interaction data).
- *Matrix (placeholder — bổ sung tên đầy đủ/link):* cách tổ chức multi-view và cross-view interaction giữa các field resume–job mà nhóm tham khảo.

**Định vị FREVIA so với các hướng trên:**
- Khác biệt chính: các hướng này làm việc trên **field text thô** (đòi hỏi CV có cấu trúc field và thường cần labeled interaction data); FREVIA dùng **LLM trích xuất semantic view không cấu trúc → có cấu trúc**, phù hợp CV/JD dạng văn bản tự do, không cần interaction label (chỉ cần domain label để đánh giá).
- Cross-view interaction + adaptive fusion trong FREVIA giải quyết theo hướng nhẹ, interpretable, không cần fine-tune LLM.

### 3.3 Cross-view / cross-field interaction
- Motivation: đối sánh chéo khía cạnh (skill ↔ experience, project ↔ requirement) bắt được tín hiệu mà same-view bỏ sót.
- Cơ chế tham khảo: self-attention nội bộ document + cross-attention giữa hai document (InEXIT); co-training nhiều view bổ trợ nhau (Bian et al.).
- FREVIA áp dụng cross-attention trên **semantic views** (khác field text thô của các work trên).

### 3.4 Adaptive fusion
- Trọng số tĩnh vs trọng số học được; gating/attention để view góp trọng số khác nhau theo context; weighted sum linear fusion.
- FREVIA: học trọng số theo đặc trưng của JD (xem Methodology 4.5).

### 3.5 LLM-based information extraction (resume parsing)
- LLM (Qwen) trích xuất structured info từ văn bản không cấu trúc.
- Thách thức: parsing quality, hallucination, chi phí inference, JSON schema validation/fallback.
- FREVIA phụ thuộc bước này → bắt buộc có **đánh giá độ tin cậy extraction** (mục 5.3) vì sai số bước đầu lan truyền toàn pipeline.

### 3.6 Bảng định vị FREVIA vs related work

| Hướng | Biểu diễn | Interaction | Fusion | Nhãn cần cho học |
|---|---|---|---|---|
| Keyword / TF-IDF (2026) | whole-document | không | static (cosine) | không |
| SBERT | whole-document embedding | không | static | không (zero-shot) |
| InEXIT/DPGNN/PJFNN | field text thô | field-level | học | interaction label |
| Bian et al. (co-teaching) | multi-view | bổ trợ | học | interaction label |
| **FREVIA (đề xuất)** | **semantic view (LLM)** | **cross-view attention** | **adaptive (learned)** | **chỉ domain label (đánh giá)** |

### 3.7 Khoảng trống nghiên cứu
- Baseline semantic similarity chưa tận dụng cấu trúc multi-view.
- Các hướng multi-view hiện có cần interaction label và dựa trên field text thô.
- Chưa có sự kết hợp: **LLM-structured semantic views + cross-view interaction + adaptive fusion** cho bài toán ranking CV→JD không cần fine-tune LLM.

---

## 4. Phương pháp đề xuất (FREVIA) ← ĐÓNG GÓP

### 4.1 Kiến trúc tổng quan
```
JD ─┐
    ├─► [Qwen: Semantic View Extraction] ─► JD views (Title, Skills, Experience, Projects)
CV ─┘
                                                ▼
                             [Structured Multi-view Representation]
                                      (encode từng view riêng biệt)
                                                ▼
                             [Multi-view Matching + Cross-view Interaction]
                                      (ma trận view × view, kể cả cross)
                                                ▼
                             [Adaptive Fusion] ─► Final Matching Score ─► Ranking
```

### 4.2 Semantic View Extraction (Qwen)
- Đầu vào: văn bản CV/JD thô → đầu ra JSON schema cố định: `title`, `skills`, `experience`, `projects`.
- Prompt design: yêu cầu trích xuất theo JSON schema, kèm hướng dẫn trích nguyên cụm kỹ năng/kinh nghiệm/dự án.
- Robustness: parse + validate schema; có **fallback** (nếu JSON lỗi → retry 1 lần → bỏ bản ghi lỗi hoặc dựng view rỗng, ghi log sai số để báo cáo).
- Chi phí kiểm soát: batching, dùng Qwen bản nhỏ phù hợp tài nguyên nhóm.

### 4.3 Structured Multi-view Representation
- Mỗi view của CV và JD encode riêng thành vector (không gộp như baseline).
- **Lựa chọn mặc định**: semantic embedding (SBERT `all-MiniLM-L6-v2`) cho từng view để giữ đối sánh ngữ nghĩa; **alternative**: TF-IDF per-view (benchmark tính chủ động, ablation đổi encoder).
- Chuẩn hoá vector L2 trước khi tính similarity.

### 4.4 Multi-view Matching & Cross-view Interaction
- Ma trận tương tác **S** ∈ R^(V×V), S[i,j] = sim(view_CV_i, view_JD_j), V = 4.
- **Same-view**: Title↔Title, Skills↔Skills, Experience↔Experience, Projects↔Projects.
- **Cross-view** (mở rộng, tín hiệu liên khía cạnh): Skills↔Experience, Experience↔Projects, Projects↔Skills, Title↔Skills, Title↔Experience, ...
- Cơ chế interaction: biến đổi ma trận S qua cross-attention giữa các view của hai phía → vectơ tín hiệu cross-view; tổng hợp thành **tín hiệu same-view** và **tín hiệu cross-view** riêng biệt để phục vụ ablation.

### 4.5 Adaptive Fusion
- Trọng số học được: α(view|JD) = softmax(g(JD_rep)), với g là MLP/gating nhẹ nhận đặc trưng JD → trọng số cho từng tín hiệu (same-view từng loại + cross-view).
- **Cách học (trả lời trực tiếp câu hỏi giám khảo):** giám sát yếu (weakly supervised) trên **domain label** — coi CV thuộc target domain là relevant (đúng cách paper 2026 xây GT); train tầng fusion (chỉ vài nghìn tham số) bằng **pairwise ranking loss (BPR) / cross-entropy** trên cặp (relevant, non-relevant) với cùng một JD. Không fine-tune Qwen.
- **Baseline fusion** để ablation: trọng số đồng đều (trung bình cộng) và trọng số cố định (heuristic).
- Điểm final chuẩn hoá [0,1] đồng nhất với baseline score.

### 4.6 Scoring & Ranking
- Ranking ứng viên theo final matching score của từng JD.
- (Tùy chọn) greedy one-to-one matching theo baseline để bảo toàn công bằng giữa các ứng viên; so sánh có/không có ràng buộc này.

### 4.7 Các biến thể rút gọn (variants) cho ablation

| Ký hiệu | Thành phần | Trả lời |
|---|---|---|
| FREVIA-min | multi-view + same-view matching + fusion đều | RQ1 (vs whole-document) |
| FREVIA-cross | FREVIA-min + cross-view interaction (fusion đều) | RQ2 (giá trị cross-view) |
| FREVIA-full | FREVIA-cross + adaptive fusion | RQ3 (giá trị adaptive) |

---

## 5. Dataset + Ground Truth

### 5.1 Dataset
- **JD**: Glassdoor "Data Science" (Kaggle) — 660 bài.
- **CV**: Kaggle "Resume Dataset" — 962 hồ sơ, 25 domain (Data Science, Hadoop, Civil Engineering, Operations Manager, Testing, Network Security...).
- Đánh giá theo domain trọng tâm: Data Science (40 CV) và Hadoop (42 CV) — khớp protocol paper 2026 (tổng 82 CV, Precision@10/Recall@10).
- (Tùy chọn) bổ sung dữ liệu mô phỏng theo paper để tái hiện Table 2 / Fig. 4; ghi rõ giới hạn single-domain JD (bias Data Science terminology).

### 5.2 Ground Truth
- Không có nhãn recruiter chính thức → theo paper, dùng **domain label** của CV làm nhãn relevance: CV thuộc domain trùng JD = relevant.
- Metric theo domain:
  - Precision@10 = số CV thuộc target domain trong top-10 / 10.
  - Recall@10 = số CV đó / tổng CV relevant trong pool (40 hoặc 42).
- Nhãn này đồng thời dùng làm **nhãn giám sát yếu** cho bước adaptive fusion (4.5) — cần tách tàu/test theo domain để tránh rò rỉ.

### 5.3 Đánh giá độ tin cậy LLM extraction (Qwen)
- Trên mẫu đối chiếu thủ công (~50 CV + ~50 JD), đo: tỷ lệ JSON hợp lệ; field-level accuracy trên `skills` (precision/recall so với chú thích tay); độ ổn định khi thay đổi prompt/thử lại (consistency).
- Báo cáo sai số extraction và lan truyền lỗi lên điểm cuối (độ nhạy của pipeline theo chất lượng extraction).

---

## 6. Experiment (Thiết kế thực nghiệm)

### 6.1 Phương pháp so sánh
1. Keyword-based.
2. TF-IDF + cosine (paper 2026).
3. SBERT (`all-MiniLM-L6-v2`).
4. FREVIA-min / FREVIA-cross / FREVIA-full.

### 6.2 Metrics
- Chính: **Precision@10, Recall@10** (khớp paper).
- Bổ sung: MRR@10, NDCG@10, Accuracy@k.
- Kiểm định: **Wilcoxon signed-rank test** (không tham số, khớp paper) + **Cohen's d**, mức ý nghĩa α = 0.05.

### 6.3 Protocol đánh giá
- Cùng bộ dữ liệu, cùng bộ chia (train/test theo domain để fusion học/baseline đánh giá không rò rỉ), cùng seed.
- Với FREVIA: train tầng fusion trên target domain train-split; đánh giá trên test.
- Lặp lại nhiều target domain (Data Science, Hadoop) như paper.

### 6.4 Ánh xạ hypothese → thiết kế
| Hypotheses | Thiết kế trả lời |
|---|---|
| H1 | FREVIA-full vs TF-IDF trên P@10/R@10 từng domain |
| H2 | FREVIA-cross vs FREVIA-min |
| H3 | FREVIA-full vs FREVIA-cross (và vs fusion tĩnh cùng thành phần) |
| H4 | FREVIA-full vs SBERT |

### 6.5 Ablation & phân tích qualitative
- Ablation encoder (SBERT per-view vs TF-IDF per-view).
- Ảnh hưởng chất lượng extraction (chạy pipeline với extraction "sạch" vs raw-fallback).
- Case study cross-domain (như Table 3: Civil Engineer → RFP Data Analyst) kèm phân tích view-matched để giải thích vì sao FREVIA hơn baseline.
- Phân tích độ trễ/chi phí: LLM extraction vs TF-IDF (runtime, tài nguyên).

### 6.6 Tính khả thi ngắn hạn
- Dataset công khai, không cần thu thập.
- Qwen dùng bản nhỏ (chạy được trên GPU sinh viên / API), chỉ fine-tune tầng fusion nhỏ.
- Pipeline modular: tách baseline (chạy trước) → extraction → matching → fusion, từng khối kiểm chứng được độc lập.

---

## 7. Kết quả + phân tích (dự kiến)

### 7.1 Kết quả dự kiến
- Bảng chính: P@10/R@10 theo domain (Data Science, Hadoop) cho keyword / TF-IDF / SBERT / FREVIA-min / FREVIA-cross / FREVIA-full.
- Bảng ablation 3 biến thể + bảng ảnh hưởng encoder/extraction.
- Thống kê Wilcoxon + Cohen's d cho từng cặp so sánh H1–H4.

### 7.2 Phân tích
- Vì sao FREVIA hơn baseline: chi tiết view-matched, cross-domain case study.
- Khi nào FREVIA không thắng (JD mơ hồ, CV thưa, extraction lỗi) → nêu hạn chế.
- Trade-off LLM extraction vs TF-IDF (chi phí, độ trễ, mức tăng điểm đáng giá không).
- Bias & fairness: giữ nguyên nguyên tắc paper (loại thuộc tính nhạy cảm, one-to-one matching); fairness audit để future work.

---

## 8. Kế hoạch thực hiện (thời gian ngắn hạn)

| Giai đoạn | Nội dung | Kết quả (deliverable) |
|---|---|---|
| 1 | Reproduce paper 2026 + 3 baseline (keyword, TF-IDF, SBERT) | Baseline reproducible + số liệu đối chiếu |
| 2 | Pipeline extraction Qwen + đánh giá độ tin cậy (5.3) | JSON views + báo cáo extraction accuracy |
| 3 | Multi-view representation + same-view matching (FREVIA-min) | Mô-đun chạy được, ablation RQ1 |
| 4 | Cross-view interaction (FREVIA-cross) + adaptive fusion (FREVIA-full) | FREVIA hoàn chỉnh |
| 5 | Thí nghiệm + ablation + thống kê (Wilcoxon/Cohen's d) | Bảng kết quả H1–H4 |
| 6 | Viết báo cáo, slide, case study | Khóa luận hoàn chỉnh |

---

## 9. Kết luận, Hạn chế, Hướng phát triển

### 9.1 Kết luận
Tóm tắt đóng góp (Contributions 1–3) và trả lời RQ1–RQ4 / H1–H4.

### 9.2 Hạn chế
- Phụ thuộc chất lượng LLM extraction (hallucination, format lỗi).
- Chỉ tiếng Anh, JD đơn domain (Data Science) → bias từ vựng.
- Chi phí inference LLM; không xử lý cold-start domain; GT dùng domain label là proxy cho quyết định tuyển dụng thật.

### 9.3 Hướng phát triển
- Multilingual (dịch preprocessing hoặc cross-lingual embedding).
- Mở rộng view: education, certification, location, salary expectation.
- Tối ưu: ANN search, batching, distillation cho LLM extraction; khám phá tầng fusion lớn hơn hoặc fine-tune nhẹ Qwen có kiểm soát bias.
- Fairness audit bằng synthetic demographics (khớp đề xuất paper).