# VIETNAM Bank Campaign Project

## 1) Mục tiêu dự án
Bài toán: **dự đoán xác suất khách hàng đăng ký term deposit** để tối ưu danh sách gọi điện/marketing.

Output chính của dự án:
- **Model dự đoán xác suất (propensity)** trên từng khách hàng.
- **Call list top-K** (ví dụ top 10%) xuất ra CSV để bộ phận Growth/Marketing ưu tiên gọi trước.
- Pipeline: tách ingestion → training → batch scoring, lưu artifacts, có thể container hóa.

---

## 2) Business framing
Trong chiến dịch call center, ngân sách gọi điện là hữu hạn. Nếu ta gọi ngẫu nhiên, tỷ lệ chuyển đổi chỉ ~9.3%.  
Mục tiêu là **xếp hạng khách hàng theo xác suất chuyển đổi**, gọi nhóm có điểm cao trước để:
- tăng conversion rate trong nhóm được gọi,
- giảm chi phí/1 conversion,
- cải thiện hiệu quả chiến dịch.

Metric business trọng tâm:
- **Precision@K%**: tỷ lệ chuyển đổi trong nhóm top K% (ví dụ top 10%).
- **Lift@K%**: Precision@K% / baseline conversion rate.

---

## 3) Dataset & dữ liệu đầu vào
- Dữ liệu khách hàng + lịch sử chiến dịch marketing.
- Dữ liệu sau khi làm sạch được lưu ở:
   `data/interim/bank_customers_clean.pandas.parquet`

Thông tin sau ingestion:
- Rows: **42,639**
- Columns: **17**
- Positive rate (label=1): **~9.29%**
- Missing values: **gần như 0**

---

## 4) Lưu ý quan trọng: Data Leakage
Feature `duration` (thời lượng cuộc gọi) có tương quan cao với label vì nó chỉ xuất hiện **sau khi đã gọi**.
Nếu dùng `duration`, model sẽ nhìn trước được tương lai gây ra hiện tượng data leakage và không dùng được trong thực tế “pre-call”.

✅ Vì vậy mô hình trong repo là **PRE-CALL model**:
- **drop `duration`** khi train/predict.

---

## 5) Quy trình thực hiện (End-to-end)
Pipeline gồm 3 bước chính:

1) **Ingestion / Cleaning**
   - đọc raw csv (hoặc nguồn input)
   - làm sạch
   - lưu parquet ở `data/interim/...`

2) **EDA + Insight**
   - phân phối numeric (age, balance, campaign, pdays, previous…)
   - conversion breakdown theo nhóm (month/job/education/loans…)
   - phân tích segment & lift theo logic marketing

3) **Model training + Evaluation**
   - baseline: Gradient Boosting models (LightGBM & XGBoost)
   - preprocessing: OneHotEncoder cho categorical
   - metric: ROC-AUC, PR-AUC, Precision@K, Lift@K
   - lưu model artifacts `.joblib`

4) **Batch Scoring / Export Call List**
   - load model
   - dự đoán score
   - xuất top-K file CSV

---

## 6) Repo structure
```
vn-bank-campaign-ml/
├─ src/
│ ├─ ingestion/
│ │ ├─ load_data_raw.py # đọc raw -> clean -> lưu parquet ở data/interim/
│ │ └─ check_interim.py # kiểm tra nhanh: shape, label rate, missing, stats
│ │
│ ├─ modeling/
│ │ ├─ train.py # train LightGBM & XGBoost + save models/metrics
│ │ ├─ predict.py # batch scoring + export call list top-K (CSV)
│ │ └─ evaluate_calllist.py # (optional) phân tích composition + actual precision/lift
│ │
│ ├─ notebooks/ # (optional) EDA/feature engineering ipynb
│ │ └─ eda_feature_insights.ipynb
│ │
│ ├─ common/ # (optional) helper: path, config, io, logging...
│ │ └─ utils.py
│ │
│ └─ init.py
│
├─ configs/ # (optional) config YAML/JSON cho paths, seed, params
│ └─ config.yaml
│
├─ data/
│ ├─ raw/ # dữ liệu thô (csv) / input ban đầu (KHÔNG commit nếu lớn)
│ │ └─ bank_marketing_raw.csv
│ │
│ └─ interim/ # dữ liệu sau cleaning/standardize
│ └─ bank_customers_clean.pandas.parquet
│
├─ artifacts/
│ ├─ models/ # model đã train (joblib)
│ │ ├─ precall_lgbm.joblib
│ │ └─ precall_xgb.joblib
│ │
│ ├─ metrics/ # metrics & báo cáo đánh giá model
│ │ └─ test_metrics.csv
│ │
│ └─ predictions/ # output scoring cho marketing/call center
│ └─ call_list_top10.csv
│
├─ requirements.txt # dependencies của project
├─ Dockerfile # (optional) containerization cho batch scoring/cloud
├─ .gitignore # ignore .venv/, data/raw lớn, artifacts tạm...
└─ README.md # tài liệu dự án (mục tiêu, run, results, insights)
```

---

## 7) Environment setup (Windows)
### 7.1 Tạo venv
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
### 7.2 Dependencies tối thiểu

pandas, numpy, pyarrow

scikit-learn, joblib

lightgbm, xgboost

matplotlib (EDA)

## 8) Run pipeline (local)
### Step 1 — Ingestion
```python
python -m src.ingestion.load_data_raw
```

Output:
```
data/interim/bank_customers_clean.pandas.parquet
```
### Step 2 — Quick check
```python
python -m src.ingestion.check_interim
```

In ra shape, columns, label distribution, summary stats.

### Step 3 — Train models
```python
python -m src.modeling.train --model both
```

Artifacts:
```
artifacts/models/precall_lgbm.joblib

artifacts/models/precall_xgb.joblib

artifacts/metrics/test_metrics.csv
```
### Step 4 — Batch scoring + export call list top 10%
```python
python -m src.modeling.predict --model_path artifacts/models/precall_xgb.joblib --top_k 0.10
```

Output:
```
artifacts/predictions/call_list_top10.csv
```
## 9) Kết quả mô hình (Model Results)

### Test metrics (trích từ artifacts/metrics/test_metrics.csv):

| Model    |    ROC-AUC |     PR-AUC | Precision@10% |  Lift@10% | Precision@20% |  Lift@20% |
| -------- | ---------: | ---------: | ------------: | --------: | ------------: | --------: |
| LightGBM |     0.6957 |     0.2536 |          0.30 |     3.23× |        0.2102 |     2.26× |
| XGBoost  | **0.7113** | **0.2836** |      **0.35** | **3.77×** |    **0.2273** | **2.45×** |


Business interpretation:

### Baseline conversion rate ~9.3%.

Với top 10% khách hàng theo XGBoost score, conversion trong nhóm đạt ~35%
→ hiệu quả gấp ~3.77 lần so với gọi ngẫu nhiên.

=> Model được chọn cho batch scoring: XGBoost.

## 10) EDA insights đáng giá (Marketing/Growth)
### 10.1 Conversion theo tháng (month)

Conversion rate khác nhau rất mạnh theo month.
Nhóm “peak months” có conversion cực cao nhưng n nhỏ, cần dùng như tín hiệu “timing”.

### 10.2 Warm leads

Feature pdays cho biết khách hàng đã từng được liên hệ trước đó.

pdays != -1 (warm leads) có conversion cao hơn rõ rệt.

### 10.3 Loans effect

Nhóm không có loan & không có housing loan thường có conversion cao hơn.

### 10.4 Segment lift (gợi ý chiến lược)

Ví dụ segment analysis cho thấy:

Peak months: lift ~4.9×

Warm leads: lift ~1.6×

No loans: lift ~1.5×

Combo Peak+Warm có thể đạt lift ~5.5×

=> Có thể biến thành rule-based strategy hoặc feature để model học.

## 11) Batch call list output (sample fields)

File call_list_top10.csv gồm:
```
ID, score

context fields: month, job, education, loan, housing, pdays, previous

segment hints: warm_lead, no_loans, peak_month

```
Dùng cho stakeholder:

Marketing/Growth ưu tiên gọi top list

hoặc chạy A/B test: gọi top list vs random list để đo uplift thực tế


## 12) Production readiness 
### 12.1 Batch architecture 
Kiến trúc cho batch scoring:

Storage: object store (ví dụ S3-compatible)

Batch compute: container job (ECS/Fargate, K8s job, VM cron)

Output: CSV/Parquet về object store

Logging/monitoring: CloudWatch/ELK

### 12.2 Dockerization

Repo cung cấp Dockerfile để đóng gói batch scoring.

---
## Conclusion

Như vậy, với yêu cầu bài toán đặt ra là: với ngân sách gọi điện/marketing hữu hạn, làm sao **ưu tiên đúng khách hàng** để tăng tỷ lệ đăng ký term deposit thay vì gọi ngẫu nhiên (baseline conversion ~9.3%). Mục tiêu là **ưu tiên đúng khách hàng** để tăng tỷ lệ chuyển đổi và doanh thu.

**Sau khi phân tích EDA, mình rút ra:**
- Conversion khác nhau rất mạnh theo **month**; tồn tại nhóm “peak months” có conversion cao hơn rõ rệt.
- Nhóm **warm lead** (`pdays != -1`) chuyển đổi cao hơn so với khách chưa từng được liên hệ.
- Nhóm **không có loan & không có housing loan** có xu hướng chuyển đổi tốt hơn.
  
**Sau khi train và đánh giá mô hình, mình thu được:**
- So sánh hiệu suất 2 mô hình LightGBM vs XGBoost cho thấy **XGBoost** tốt hơn với: ROC-AUC = 0.711, PR-AUC = 0.284.
- Quan trọng nhất cho marketing: **Lift@10% = 3.77× (Precision@10% ≈ 35%)**  
  → nghĩa là nếu chỉ gọi **top 10%** theo score, tỷ lệ chuyển đổi trong nhóm này cao gấp ~3.77 lần so với gọi ngẫu nhiên.

**Dựa trên các kết quả đó, action đề xuất để tăng doanh thu:**
1) **Triển khai batch scoring định kỳ** (daily/weekly) để tạo `call_list_topK.csv`, ưu tiên gọi nhóm score cao trước → tăng số lượng conversion trong cùng call budget.
2) **Tập trung nguồn lực theo segment hiệu quả**: ưu tiên warm leads và giai đoạn peak months; với nhóm low months có thể giảm tần suất hoặc dùng ưu đãi khác để cải thiện ROI.
3) **Đo lường uplift thực tế** bằng A/B test: top-K (treatment) vs random (control) để chứng minh tác động lên doanh thu và tối ưu ngưỡng K (10%, 20%…).
4) **Vận hành bền vững**: theo dõi drift/hiệu suất theo thời gian và retrain khi dữ liệu hành vi thay đổi.

Tóm lại, dự án chứng minh rằng việc chuyển từ “gọi đại trà” sang “ưu tiên theo propensity” có thể **tăng hiệu quả chiến dịch rõ rệt**, và tạo ra một quy trình có thể tích hợp vào vận hành thực tế để cải thiện doanh thu.

