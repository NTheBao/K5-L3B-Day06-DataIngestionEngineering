# Starter · Chroma

Bộ pipeline local cho lab cá nhân. Pipeline đọc PDF, tạo embedding bằng Sentence Transformers và lưu vector bền vững trong Chroma. Mã hỗ trợ hai cách chia chunk; cấu hình C dùng lại index B và chỉ thêm metadata filter khi câu hỏi có filter.

## Cấu trúc

```text
starter/
├── requirements.txt       # ChromaDB, pypdf, python-dotenv và Sentence Transformers
├── .env.example           # cấu hình mẫu; copy thành .env
├── scripts/check_setup.py # kiểm tra Chroma persistent client (đã chạy được)
├── src/
│   ├── models.py          # cấu trúc dữ liệu mẫu
│   ├── ingestion.py       # đọc PDF, chuẩn hóa và gắn page/section
│   ├── chunking.py        # fixed-size và paragraph/step-aware
│   ├── embeddings.py      # Sentence Transformers local
│   ├── store.py           # Chroma persistent, cosine và metadata filter
│   ├── evaluation.py      # phần đánh giá top-k được hoàn thiện ở bước tiếp
│   └── run.py             # nạp dữ liệu, lập index A/B và báo cáo
├── data/                  # bạn tự chuẩn bị corpus được phép sử dụng
└── results/               # lưu kết quả của bạn
```

## 1. Cài và kiểm tra database

Cần Python **3.11–3.14** và mạng để tải dependencies lần đầu. Các lệnh sau chạy từ thư mục `starter/`:

```bash
cd starter
python3 -m venv .venv
```

Kích hoạt `.venv`: macOS/Linux dùng `source .venv/bin/activate`; Windows PowerShell dùng `.venv\Scripts\Activate.ps1`. Sau đó:

```bash
python -m pip install -r requirements.txt
# macOS/Linux: cp .env.example .env
# Windows PowerShell: Copy-Item .env.example .env
python scripts/check_setup.py
```

Kỳ vọng: hiện phiên bản ChromaDB, đường dẫn lưu local và `Setup OK`. Đây **chỉ là kiểm tra setup**, không phải kết quả benchmark. `CHROMA_PATH` mặc định là `data/chroma` trong thư mục `starter/`; thư mục này chứa database bền vững của bạn. Không xóa nó nếu cần giữ dữ liệu.

## 2. Chọn embedding model

Mặc định dùng `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` trên CPU; cần mạng để tải model lần đầu, không cần API key. Đổi `EMBEDDING_MODEL` trong `.env` nếu chọn model khác. Giữ nguyên model, cách chuẩn hóa vector và số chiều giữa mọi chiến lược; không commit `.env`.

## 3. Chạy pipeline ingestion

Từ PowerShell, chạy lệnh trong thư mục `starter/` sau khi tạo/kích hoạt môi trường và cài `requirements.txt`:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
python -m src.run
```

Mặc định pipeline dùng PDF trong `data/`, benchmark `data/benchmark.json`, Chroma ở `data/chroma/` và ghi báo cáo timestamped vào `results/`. Có thể truyền `--pdf`, `--benchmark`, `--results`, `--collection` hoặc `--model`. Báo cáo lưu fingerprint nguồn/benchmark, model và version, số chiều, metadata section, số chunk/ký tự/token, thời gian, chi phí local bằng 0 và trang lỗi/không nhận diện được section.

Chiến lược A dùng chunk 500 ký tự, overlap 50; B gom khối/step theo cấu trúc trang với giới hạn 300 ký tự. C chia sẻ nguyên index B, chỉ lọc theo `section` khi filter được cung cấp trước truy xuất. `python -m src.run` tạo index, chưa chạy bộ câu hỏi top-3 hoặc chấm điểm thủ công; xem [BENCHMARK.md](../BENCHMARK.md) cho bước đánh giá.

Nạp lại cùng nguồn upsert chunk có ID ổn định và xóa chunk cũ không còn dùng trong phạm vi tài liệu/chiến lược đó. Không xóa database để đổi model: dùng collection mới hoặc xử lý migration có chủ đích. Lưu code, cấu hình, kết quả và lệnh chạy để người chấm tái lập; không commit key hoặc dữ liệu không có quyền chia sẻ.
