# Báo cáo cá nhân — Day 6

> Bản nháp theo kết quả chạy ngày 2026-11-14. Điền tên/MSSV, xác minh quyền dữ liệu và hoàn thành các nhãn thủ công trước khi nộp. Đổi tên file theo MSSV và họ tên thật.

## Dữ liệu và pipeline

- **Chủ đề:** Truy xuất hướng dẫn sử dụng ZaloPay liên kết Agribank.
- **Nguồn:** [`Tài liệu hướng dẫn sử dụng ZaloPay (App)`](../../starter/data/HUONG+DAN+LIEN+KET+THE+AGRIBANK+VOI+ZALOPAY.pdf); manifest tại [`data/manifest.md`](../../data/manifest.md).
- **Trang:** 62 trang vật lý, 60 trang nội dung. Trang nội dung là trang có text trích xuất được và nhận diện heading; trang 1–2 không có heading section. `pypdf`, không dùng OCR; không có trang mất text.
- **Chuẩn hóa:** Unicode NFKC; giữ liên kết document/page/section. Có text ở cả 62 trang nhưng trang 1–2 không vào tập nội dung do thiếu section.
- **Embedding:** Sentence Transformers local, `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, package 6.3.0, 384 chiều, CPU, encode query/document cùng cách; chi phí dịch vụ $0.
- **Vector store:** Chroma 1.5.9 persistent, cosine, HNSW approximate nearest-neighbor; database `starter/data/chroma`.
- **Metadata:** `document_id`, `source`, `page`, `section`. `section` là string trích từ heading trong PDF, dùng làm filter theo ngữ cảnh người dùng cung cấp trước truy xuất.
- **Lệnh:** từ `starter/`, `python -m src.run` (trên máy này chạy bằng `.venv\Scripts\python.exe`). Benchmark mặc định ở `../benchmark/benchmark.json`; output ở `../results/`.

## Cấu hình và benchmark

- **A:** fixed-size 500 ký tự, overlap 50.
- **B:** gom paragraph/step theo cấu trúc trang, tối đa 300 ký tự.
- **C:** cùng chunks/index B; chỉ thêm filter `section` khi câu hỏi có filter. Vì vậy B–C chỉ khác lọc truy xuất.
- Dùng chung corpus, model, vector store, metric cosine và bộ benchmark. Bộ câu hỏi có 20 câu (15 dev / 5 test), 7 câu filter trong đó 1 câu test. Gold quote và trang đã được đối chiếu với PDF; test được giữ cố định.
- **Selection policy ghi trước lần chạy retrieval:** ưu tiên dev Evidence Hit@3, sau đó dev Hit@1, rồi thời gian build index thấp hơn; không dùng test để chọn.
- **Nguy cơ rò rỉ:** chỉ có một PDF; các câu test lấy từ các sự kiện khác nhau nhưng vẫn có thể trùng chủ đề/section với dev. Kết luận không đại diện cho tài liệu khác.

## Kết quả retrieval

| Strategy | Split | Hit@1 | Evidence Hit@3 | Chunks | Index build (s) | Chi phí |
|---|---|---:|---:|---:|---:|---:|
| A | Dev | 8/15 | 11/15 | 61 | 2.6318 | $0 |
| A | Test | 3/5 | 5/5 | 61 | 2.6318 | $0 |
| B | Dev | 8/15 | 8/15 | 66 | 1.7381 | $0 |
| B | Test | 2/5 | 4/5 | 66 | 1.7381 | $0 |
| C | Dev | 8/15 | 9/15 | 66 | 1.7381* | $0 |
| C | Test | 2/5 | 4/5 | 66 | 1.7381* | $0 |

`*` C chia sẻ index B, không tạo embeddings/storage bổ sung. Tổng search time trên 20 câu: A 0.1484s, B 0.1343s, C 0.1324s.

| Strategy | Metadata đúng trên 5 câu test × top-3 | Filter test tuân thủ |
|---|---:|---:|
| A | 15/15 | N/A — không bật filter |
| B | 15/15 | N/A — không bật filter |
| C | 15/15 | 1/1 |

Metadata `source/page` giúp truy vết; `section` cho phép giới hạn phạm vi. Hạn chế: chỉ có một PDF và một câu test filter, nên bằng chứng về tính hữu dụng của filter còn yếu.

### Coherence và grounding (chờ chấm thủ công)

Phiếu [`manual-review-20261114T093454861706Z.json`](../../results/manual-review-20261114T093454861706Z.json) có 10 trang nguồn cố định dùng chung cho A/B/C (30 mục coherence) và 5 câu test cho mỗi chiến lược (15 mục grounding). Các điểm/lý do đang để trống (`pending_author_review`); chưa báo trung bình hoặc phân bố điểm để tránh coi retrieval metric là nhãn thủ công.

Trace đầy đủ: [`traces-20261114T093454861706Z.json`](../../results/traces-20261114T093454861706Z.json). Báo cáo pipeline: [`pipeline-20261114T093454861706Z.json`](../../results/pipeline-20261114T093454861706Z.json).

## Phân tích và kết luận

- Theo selection policy, **A được chọn trên dev** với Evidence Hit@3 11/15, so với C 9/15 và B 8/15. Test chỉ được báo cáo sau khi chọn.
- **A–B:** B giảm 3 hit Evidence Hit@3 trên dev và 1 hit trên test; Hit@1 dev bằng nhau. B có ít thời gian build hơn trong lần chạy này nhưng nhiều hơn 5 chunks.
- **B–C:** C tăng 1 hit Evidence Hit@3 trên dev; test không đổi. Filter đạt 1/1 câu test có filter nhưng mẫu nhỏ; C dùng lại index B.
- Ví dụ cần xem trace: **D04** (evidence trang 11 về hai cách đăng nhập và trường hợp chưa có tài khoản Zalo) không đạt exact Evidence Hit@3 ở cả ba cấu hình. **T19** (trang 57, điều kiện OTP và giao dịch dưới 500.000đ) đạt ở A nhưng không đạt exact Evidence Hit@3 ở B/C. Đây là lỗi khớp quote tự động; cần đọc top-3 và chấm grounding để xác nhận liệu có bằng chứng tương đương.
- Kết luận giới hạn ở corpus này. Thời gian là một lần chạy local; test chỉ có 5 câu. Chưa có kết luận coherence/grounding cho tới khi điền phiếu audit.

## Trước khi nộp

- [ ] Điền họ tên/MSSV trong [`STUDENT.md`](../../STUDENT.md) và đổi tên file báo cáo.
- [ ] Xác minh quyền sử dụng PDF, ghi URL/nhà phát hành vào manifest; không nộp file nếu không có quyền.
- [ ] Điền điểm, lý do và người chấm trong phiếu audit; cập nhật kết quả thủ công ở báo cáo.
- [ ] Kiểm tra repo và nộp link theo hướng dẫn lớp.
