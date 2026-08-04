# RAG pháp luật lao động — bản chạy local không phát sinh chi phí

## Phạm vi dữ liệu

- 4 văn bản gốc: Bộ luật Lao động 2019, Luật BHXH 2024, Nghị định 145/2020/NĐ-CP và tài liệu hỏi đáp pháp luật lao động.
- 7 bài viết/hướng dẫn của Cổng Thông tin điện tử Chính phủ; vượt yêu cầu Checkpoint 1 là tối thiểu 5 bài.
- 11 file Markdown chuẩn hóa, được chia thành chunk 800 ký tự, overlap 100.

Benchmark có 15 câu: 12 câu đúng domain và 3 câu ngoài domain. Corpus hiện tại có bằng chứng cho cả 12 câu đúng domain nên không cần crawl thêm. Câu thai sản cho lao động nam đã được cập nhật theo Điều 53 Luật BHXH 2024.

## Kiến trúc zero-cost

`Markdown → chunking → local char n-gram embedding → ChromaDB`

`Query → BM25 + structural search → RRF(k=60) → evidence guard → extractive answer + citation`

- Không dùng Voyage/OpenRouter/PageIndex cloud mặc định.
- `RAG_ALLOW_EXTERNAL_APIS=false` là chế độ mặc định.
- Task 8 dùng structural index local, giữ interface PageIndex của bài lab nhưng không upload.

## Chạy trong conda env `action`

Từ thư mục project, dùng trực tiếp `conda.exe` vì PowerShell hiện chưa được `conda init`:

```powershell
$conda = "C:\Users\ADMIN\anaconda3\Scripts\conda.exe"
$env:PYTHONIOENCODING = "utf-8"

& $conda run --no-capture-output -n action python -m src.task1_collect_legal_docs
& $conda run --no-capture-output -n action python -m src.task2_crawl_news
& $conda run --no-capture-output -n action python -m src.task3_convert_markdown --refresh
& $conda run --no-capture-output -n action python -m src.task4_chunking_indexing
& $conda run --no-capture-output -n action python -m src.task8_pageindex_vectorless
& $conda run --no-capture-output -n action python -m group_project.evaluation.eval_pipeline
& $conda run --no-capture-output -n action python -m pytest tests/test_individual.py tests/test_checkpoint34_local.py -q
```

Task 2 mặc định chỉ xác thực 7 bài đã crawl, không gọi mạng. Chỉ dùng `python -m src.task2_crawl_news --refresh` khi chủ động muốn crawl lại tối đa 15 bài.

UI (không bắt buộc khi test pipeline):

```powershell
& $conda run --no-capture-output -n action streamlit run app.py
```

## Demo Checkpoint 6

1. Trình bày corpus và lý do không crawl thêm.
2. So sánh Task 5 dense local với Task 6 BM25.
3. Giải thích RRF và structural fallback local.
4. Hỏi một câu về thử việc hoặc OT, kiểm tra citation.
5. Hỏi một câu ngoài domain, kiểm tra câu từ chối an toàn.
6. Mở `evaluation/results.md` để trình bày A/B benchmark chạy thật.

Việc thuyết trình trực tiếp và đẩy code lên GitHub vẫn là thao tác của nhóm; repository đã có code, benchmark và kịch bản demo sẵn sàng.
