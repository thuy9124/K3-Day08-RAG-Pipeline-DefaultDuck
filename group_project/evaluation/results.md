# Báo cáo benchmark RAG pháp luật lao động

Kết quả được tạo từ lần chạy thật, hoàn toàn local. Bốn chỉ số dưới đây là **proxy deterministic theo token overlap**, không phải điểm RAGAS do LLM chấm.

## A/B retrieval

| Metric | Hybrid local (BM25 + structural + RRF) | Dense local (char n-gram + Chroma) |
|---|---:|---:|
| Faithfulness | 1.0000 | 0.9630 |
| Answer relevancy | 0.5012 | 0.5212 |
| Context recall | 0.9721 | 0.8676 |
| Context precision | 1.0000 | 1.0000 |

## Chi tiết cấu hình Hybrid local

| # | Retrieved | Safe reject | Faithfulness | Relevancy | Recall | Precision |
|---:|---:|:---:|---:|---:|---:|---:|
| 1 | 5 | no | 1.000 | 0.244 | 1.000 | 1.000 |
| 2 | 5 | no | 1.000 | 0.277 | 0.969 | 1.000 |
| 3 | 5 | no | 1.000 | 0.500 | 0.947 | 1.000 |
| 4 | 5 | no | 1.000 | 0.609 | 1.000 | 1.000 |
| 5 | 5 | no | 1.000 | 0.418 | 1.000 | 1.000 |
| 6 | 5 | no | 1.000 | 0.526 | 0.857 | 1.000 |
| 7 | 5 | no | 1.000 | 0.387 | 1.000 | 1.000 |
| 8 | 5 | no | 1.000 | 0.345 | 0.968 | 1.000 |
| 9 | 5 | no | 1.000 | 0.294 | 0.929 | 1.000 |
| 10 | 5 | no | 1.000 | 0.265 | 0.964 | 1.000 |
| 11 | 5 | no | 1.000 | 0.423 | 0.947 | 1.000 |
| 12 | 5 | no | 1.000 | 0.230 | 1.000 | 1.000 |
| 13 | 5 | yes | 1.000 | 1.000 | 1.000 | 1.000 |
| 14 | 5 | yes | 1.000 | 1.000 | 1.000 | 1.000 |
| 15 | 5 | yes | 1.000 | 1.000 | 1.000 | 1.000 |

## Phạm vi và lưu ý

- 12 câu đúng-domain được đối chiếu với Bộ luật Lao động 2019 và Luật BHXH 2024 trong corpus.
- 3 câu ngoài domain phải trả về từ chối an toàn.
- Muốn có điểm RAGAS chính thức cần LLM judge; chế độ đó bị tắt để tuân thủ yêu cầu không phát sinh chi phí.
