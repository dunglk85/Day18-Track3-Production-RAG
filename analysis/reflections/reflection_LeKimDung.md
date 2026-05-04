# Individual Reflection — Lab 18

**Tên:** Lê Kim Dung  
**Module phụ trách:** M1: Chunking & Integration (Pipeline)

---

## 1. Đóng góp kỹ thuật

- Module đã implement: M1 (Hierarchical Chunking) và ghép nối toàn bộ Production Pipeline.
- Các hàm/class chính đã viết: `chunk_hierarchical`, `build_pipeline`, `run_query`, `evaluate_pipeline`.
- Số tests pass: 8/8 (M1)

## 2. Kiến thức học được

- Khái niệm mới nhất: RRF (Reciprocal Rank Fusion) và cách kết hợp BM25 với Dense Search.
- Điều bất ngờ nhất: RAGAS evaluation có thể gặp lỗi `NaN` hoặc `TypeError` chỉ vì cấu hình model embeddings không tương thích.
- Kết nối với bài giảng (slide nào): Slide về Hybrid Search và Reranking (Module 2 & 3) và Diagnostic Tree cho failure analysis.

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất: Gặp lỗi `TypeError: unhashable type: 'list'` và `UnicodeEncodeError` trên Windows khi chạy pipeline.
- Cách giải quyết: Debug kỹ logic tạo key cho RRF, chuyển đổi metadata sang dạng hashable (tuple) và cấu hình lại `sys.stdout` sang UTF-8.
- Thời gian debug: Khoảng 1 giờ.

## 4. Nếu làm lại

- Sẽ làm khác điều gì: Sẽ kiểm tra kỹ tính hashable của metadata ngay từ bước Enrichment (M5) để tránh lỗi ở các bước sau.
- Module nào muốn thử tiếp: Module 5 (Enrichment) với các kỹ thuật nâng cao hơn như HyQA.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 5 |
| Code quality | 4 |
| Teamwork | 5 |
| Problem solving | 5 |
