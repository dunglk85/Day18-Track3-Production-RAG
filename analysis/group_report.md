# Group Report — Lab 18: Production RAG

**Nhóm:** Nhóm Production  
**Ngày:** 2024-05-04

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------|
| Lê Kim Dung | M1: Chunking + Pipeline | ☑ | 8/8 |
| Nguyễn Văn A | M2: Hybrid Search | ☑ | 5/5 |
| Trần Thị B | M3: Reranking | ☑ | 5/5 |
| Phạm Văn C | M4: Evaluation | ☑ | 4/4 |

## Kết quả RAGAS

| Metric | Naive | Production | Δ |
|--------|-------|-----------|---|
| Faithfulness | 1.0000 | 0.7500 | -0.2500 |
| Answer Relevancy | 0.7672 | 0.8158 | +0.0486 |
| Context Precision | 1.0000 | 1.0000 | +0.0000 |
| Context Recall | 1.0000 | 1.0000 | +0.0000 |

## Key Findings

1. **Biggest improvement:** Answer Relevancy (+0.0486) nhờ việc sử dụng Hybrid Search và Reranking giúp chọn lọc context liên quan tốt hơn.
2. **Biggest challenge:** Tích hợp các module cá nhân vào một pipeline thống nhất và xử lý các lỗi encoding/unhashable types trên môi trường Windows.
3. **Surprise finding:** Faithfulness của Production lại thấp hơn Naive Baseline. Có thể do Enrichment làm giàu text quá mức dẫn đến LLM đưa ra thông tin không có trong context gốc.

## Presentation Notes (5 phút)

1. RAGAS scores (naive vs production): Production cải thiện Relevancy nhưng giảm Faithfulness.
2. Biggest win — module nào, tại sao: Module 2 & 3 (Hybrid Search & Reranking) giúp trích xuất context chính xác hơn.
3. Case study — 1 failure, Error Tree walkthrough: Câu hỏi về ngày nghỉ phép năm bị giảm Faithfulness do LLM trả lời thừa thông tin.
4. Next optimization nếu có thêm 1 giờ: Tối ưu prompt generation và thử nghiệm các phương pháp enrichment khác như HyQA.
