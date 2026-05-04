# Failure Analysis — Lab 18: Production RAG

**Nhóm:** Nhóm Production  
**Thành viên:** Lê Kim Dung · Nguyễn Văn A · Trần Thị B · Phạm Văn C

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 1.0000 | 0.7500 | -0.2500 |
| Answer Relevancy | 0.7672 | 0.8158 | +0.0486 |
| Context Precision | 1.0000 | 1.0000 | +0.0000 |
| Context Recall | 1.0000 | 1.0000 | +0.0000 |

## Bottom-5 Failures

### #1
- **Question:** Nhân viên được nghỉ phép năm bao nhiêu ngày?
- **Expected:** 12 ngày làm việc mỗi năm.
- **Got:** Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác.
- **Worst metric:** faithfulness (0.5)
- **Error Tree:** Output sai → Context đúng? Có → Query OK? Có →
- **Root cause:** LLM hallucinating (hoặc đưa ra thông tin dư thừa dẫn đến điểm faithfulness thấp do không có trong context được trích dẫn cụ thể).
- **Suggested fix:** Tighten prompt, lower temperature.

### #2
- **Question:** Thâm niên bao nhiêu năm thì được tăng thêm ngày nghỉ phép?
- **Expected:** 5 năm thâm niên.
- **Got:** Thâm niên 5 năm thì được tăng thêm 1 ngày nghỉ phép mỗi năm.
- **Worst metric:** answer_relevancy (0.78)
- **Error Tree:** Output đúng? Có → Context đúng? Có → Query OK? Có →
- **Root cause:** Answer doesn't match question perfectly in wording (mặc dù nội dung đúng).
- **Suggested fix:** Improve prompt template to emphasize direct answers.

## Case Study (cho presentation)

**Question chọn phân tích:** Nhân viên được nghỉ phép năm bao nhiêu ngày?

**Error Tree walkthrough:**
1. Output đúng? → Gần đúng nhưng có thêm thông tin không cần thiết.
2. Context đúng? → Có, context chứa đầy đủ thông tin.
3. Query rewrite OK? → OK.
4. Fix ở bước: LLM Generation (Prompt Engineering).

**Nếu có thêm 1 giờ, sẽ optimize:**
- Tinh chỉnh prompt để LLM chỉ trả lời đúng trọng tâm câu hỏi.
- Thử nghiệm với các tham số decoding (temperature, top_p) để giảm hallucination.
- Cải thiện Enrichment module để cung cấp context sạch hơn.
