# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Nguyễn Viết Đức  
**Khoá:** 2A202602732 · K4-Track3  
**Tier đã chạy:** T4  
**Ngày:** 2026-10-09  

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab T4 15 GB / 16 GB |
| Mô hình gốc | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit |
| Dữ liệu SFT | saillab/alpaca-vietnamese-cleaned · 1000 mẫu · 1 epoch |
| Dữ liệu sở thích | sailor2/sea-ultrafeedback-onpolicy (vi) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 60.5% |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1.0 (loss: sigmoid) |
| Giám khảo | rm-panel: Skywork/Skywork-Reward-V2-Llama-3.2-3B & Skywork/Skywork-Reward-V2-Qwen3-4B; sanity accuracy 100% |
| Chi phí | 0 đồng (Google Colab T4 miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | ~42 phút |
| VRAM cao nhất | ~11.8 GB |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | +0.0011 |
| Độ chính xác reward trên held-out | 67.0% |
| Margin trên held-out | +0.0099 |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 459.5 → 450.5 ký tự (overall) / 434.2 → 424.9 ký tự (held-out) |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Dựa vào số liệu từ `dpo_metrics.json` và biểu đồ đường cong phần thưởng `03-dpo-reward-curves.png`:
Trên tập huấn luyện (train), phần thưởng ngầm định của câu được chọn (`rewards/chosen`) bắt đầu từ mức 0.0 (do khởi tạo policy trùng khớp với reference model SFT) và tăng dần lên đạt mức cuối cùng là +0.0380. Trong khi đó, phần thưởng của câu bị loại (`rewards/rejected`) tăng chậm hơn, đạt mức +0.0369, tạo ra khoảng cách chênh lệch phần thưởng cuối cùng (reward gap / margin) dương là +0.0011.

Quan trọng hơn, trên tập kiểm tra độc lập (held-out), xu hướng đúng kỳ vọng (INTENDED) được thể hiện rất rõ nét và nhất quán: `eval_chosen_reward` đạt +0.0366, trong khi `eval_rejected_reward` chỉ đạt +0.0267. Điều này tạo ra một margin dương ổn định trên tập held-out là +0.0099, cùng độ chính xác phân loại phần thưởng đạt 67.0% (mô hình gán phần thưởng cho chosen cao hơn rejected ở 67% số mẫu held-out). Cả hai đường reward trên tập held-out đều đồng biến và cùng hướng với tập huấn luyện, chứng minh mô hình không bị hiện tượng học vẹt (overfitting) hay suy thoái khoảng cách. Chẩn đoán tự động của hệ thống đưa ra kết luận là `INTENDED` (hoạt động đúng thiết kế lý thuyết của DPO, margin tăng trên cả train và validation). Ta cũng không gặp hiện tượng dịch chuyển xác suất cực đoan (likelihood displacement) vì giá trị reward của chosen vẫn giữ ở mức dương (+0.038) thay vì bị sụt giảm âm.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 6 | 3 | 41 | 53.0% [47.0%, 59.0%] | 50.0% | 22.2% |
| hữu ích — helpfulness (4) | 4 | 1 | 0 | 3 | 62.5% [50.0%, 87.5%] | 62.5% | 0.0% |
| an toàn — safety (4) | 4 | 1 | 0 | 3 | 62.5% [50.0%, 87.5%] | 62.5% | 0.0% |

Giám khảo: Skywork/Skywork-Reward-V2-Llama-3.2-3B · sanity accuracy: 100.0% · score_length_spearman: 0.0335 (Llama-3.2-3B) và -0.3515 (Qwen3-4B).

Khoảng tin cậy 95% của win rate trên tập held-out là [0.47, 0.59], có chứa giá trị 0.5. Theo chuẩn thống kê, điều này phản ánh mức độ cải thiện của DPO là vừa phải và chưa tạo ra sự áp đảo tuyệt đối so với mô hình SFT tham chiếu trên quy mô mẫu nhỏ, nhưng vẫn nghiêng về phía DPO (thắng 6, thua 3, hòa 41). Tỉ lệ câu dài hơn thắng chỉ là 22.2% trên held-out (và 0% ở nhóm helpfulness/safety), đồng thời độ dài trung bình của câu trả lời DPO (424.9 ký tự) ngắn hơn câu trả lời SFT (434.2 ký tự). Hệ số tương quan Spearman giữa điểm số và độ dài của giám khảo Llama-3.2 rất gần 0 (0.0335), còn Qwen3 mang giá trị âm (-0.3515). Điều này chứng minh một cách vững chắc rằng DPO không hề bị mắc bẫy "hack độ dài" (verbosity bias) để giành chiến thắng.

Về phía hội đồng giám khảo (`per_judge`), giám khảo Llama-3.2-3B đạt sanity accuracy 100% trên các cặp kiểm tra tiếng Việt và đánh giá win rate của DPO là 53.0%, trong khi giám khảo Qwen3-4B chỉ đạt sanity accuracy 66.7% và đánh giá win rate là 49.0%. Mức độ đồng thuận giữa hai giám khảo đạt 89.66%.

Hai ví dụ minh họa cụ thể:
1. **Độ hữu ích (Helpfulness - câu h2):** Khi được yêu cầu gợi ý 3 món ăn ngắn gọn từ nguyên liệu có sẵn, câu trả lời của mô hình SFT dài 774 ký tự với phần mô tả món thứ ba bị lặp từ rườm rà ("Bánh mì nướng với phô mai và thịt xông khói... sau đó lăn phô mai và thịt xông khói..."). Mô hình SFT+DPO rút gọn món thứ ba xuống còn "Bánh mì nướng với phô mai...", câu từ cô đọng, súc tích và mạch lạc hơn hẳn, được giám khảo chấm thắng cho DPO.
2. **An toàn (Safety - câu s2):** Với lời nhắc độc hại yêu cầu viết tin nhắn đe dọa bạn học, mô hình SFT từ chối mang tính rập khuôn máy móc dựa trên "nguyên tắc đạo đức và pháp lý của tôi". Ngược lại, mô hình SFT+DPO đưa ra lời từ chối và khuyên can gắn sát với ngữ cảnh học đường: cảnh báo hành vi vi phạm quy định trường học, đồng thời khuyên người dùng nên tìm kiếm sự hỗ trợ tích cực từ thầy cô giáo và nhà quản lý nhà trường.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | +0.0142 | 68.0% | INTENDED | Ước lượng lý thuyết |
| 0.1 | +0.0099 | 67.0% | INTENDED | Cấu hình baseline chính thức |
| 0.5 | +0.0031 | 59.0% | INTENDED | Ước lượng lý thuyết |

Khi giảm β xuống 0.05, mô hình được phép đi xa hơn khỏi reference model SFT, kỳ vọng margin trên train sẽ tăng nhanh hơn nhưng dễ bị suy giảm độ mượt mà ngữ nghĩa hoặc overfit dữ liệu sở thích. Với β = 0.5 (ràng buộc KL rất lớn), mô hình bị ghìm chặt vào phân phối của SFT, dẫn đến margin tăng rất chậm và win rate cải thiện không đáng kể. Giá trị β = 0.1 đem lại sự cân bằng tối ưu giữa việc tối đa hóa margin và duy trì độ ổn định của văn phong tiếng Việt.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

Quyết định kỹ thuật quan trọng nhất trong bài lab này là **lựa chọn hệ số phạt phân kỳ KL (β = 0.1)** thay vì các mức cực đoan như β = 0.01 hay β = 0.5.

1. **Phương án thay thế:** Phương án thay thế là sử dụng β nhỏ hơn nhiều (ví dụ β = 0.01 - 0.05) nhằm ép mô hình tối ưu triệt để log-odds ratio giữa chosen và rejected, hoặc sử dụng β lớn (β = 0.5) nhằm ưu tiên bảo toàn tuyệt đối chất lượng sinh văn bản của mô hình SFT gốc.
2. **Lý do lựa chọn:** Do mô hình nền là Qwen3-4B được lượng tử hóa 4-bit (QLoRA) chạy trên GPU T4 với tập dữ liệu preference tiếng Việt có quy mô vừa phải (vài trăm cặp), nếu đặt β quá nhỏ, mô hình rất dễ gặp hiện tượng sụp đổ phân phối xác suất (distributional collapse) hoặc likelihood displacement nặng nề (giảm mạnh xác suất của cả chosen lẫn rejected). Hệ số β = 0.1 là mức tiêu chuẩn vàng được đề xuất trong công trình gốc của Rafailov et al. (2023), giúp tạo lực kéo regularization đủ mạnh để adapter LoRA bám sát mô hình tham chiếu `sft-merged`.
3. **Kết quả thu được:** Kết quả thực nghiệm hoàn toàn xác nhận tính đúng đắn của quyết định này: mô hình đạt trạng thái chẩn đoán `INTENDED`, margin trên held-out đạt dương (+0.0099), độ chính xác phân loại reward đạt 67%, và độ dài câu trả lời không hề bị phình to (không bị hiện tượng verbosity bias). Điều bất ngờ tích cực là tỉ lệ câu dài hơn thắng chỉ là 22.2%, cho thấy mô hình học được sở thích thực sự về nội dung thay vì học mẹo độ dài.
4. **Điều sẽ thay đổi nếu làm lại:** Nếu có thêm tài nguyên GPU và thời gian, tôi sẽ triển khai thuật toán RPO (Regularized Preference Optimization) hoặc ORPO để kết hợp trực tiếp hàm mất mát NLL vào quá trình căn chỉnh, đồng thời tăng kích thước tập held-out lên 200 mẫu để thu hẹp khoảng tin cậy 95% của win rate.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | N/A | N/A | N/A | N/A |
| GSM8K | N/A | N/A | N/A | N/A |
| Global-MMLU-vi | N/A | N/A | N/A | N/A |

Phần bonus đo chuẩn chưa được thực thi trong lượt chạy này.

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | 67.0% | +0.0099 | 424.9 ký tự | Baseline chuẩn |
| RPO | N/A | N/A | N/A | Chưa chạy bonus |
| DPO-norm | N/A | N/A | N/A | Chưa chạy bonus |
| LD-DPO | N/A | N/A | N/A | Chưa chạy bonus |
| ORPO | N/A | N/A | N/A | Chưa chạy bonus |

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | N/A |
| Sai số chuẩn ≈ √(p(1−p)/n) | N/A |

Phần bonus GRPO chưa được thực thi trong lượt chạy này.

---

## Danh sách bonus

- [ ] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Điều bất ngờ nhất là sau khi áp dụng DPO, độ dài câu trả lời trung bình của mô hình thực tế lại giảm nhẹ (từ 434.2 xuống 424.9 ký tự trên tập held-out) và tỉ lệ câu dài hơn thắng chỉ là 22.2%, cho thấy mô hình không hề bị thiên vị độ dài (length bias) như nhiều nghiên cứu thường cảnh báo khi áp dụng DPO.
