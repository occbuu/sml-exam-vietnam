# Sửa theo báo cáo phản biện — cập nhật 19/10/2026

Bản mã công khai được chỉnh cho khớp manuscript 19/10. Gradient thu nhập tỉnh được xuất cho đủ tám môn trắc nghiệm. Bảng độ chính xác S1 kèm khoảng tin cậy Wilson. Phụ lục bổ sung hồi quy đặc điểm câu và phân bố chữ cái phương án. Đoạn kết quả, mô tả RQ4, giới hạn prompt 8 token và bốn tài liệu mới (Messick, Dawson, Al Faraby, Wei) đi theo bản LaTeX đã sửa. Danh mục hiện 52 mục, tất cả được trích dẫn.

## Cập nhật 15/10/2026

## Xử lý ba điểm còn lại ở lượt sửa mới

1. **Background và danh mục:** thêm 18 tài liệu, đạt 48 mục và trích dẫn đủ 48; bổ sung Gaokao, CSAT, ENEM, an ninh khảo thí, đạo đức AI, validity và detector fairness. Dùng nguồn gốc, phân biệt preprint với bài xuất bản; metadata lưu trong `outputs/reference_metadata/`. Chi tiết: `LITERATURE_REVISION_AUDIT.md`.
2. **APA 7:** `main.tex` đã mặc định dùng biblatex-apa/Biber. Biber 2.22 chạy được ngoài sandbox trên chính máy này; lỗi DLL trước đây là hạn chế khi khởi chạy trong sandbox. `main.pdf` đã biên dịch APA 7 và kiểm tra bố cục. Bản APA 6 cũ được giữ riêng. Đã kiểm tra trích dẫn từ 3 tác giả dùng et al. ngay lần đầu, DOI dạng https://doi.org/ và quy tắc 19 tác giả đầu + tác giả cuối khi danh sách dài.
3. **S3:** xác minh README và mã collector công khai: Dân trí (2022) và VietNamNet (notebook 2021). Bổ sung đoạn Data với tham chiếu đúng Table A.1 bằng nhãn LaTeX tự động; tách rõ đối soát thống kê tổng hợp khỏi khẳng định tính đầy đủ và toàn vẹn từng bản ghi. Dataset dùng entry `@dataset`, bắt đầu bằng tiêu đề vì tên cá nhân chưa xác minh.

Các mục dưới đây mô tả sửa phương pháp từ lượt trước và những yêu cầu thực nghiệm vẫn còn thiếu.

## Những sửa đổi đã thực hiện

1. **Phạm vi và tiêu đề:** định vị thành benchmark các mô hình nhỏ chạy cục bộ, không kết luận năng lực frontier hay khả năng kháng AI của toàn bộ kỳ thi. Thay RQ2 và tiêu đề kết quả để mô tả liên hệ với lỗi dưới protocol đã thử.
2. **Giá trị đo lường:** bổ sung khung Kane (2013), phân biệt scoring, generalisation, extrapolation và decision/use. Phân vị là tham chiếu phân phối điểm, không chứng minh năng lực tiềm ẩn tương đương giữa người và mô hình. Bổ sung IRT, phân biệt guessing 25% với tham số c; không tuyên bố DIF khi chỉ có điểm tổng môn.
3. **Hình và Atlat:** ghi rõ đây là điều kiện thiếu thông tin. Không suy từ 28,9% sang khả năng kháng multimodal AI. Bổ sung MathVista với nguồn ICLR chính thức.
4. **Prompting và độ bất định:** làm rõ giới hạn 8 token, thiếu scratchpad, không thể dùng S1/S2 để ước lượng tác dụng CoT vì đồng thời đổi tập câu. Wilson CI mô tả độ bất định theo câu, không phải biến thiên giữa các lần giải mã.
5. **Proxy vị trí:** giữ mô tả 40/30/20/10 nhưng bỏ suy luận rằng hồi quy đã kiểm định Bloom hoặc độ khó. Dữ liệu S3 chỉ có điểm tổng, không thể tính p-value từng câu. Table Note giải thích thuật ngữ và hạn chế.
6. **Bao phủ thí sinh:** thêm Appendix D và `H1b_coverage_sensitivity.csv`, so sánh phân vị 2019–2023 với 2020–2023. Đây là phân tích loại năm, không khôi phục người bị thiếu hay tách ảnh hưởng độ khó đề. Không thay các ước lượng chính bằng kết quả này.
7. **Bảng và hình:** tái sinh từ CSV, tên mô hình học thuật, phân vị 1 chữ số thập phân; nhãn hình tiếng Anh; PDF vector cho cả năm hình; giữ viridis cho heatmap. LaTeX chỉ sự hiện diện markup, không đo độ phức tạp.
8. **Nguồn dữ liệu và danh mục:** thêm 3 nguồn có liên quan (Kane, Lord, MathVista), bổ sung URL hội nghị MMLU/MCQ robustness. Khi tác giả THPTQG chưa được xác minh, chuyển sang trích dẫn bắt đầu bằng tên dataset, giữ account trong provenance; không tự gán mirror cho MOET. Qwen và Gemma là tên được ghi trong technical reports, không tự thay bằng một tổ chức khác.
9. **Tệp tái lập:** sửa đường dẫn bộ sinh bảng/hình/văn bản và build script từ thư mục `latex` không tồn tại sang `Paper2_LLM_VNHSGE`. Các đoạn sinh tự động cũng giữ cách diễn giải đã sửa.
10. **Thông tin hành chính:** loại các tên/email/cơ quan giả định khỏi bản ẩn danh; ghi đúng trạng thái chưa có public deposit thay vì tuyên bố đã công bố code. Cần tác giả cung cấp title page thật trước khi nộp.

## Những yêu cầu chưa được đáp ứng bằng bằng chứng mới

- Chưa có frontier/70B hay multimodal benchmark. Bài được thu hẹp theo phương án thay thế reviewer đề xuất.
- Chưa chạy CoT cho cả ba mô hình trên cùng tập S1 hoặc tập con đại diện; chưa có 3–5 lần chạy. Không tạo số liệu giả cho các yêu cầu này.
- Chưa có teacher coding, Cohen/Fleiss kappa hoặc candidate item responses. Không thể kiểm định proxy bằng điểm tổng môn của 7,29 triệu bản ghi.
- Digest cũ không thể khôi phục chắc chắn từ tag hiện tại; hạn chế provenance vẫn được công khai.
- Giữ log 17 câu không parse và 62 identity permutations cùng phân tích 1.621 cặp đổi thật. Không sửa hồi tố input của các log đã chạy.
- Danh mục bản 15/10 có 48 mục. Bản 19/10 thêm 4 nguồn và đạt 52 mục đã trích dẫn. Chưa có một audit độc lập của toàn bộ nguồn cũ và mọi khẳng định trong bài.
- Đã thay `others` bằng danh sách tác giả từ metadata gốc cho Kasneci và ba technical reports; chưa thể bảo đảm trạng thái xuất bản cuối của mọi preprint trong danh mục.

## Trạng thái APA và biên dịch

`main.tex` và `main_apa7.tex` dùng biblatex-apa/Biber cho APA 7. LuaLaTeX + Biber 2.22 đã biên dịch thành công tại local ngoài sandbox. `main.pdf` là bản chính APA 7; `main_legacy_APA6.pdf` chỉ giữ để đối chiếu. Gói `Paper2_APA7_Overleaf.zip` chuẩn bị cho người dùng muốn biên dịch tiếp trên Overleaf, không tự tải tài liệu lên dịch vụ ngoài.

Đây là bản sửa có giới hạn thực nghiệm được công khai, chưa phải bản đáp ứng toàn bộ major revision và sẵn sàng nộp.
