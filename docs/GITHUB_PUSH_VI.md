# Đưa thư mục này lên GitHub

Thư mục `Paper2_GitHub` đã tách riêng khỏi dữ liệu gốc. Chưa tạo repository online hoặc push từ phiên làm việc này.

## Cách dùng Git

1. Đăng nhập GitHub và tạo repository mới, ví dụ `paper2-llm-vnhsge`. Chọn **Public** để mọi người xem được. Tạo repo trống, không thêm README hay `.gitignore` vì chúng đã có trong thư mục.
2. Mở PowerShell hoặc terminal tại thư mục `Paper2_GitHub` và chạy:

```powershell
git init
git add .
git status --short
git commit -m "Publish Paper 2 code, aggregate results and APA 7 manuscript"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Thay URL bằng URL repository thật. GitHub sẽ yêu cầu đăng nhập nếu máy chưa được cấu hình. Không đặt token trong lệnh hay trong file mã nguồn. Sau khi push, mở trang repo và kiểm tra README, thư mục `code` và PDF.

Có thể dùng GitHub Desktop: thêm thư mục này thành local repository, commit, chọn **Publish repository**, tắt **Keep this code private** để công khai.

Trước khi nộp paper, cập nhật URL repository thật vào mục Data and code availability của bản thảo. Hiện chưa có tên tác giả và licence xác minh, nên gói không tự tạo thông tin tác giả hoặc áp đặt licence. Đọc `REUSE.md` khi bổ sung các thông tin đó.
