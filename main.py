import numpy as np
from data.data_generator import generate_dummy_data
from src.ai_model import train_risk_model, infer_risk
from src.optimizer import schedule_multi_term, Section

def main():
    print("=== HỆ THỐNG TỐI ƯU HỌC TẬP & TỐT NGHIỆP SỚM ===")
    
    # 1. LOAD DATA
    print("[1/4] Đang khởi tạo dữ liệu giả lập...")
    courses_df, prereq_df, history_df, sections_df = generate_dummy_data()
    
    # 2. TRAIN AI
    print("[2/4] Đang huấn luyện mô hình dự đoán rủi ro (MLP)...")
    X = history_df[['student_gpa_avg', 'course_difficulty', 'course_credits']].values.astype('float32')
    y = history_df[['passed']].values.astype('float32')
    model = train_risk_model(X, y, in_dim=3)
    
    # 3. NHẬP THÔNG TIN SINH VIÊN HIỆN TẠI
    current_gpa = 2.8 # Sinh viên khá
    print(f"[3/4] Đang lập kế hoạch cho sinh viên có GPA: {current_gpa}")
    
    # Dự đoán rủi ro cho từng môn
    course_risks = {}
    for _, c in courses_df.iterrows():
        # Input: [GPA, Difficulty, Credits]
        inp = np.array([[current_gpa, c['difficulty'], c['credits']]], dtype='float32')
        fail_prob = 1.0 - infer_risk(model, inp)[0]
        course_risks[c['id']] = fail_prob
    
    # 4. CHẠY TỐI ƯU HÓA (CP-SAT)
    print("[4/4] Đang chạy thuật toán tối ưu xếp lịch...")
    
    # Chuẩn bị dữ liệu cho Solver
    sections = [
        Section(r['id'], r['course_id'], r['term'], r['day'], r['start'], r['end'], r['credits'])
        for _, r in sections_df.iterrows()
    ]
    
    prereqs = {}
    for _, r in prereq_df.iterrows():
        prereqs.setdefault(r['course'], []).append(r['prereq'])
        
    # Cấu hình mong muốn
    target_terms = [1, 2, 3] # Lập kế hoạch cho 3 kỳ tới
    credit_bounds = {
        1: (8, 30),  # Giảm xuống 8 để dễ tìm lịch hơn
        2: (8, 30),
        3: (5, 30)
    }
    
    chosen, status = schedule_multi_term(
        sections, prereqs, target_terms, credit_bounds, course_risks, risk_weight=10.0
    )
    
    # 5. IN KẾT QUẢ
    if chosen:
        print("\n✅ LỘ TRÌNH TỐI ƯU ĐƯỢC ĐỀ XUẤT:")
        chosen.sort(key=lambda x: (x.term, x.day, x.start))
        
        current_t = 0
        total_cre = 0
        for s in chosen:
            if s.term != current_t:
                current_t = s.term
                print(f"\n--- KỲ {current_t} ---")
            
            risk = course_risks.get(s.course_id, 0)
            risk_str = "CAO" if risk > 0.4 else "Thấp"
            print(f"  {s.day} (Tiết {s.start}-{s.end}) | {s.course_id:<5} | {s.credits} tín | Rủi ro: {risk:.1%} ({risk_str})")
            total_cre += s.credits
            
        print(f"\nTổng tín chỉ: {total_cre}")
        if total_cre >= 20: # Giả sử cần tích lũy thêm 20 tín để ra trường
            print("=> KẾT LUẬN: Bạn sẽ TỐT NGHIỆP SỚM sau lộ trình này.")
    else:
        print("❌ Không tìm thấy lịch. Hãy giảm bớt ràng buộc tín chỉ.")

if __name__ == "__main__":
    main()