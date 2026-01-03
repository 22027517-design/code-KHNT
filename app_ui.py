import streamlit as st
import pandas as pd
import numpy as np
import math

# --- IMPORT MODULE BACKEND ---
from data.data_generator import generate_dummy_data
from src.ai_model import train_risk_model, infer_risk
from src.optimizer import schedule_multi_term, Section
from ortools.sat.python import cp_model 

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(page_title="Hệ Thống Cố Vấn Học Tập", page_icon="🎓", layout="wide")

# ==============================================================================
# PHẦN 1: DỮ LIỆU & MODEL (BACKEND)
# ==============================================================================
def get_university_schedule():
    """ 
    DATABASE CÁC LỚP HỌC PHẦN (Đã cập nhật theo yêu cầu của bạn)
    Tôi đã phân bổ thời gian (Thứ/Tiết) để tránh trùng lịch.
    """
    schedule_db = [
        # Thứ 2
        {"id": "NNPL_01", "course_id": "NNPL", "name": "Nhà nước và pháp luật đại cương", "day": "Mon", "start": 1, "credits": 3},
        {"id": "KTDK_01", "course_id": "KTDK", "name": "Kỹ thuật điều khiển",             "day": "Mon", "start": 7, "credits": 3},
        
        # Thứ 3
        {"id": "KTVM_01", "course_id": "KTVM", "name": "Kinh tế vi mô",                   "day": "Tue", "start": 1, "credits": 3},
        {"id": "THHT_01", "course_id": "THHT", "name": "Tín hiệu và hệ thống",            "day": "Tue", "start": 7, "credits": 3},
        
        # Thứ 4
        {"id": "TTHCM_01","course_id": "TTHCM","name": "Tư tưởng Hồ Chí Minh",            "day": "Wed", "start": 1, "credits": 2},
        {"id": "CNXHKH_01","course_id":"CNXHKH","name":"Chủ nghĩa xã hội khoa học",       "day": "Wed", "start": 7, "credits": 2},
        
        # Thứ 5
        {"id": "CHHNV_01","course_id": "CHHNV","name": "Cơ học hệ nhiều vật",             "day": "Thu", "start": 1, "credits": 2},
        {"id": "MKD_01",  "course_id": "MKD",  "name": "Mạng không dây",                  "day": "Thu", "start": 7, "credits": 3},
    ]
    
    sections = []
    for s in schedule_db:
        sections.append(Section(
            id=s["id"], course_id=s["course_id"], term=1, 
            day=s["day"], start=s["start"], end=s["start"] + s["credits"] - 1, credits=s["credits"]
        ))
    return sections

@st.cache_resource
def init_ai_model():
    data = generate_dummy_data()
    if len(data) == 5: _, _, _, history_df, _ = data
    else: _, _, history_df, _ = data
    X = history_df[['student_gpa_avg', 'course_difficulty', 'course_credits']].values.astype('float32')
    y = history_df[['passed']].values.astype('float32')
    return train_risk_model(X, y, in_dim=3)

# ==============================================================================
# PHẦN 2: GIAO DIỆN CHÍNH
# ==============================================================================

def main():
    st.title("🎓 Hệ Thống Cố Vấn Học Tập Thông Minh")
    model = init_ai_model()

    tab1, tab2 = st.tabs(["📅 CHỨC NĂNG 1: Xếp Lịch Kỳ Tới", "🚀 CHỨC NĂNG 2: Dự Báo Tốt Nghiệp"])

    # --- TAB 1: XẾP LỊCH ---
    with tab1:
        st.header("🛠️ Xếp Thời Khóa Biểu Tự Động")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("##### 1. Danh sách môn bạn muốn đăng ký")
            # Cập nhật bảng mặc định theo dữ liệu bạn gửi
            default_wants = pd.DataFrame([
                {"Mã môn": "NNPL",   "Tên môn": "Nhà nước & PL đại cương", "Tín chỉ": 3, "Độ khó": 0.2},
                {"Mã môn": "KTDK",   "Tên môn": "Kỹ thuật điều khiển",     "Tín chỉ": 3, "Độ khó": 0.7},
                {"Mã môn": "KTVM",   "Tên môn": "Kinh tế vi mô",           "Tín chỉ": 3, "Độ khó": 0.3},
                {"Mã môn": "THHT",   "Tên môn": "Tín hiệu và hệ thống",    "Tín chỉ": 3, "Độ khó": 0.7},
                {"Mã môn": "TTHCM",  "Tên môn": "Tư tưởng Hồ Chí Minh",    "Tín chỉ": 2, "Độ khó": 0.2},
                {"Mã môn": "CNXHKH", "Tên môn": "CN xã hội khoa học",      "Tín chỉ": 2, "Độ khó": 0.2},
                {"Mã môn": "CHHNV",  "Tên môn": "Cơ học hệ nhiều vật",     "Tín chỉ": 2, "Độ khó": 0.7},
                {"Mã môn": "MKD",    "Tên môn": "Mạng không dây",          "Tín chỉ": 3, "Độ khó": 0.8},
            ])
            wants_df = st.data_editor(default_wants, num_rows="dynamic", use_container_width=True, key="tab1_editor")
        
        with col2:
            st.markdown("##### 2. Cấu hình")
            gpa_input = st.number_input("GPA hiện tại:", 0.0, 4.0, 2.5)
            # Tăng giới hạn Min/Max lên vì danh sách bạn gửi tổng cộng khoảng 21 tín
            min_cre = st.number_input("Tín chỉ Min:", 0, 30, 10)
            max_cre = st.number_input("Tín chỉ Max:", 0, 40, 25)

        if st.button("🚀 Xếp Lịch Học Tối Ưu", type="primary"):
            school_schedule = get_university_schedule()
            wanted_ids = wants_df['Mã môn'].unique()
            
            # Lọc các lớp có trong danh sách muốn học
            candidate_sections = [s for s in school_schedule if s.course_id in wanted_ids]
            
            if not candidate_sections:
                st.error("⚠️ Không tìm thấy lớp học phần phù hợp (Kiểm tra mã môn).")
            else:
                course_risks = {}
                for _, row in wants_df.iterrows():
                    cid = row['Mã môn']
                    # Dự báo rủi ro dựa trên độ khó bạn cung cấp
                    risk = 1.0 - infer_risk(model, np.array([[gpa_input, row.get('Độ khó',0.5), row.get('Tín chỉ',3)]], dtype='float32'))[0]
                    course_risks[cid] = risk

                # Chạy thuật toán xếp lịch
                chosen, status = schedule_multi_term(candidate_sections, {}, [1], {1: (min_cre, max_cre)}, course_risks)

                if chosen:
                    st.success(f"✅ Đã xếp xong! Tổng tín chỉ: {sum(s.credits for s in chosen)}")
                    results = []
                    # Mapping thứ sang tiếng Việt
                    day_map = {'Mon': 'Thứ 2', 'Tue': 'Thứ 3', 'Wed': 'Thứ 4', 'Thu': 'Thứ 5', 'Fri': 'Thứ 6'}
                    for s in chosen:
                        results.append({
                            "Thứ": day_map.get(s.day, s.day), 
                            "Ca": f"Tiết {s.start}-{s.end}", 
                            "Mã Môn": s.course_id, 
                            "Tên Môn": next((r['Tên môn'] for _, r in wants_df.iterrows() if r['Mã môn'] == s.course_id), s.course_id),
                            "Tín chỉ": s.credits, 
                            "Rủi ro trượt": f"{course_risks[s.course_id]:.1%}"
                        })
                    st.table(pd.DataFrame(results))
                else:
                    st.warning("⚠️ Không xếp được lịch. Có thể do tổng tín chỉ các môn vượt quá 'Tín chỉ Max' hoặc bị trùng giờ.")

    # --- TAB 2: DỰ BÁO TỐT NGHIỆP ---
    with tab2:
        st.markdown("## 📊 Phân Tích & Kế Hoạch Tốt Nghiệp")
        
        with st.expander("🔻 Nhập dữ liệu bảng điểm (Nhấn để mở)", expanded=True):
            c_input, c_param = st.columns([3, 1])
            with c_input:
                if 'history_data' not in st.session_state:
                    # Dữ liệu mẫu ban đầu
                    st.session_state['history_data'] = pd.DataFrame([
                        {"Học kỳ": 1, "Tên môn": "Giải tích 1", "Tín chỉ": 3, "Điểm GPA": 3.5},
                        {"Học kỳ": 1, "Tên môn": "Đại số", "Tín chỉ": 3, "Điểm GPA": 2.0},
                        {"Học kỳ": 2, "Tên môn": "Triết học", "Tín chỉ": 2, "Điểm GPA": 0.0},
                    ])

                history_df = st.data_editor(
                    st.session_state['history_data'],
                    num_rows="dynamic", use_container_width=True,
                    column_config={
                        "Học kỳ": st.column_config.NumberColumn("Học kỳ", format="%d"),
                        "Điểm GPA": st.column_config.NumberColumn("Điểm (hệ 4)", format="%.1f")
                    },
                    key="user_history_input"
                )
            with c_param:
                req_credits = st.number_input("Tổng tín chỉ cần tốt nghiệp:", value=150)
                limit_credits = st.number_input("Giới hạn tín chỉ/kỳ:", value=20)

        # --- XỬ LÝ DỰ BÁO ---
        if not history_df.empty:
            valid_df = history_df.dropna(subset=["Học kỳ", "Tín chỉ", "Điểm GPA"])
            
            if len(valid_df) > 0:
                total_cre_learned = 0
                total_points = 0
                total_cre_attempted = 0
                failed_cre = 0
                current_max_sem = int(valid_df["Học kỳ"].max())

                for _, row in valid_df.iterrows():
                    c = float(row['Tín chỉ'])
                    g = float(row['Điểm GPA'])
                    total_points += c * g
                    total_cre_attempted += c
                    if g < 1.0: 
                        failed_cre += c
                    else:
                        total_cre_learned += c
                
                gpa_avg = total_points / total_cre_attempted if total_cre_attempted > 0 else 0.0
                missing_cre = max(0, req_credits - total_cre_learned)
                
                total_needed = missing_cre + failed_cre
                semesters_needed = math.ceil(total_needed / limit_credits) if limit_credits > 0 else 99
                grad_sem = current_max_sem + semesters_needed

                st.divider()
                
                # Metrics
                st.markdown("### Kết quả phân tích hiện tại:")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Tín chỉ tích lũy", f"{int(total_cre_learned)}/{req_credits}")
                m2.metric("GPA Trung bình", f"{gpa_avg:.2f}")
                m3.metric("Số tín chỉ còn thiếu", f"{int(missing_cre)}")
                m4.metric("Số tín chỉ TRƯỢT (Nợ)", f"{int(failed_cre)}", delta_color="inverse")
                
                st.markdown("---")

                # Text Dự báo
                st.markdown(f"## 🔮 Dự báo: Bạn cần thêm khoảng {semesters_needed} kỳ nữa.")
                st.caption(f"Dự kiến tốt nghiệp vào: **Học kỳ thứ {grad_sem}**")

                # Chiến lược Box
                st.markdown("### 💡 AI Đề Xuất Chiến Lược:")

                if failed_cre > 0:
                    st.error(f"⚠️ CẢNH BÁO: Bạn đang nợ {int(failed_cre)} tín chỉ môn trượt! Điều này sẽ làm chậm tiến độ tốt nghiệp.")
                
                col_strat1, col_strat2 = st.columns(2)
                
                with col_strat1:
                    st.info("**1. Kế hoạch Trả nợ môn**")
                    if failed_cre > 0:
                        st.markdown(f"* **Ưu tiên SỐ 1:** Đăng ký học lại toàn bộ **{int(failed_cre)} tín** chỉ nợ trong kỳ tới.")
                        st.markdown(f"* Nếu môn nợ không mở kỳ tới, hãy tìm môn tương đương thay thế ngay lập tức.")
                        st.markdown(f"* Không đăng ký môn mới khó nếu chưa trả xong nợ môn cũ.")
                    else:
                        st.markdown("* Tuyệt vời! Bạn không nợ môn nào.")
                        st.markdown("* Hãy tập trung duy trì GPA cao.")

                with col_strat2:
                    st.info("**2. Chiến lược Tăng tốc (Học vượt)**")
                    avg_needed = total_needed / semesters_needed if semesters_needed > 0 else 0
                    st.markdown(f"* Để ra trường đúng hạn (trong {semesters_needed} kỳ tới), bạn phải đăng ký trung bình **{math.ceil(avg_needed)} tín/kỳ**.")
                    st.markdown(f"* **Học kỳ Hè:** Hãy tận dụng kỳ hè để học các môn đại cương/tự chọn (khoảng 6-9 tín) để giảm tải cho kỳ chính.")
                    if gpa_avg > 2.5:
                        st.markdown(f"* Nếu GPA > 2.5, hãy mạnh dạn đăng ký chạm trần ({limit_credits} tín) để rút ngắn thời gian.")
            else:
                st.warning("Vui lòng nhập đầy đủ thông tin (Tín chỉ và GPA không được để trống).")

if __name__ == "__main__":
    main()