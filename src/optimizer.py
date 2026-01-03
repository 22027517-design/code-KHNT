from ortools.sat.python import cp_model

class Section:
    def __init__(self, id, course_id, term, day, start, end, credits):
        self.id = id
        self.course_id = course_id
        self.term = term
        self.day = day
        self.start = start
        self.end = end
        self.credits = credits

    def __repr__(self):
        return f"Section({self.id}, {self.day} {self.start}-{self.end})"

def schedule_multi_term(sections, prereqs, target_terms, credit_bounds, risk_dict=None, risk_weight=5.0):
    """
    Hàm xếp lịch học tối ưu sử dụng Google OR-Tools (CP-SAT).
    Đã sửa lỗi kiểu dữ liệu int() vs FloatAffine.
    """
    model = cp_model.CpModel()

    # 1. Biến quyết định: Có chọn lớp s hay không? (0 hoặc 1)
    x = {}
    for s in sections:
        x[s.id] = model.NewBoolVar(f'x_{s.id}')

    # 2. Ràng buộc: Mỗi môn học chỉ được chọn tối đa 1 lớp
    course_to_sections = {}
    for s in sections:
        if s.course_id not in course_to_sections:
            course_to_sections[s.course_id] = []
        course_to_sections[s.course_id].append(s)

    for c_id, secs in course_to_sections.items():
        chosen_var = model.NewBoolVar(f'chosen_{c_id}')
        model.Add(sum(x[s.id] for s in secs) == chosen_var)

    # 3. Ràng buộc: Không trùng giờ học
    term_day_sections = {}
    for s in sections:
        key = (s.term, s.day)
        if key not in term_day_sections:
            term_day_sections[key] = []
        term_day_sections[key].append(s)

    for key, secs in term_day_sections.items():
        for i in range(len(secs)):
            for j in range(i + 1, len(secs)):
                s1 = secs[i]
                s2 = secs[j]
                if max(s1.start, s2.start) <= min(s1.end, s2.end):
                    model.Add(x[s1.id] + x[s2.id] <= 1)

    # 4. Ràng buộc: Số tín chỉ (Min - Max)
    for t in target_terms:
        s_in_term = [s for s in sections if s.term == t]
        if not s_in_term:
            continue
        
        total_credits = sum(x[s.id] * s.credits for s in s_in_term)
        
        if t in credit_bounds:
            min_c, max_c = credit_bounds[t]
            model.Add(total_credits >= min_c)
            model.Add(total_credits <= max_c)

    # 5. Hàm mục tiêu (Objective Function) - ĐÃ SỬA LỖI
    # Tính tổng tín chỉ muốn đạt được
    objective_credits = sum(x[s.id] * s.credits for s in sections)
    
    # Tính điểm phạt rủi ro
    risk_penalty = 0
    if risk_dict:
        for s in sections:
            # Lấy rủi ro (vd: 0.8), nhân 100 -> thành số nguyên (80)
            r = risk_dict.get(s.course_id, 0.0)
            r_int = int(r * 100) 
            # Cộng dồn vào biểu thức: biến x * số nguyên r_int
            risk_penalty += x[s.id] * r_int

    # CHUẨN BỊ TRỌNG SỐ (Ép kiểu ra số nguyên ở ngoài)
    w_int = int(risk_weight) 

    # THIẾT LẬP HÀM MỤC TIÊU
    # Tuyệt đối không dùng int() bao trùm cả biểu thức chứa biến x
    # Công thức: (Tín chỉ * 10) - (Rủi ro * Trọng số)
    model.Maximize((objective_credits * 10) - (risk_penalty * w_int))

    # 6. Giải
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        selected_sections = [s for s in sections if solver.Value(x[s.id]) == 1]
        return selected_sections, "Optimal"
    else:
        return [], "Infeasible"