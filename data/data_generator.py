import pandas as pd
import numpy as np
import random

def generate_dummy_data():
    # 1. Danh sách môn học
    courses = pd.DataFrame([
        {'id': 'MATH1', 'credits': 3, 'difficulty': 0.8},
        {'id': 'MATH2', 'credits': 3, 'difficulty': 0.9},
        {'id': 'PHYS1', 'credits': 4, 'difficulty': 0.7},
        {'id': 'PROG1', 'credits': 3, 'difficulty': 0.6},
        {'id': 'PROG2', 'credits': 4, 'difficulty': 0.8},
        {'id': 'DSA',   'credits': 4, 'difficulty': 0.9},
        {'id': 'DB',    'credits': 3, 'difficulty': 0.5},
        {'id': 'AI',    'credits': 4, 'difficulty': 0.95},
        {'id': 'CAP1',  'credits': 5, 'difficulty': 0.85},
    ])
    
    # 2. Môn tiên quyết
    prereqs = pd.DataFrame([
        {'course': 'MATH2', 'prereq': 'MATH1'},
        {'course': 'PROG2', 'prereq': 'PROG1'},
        {'course': 'DSA',   'prereq': 'PROG2'},
        {'course': 'DSA',   'prereq': 'MATH2'},
        {'course': 'AI',    'prereq': 'DSA'},
        {'course': 'CAP1',  'prereq': 'AI'},
    ])

    # 3. Lịch sử học tập (Training Data cho AI)
    history_data = []
    for _ in range(500): # Sinh 500 mẫu dữ liệu
        c = courses.sample(1).iloc[0]
        stud_gpa = np.random.uniform(1.5, 4.0) 
        # Logic: GPA cao + Môn dễ = Tỉ lệ qua cao
        pass_prob = (stud_gpa / 4.0) * 0.7 + (1 - c['difficulty']) * 0.3
        is_passed = 1 if random.random() < pass_prob else 0
        
        history_data.append({
            'student_gpa_avg': stud_gpa,
            'course_difficulty': c['difficulty'],
            'course_credits': c['credits'],
            'passed': is_passed
        })
    history_df = pd.DataFrame(history_data)

    # 4. Các lớp học phần mở trong 3 kỳ tới
    sections_data = []
    terms = [1, 2, 3] 
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    
    cnt = 0
    for t in terms:
        for _, row in courses.iterrows():
            # Mỗi môn mở 2 lớp mỗi kỳ
            for _ in range(2):
                cnt += 1
                start_slot = random.choice([1, 4, 7, 10])
                sections_data.append({
                    'id': f"SEC_{cnt}",
                    'course_id': row['id'],
                    'term': t,
                    'day': random.choice(days),
                    'start': start_slot,
                    'end': start_slot + int(row['credits']) - 1,
                    'credits': row['credits']
                })

    sections_df = pd.DataFrame(sections_data)
    
    return courses, prereqs, history_df, sections_df