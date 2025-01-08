import pandas as pd
import streamlit as st
import joblib

def load_model():
    try:
        model = joblib.load(r'C:\Users\Darren Eduardo\Desktop\Documents\Progres Projeck_AI_Kelompok 2\best_xgb_model.pkl')
        return model
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        return None

def main():
    st.title("Prediksi Depresi Seseorang Berdasarkan Survei")
    st.write("Isi data survei berikut untuk memprediksi kondisi Anda:")

    # Input features berdasarkan data yang diberikan
    name_input = st.text_input("Masukkan Nama Anda")
    name_numeric = hash(name_input) % 1000  # Mengubah nama ke angka numerik (3 digit)
    gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    city = st.selectbox("Kota Tinggal", ["Ghaziabad", "Kalyan", "Bhopal", "Thane", "Indore", "Pune", "Bangalore", "Hyderabad", "Srinagar", "Nashik", "Kolkata", "Ahmedabad", "Varanasi", "Chennai", "Jaipur", "Surat", "Vasai-Virar", "Rajkot", "Patna", "Mumbai", "Vadodara", "Lucknow", "Faridabad", "Meerut", "Kanpur", "Visakhapatnam", "Ludhiana", "Nagpur", "Agra", "Delhi"])
    work_status = st.selectbox("Apakah Anda seorang profesional atau pelajar?", ["Working Professional", "Student"])
    profession = st.selectbox("Profesi Anda", ["Student","Teacher", "Financial Analyst", "UX/UI Designer", "Civil Engineer", "Accountant", "Lawyer", "Content Writer", "Pilot", "Customer Support", "Judge", "Architect", "HR Manager", "Digital Marketer", "Sales Executive", "Business Analyst", "Mechanical Engineer", "Consultant", "Data Scientist", "Pharmacist", "Software Engineer", "Travel Consultant", "Manager", "Entrepreneur", "Doctor", "Researcher", "Plummer", "Educational Consultant", "Chemist", "Research Analyst", "Chef", "Electrician", "Graphic Designer", "Investment Banker", "Finanancial Analyst", "Marketing Manager", "Plumber"])
    sleep_duration = st.selectbox("Durasi Tidur", ["7-8 hours", "5-6 hours", "More than 8 hours", "Less than 5 hours"])
    dietary_habits = st.selectbox("Kebiasaan Diet", ["Moderate", "Unhealthy", "Healthy"])
    degree = st.selectbox("Gelar Pendidikan", ["MA", "B.Com", "M.Com", "MD", "BE", "MCA", "BA", "LLM", "BCA", "Class 12", "B.Ed", "M.Tech", "LLB", "B.Arch", "ME", "MBA", "M.Pharm", "MBBS", "PhD", "BSc", "MSc", "MHM", "BBA", "BHM", "B.Pharm", "B.Tech", "M.Ed"])
    suicidal_thoughts = st.selectbox("Apakah Anda pernah memiliki pikiran untuk bunuh diri?", ["Yes", "No"])
    family_history = st.selectbox("Apakah Anda memiliki riwayat keluarga dengan penyakit mental?", ["Yes", "No"])

    # Numerical inputs
    age = st.number_input("Usia", min_value=1, max_value=100, value=25)
    academic_pressure = st.number_input("Tekanan Akademik", min_value=0, max_value=5, value=3)
    work_pressure = st.number_input("Tekanan Kerja", min_value=0, max_value=5, value=3)
    cgpa = st.number_input("IPK (CGPA)", min_value=2.0, max_value=4.0, value=3.5)
    study_satisfaction = st.number_input("Kepuasan Belajar", min_value=0, max_value=5, value=3)
    job_satisfaction = st.number_input("Kepuasan Kerja", min_value=0, max_value=5, value=3)
    work_study_hours = st.number_input("Jam Kerja/Belajar per Hari (Jam)", min_value=0, max_value=24, value=6)
    financial_stress = st.number_input("Tingkat Stres Keuangan", min_value=0, max_value=5, value=3)

    # Mapping input ke nilai numerik
    mappings = {
        "Gender": {"Male": 0, "Female": 1},
        "City": {"Ghaziabad": 0, "Kalyan": 1, "Bhopal": 2, "Thane": 3, "Indore": 4, "Pune": 5, "Bangalore": 6, "Hyderabad": 7, "Srinagar": 8, "Nashik": 9, "Kolkata": 10, "Ahmedabad": 11, "Varanasi": 12, "Chennai": 13, "Jaipur": 14, "Surat": 15, "Vasai-Virar": 16, "Rajkot": 17, "Patna": 18, "Mumbai": 19, "Vadodara": 20, "Lucknow": 21, "Faridabad": 22, "Meerut": 23, "Kanpur": 24, "Visakhapatnam": 25, "Ludhiana": 26, "Nagpur": 27, "Agra": 28, "Delhi": 29},
        "Working Professional or Student": {"Working Professional": 0, "Student": 1},
        "Profession": {"Student":36,"Teacher": 0, "Financial Analyst": 1, "UX/UI Designer": 2, "Civil Engineer": 3, "Accountant": 4, "Lawyer": 5, "Content Writer": 6, "Pilot": 7, "Customer Support": 8, "Judge": 9, "Architect": 10, "HR Manager": 11, "Digital Marketer": 12, "Sales Executive": 13, "Business Analyst": 14, "Mechanical Engineer": 15, "Consultant": 16, "Data Scientist": 17, "Pharmacist": 18, "Software Engineer": 19, "Travel Consultant": 20, "Manager": 21, "Entrepreneur": 22, "Doctor": 23, "Researcher": 24, "Plummer": 25, "Educational Consultant": 26, "Chemist": 27, "Research Analyst": 28, "Chef": 29, "Electrician": 30, "Graphic Designer": 31, "Investment Banker": 32, "Finanancial Analyst": 33, "Marketing Manager": 34, "Plumber": 35},
        "Sleep Duration": {"7-8 hours": 0, "5-6 hours": 1, "More than 8 hours": 2, "Less than 5 hours": 3},
        "Dietary Habits": {"Moderate": 0, "Unhealthy": 1, "Healthy": 2},
        "Degree": {"MA": 0, "B.Com": 1, "M.Com": 2, "MD": 3, "BE": 4, "MCA": 5, "BA": 6, "LLM": 7, "BCA": 8, "Class 12": 9, "B.Ed": 10, "M.Tech": 11, "LLB": 12, "B.Arch": 13, "ME": 14, "MBA": 15, "M.Pharm": 16, "MBBS": 17, "PhD": 18, "BSc": 19, "MSc": 20, "MHM": 21, "BBA": 22, "BHM": 23, "B.Pharm": 24, "B.Tech": 25, "M.Ed": 26},
        "Have you ever had suicidal thoughts ?": {"Yes": 1, "No": 0},
        "Family History of Mental Illness": {"Yes": 1, "No": 0}
    }

    # Transformasi input menjadi DataFrame
    input_dict = {
        "Name" : name_numeric,
        "Gender": mappings["Gender"][gender],
        "Age": age,
        "City": mappings["City"][city],
        "Working Professional Or Student": mappings["Working Professional or Student"][work_status],
        "Profession": mappings["Profession"][profession],
        "Academic Pressure": academic_pressure,
        "Work Pressure": work_pressure,
        "Cgpa": cgpa,
        "Study Satisfaction": study_satisfaction,
        "Job Satisfaction": job_satisfaction,
        "Sleep Duration": mappings["Sleep Duration"][sleep_duration],
        "Dietary Habits": mappings["Dietary Habits"][dietary_habits],
        "Degree": mappings["Degree"][degree],
        "Have You Ever Had Suicidal Thoughts ?": mappings["Have you ever had suicidal thoughts ?"][suicidal_thoughts],
        "Work/Study Hours": work_study_hours,
        "Financial Stress": financial_stress,
        "Family History Of Mental Illness": mappings["Family History of Mental Illness"][family_history]
    }

    input_data = pd.DataFrame([input_dict])

    # Load model
    model = load_model()
    if model is None:
        return

    # Prediksi saat tombol ditekan
    if st.button("Prediksi"):
        try:
            prediction = model.predict(input_data)
            prob = model.predict_proba(input_data)[0]

            if prediction[0] == 1:
                st.error(f"Hasil Prediksi: Anda mungkin mengalami depresi (Probabilitas: {prob[1]*100:.2f}%).")
            else:
                st.success(f"Hasil Prediksi: Anda tidak menunjukkan tanda-tanda depresi (Probabilitas: {prob[0]*100:.2f}%).")
        except Exception as e:
            st.error(f"Error saat melakukan prediksi: {e}")

if __name__ == "__main__":
    main()
