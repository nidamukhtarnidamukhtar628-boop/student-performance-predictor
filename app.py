from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load all saved models
score_model = joblib.load('student_score_model.pkl')
grade_model = joblib.load('student_grade_model.pkl')
result_model = joblib.load('student_result_model.pkl')
result_scaler = joblib.load('student_result_scaler.pkl')

print("All models loaded successfully!")


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Step 1: Get form data
    study_hours = float(request.form['study_hours'])
    attendance = float(request.form['attendance'])
    midterm = float(request.form['midterm'])
    assignments = float(request.form['assignments'])
    quizzes = float(request.form['quizzes'])
    participation = float(request.form['participation'])
    projects = float(request.form['projects'])

    # Step 2: Build DataFrame (same column names as training)
    new_student = pd.DataFrame({
        'Study_Hours_Per_Week': [study_hours],
        'Attendance (%)': [attendance],
        'Midterm_Score': [midterm],
        'Assignments_Avg': [assignments],
        'Quizzes_Avg': [quizzes],
        'Participation_Score': [participation],
        'Projects_Score': [projects]
    })

    # Step 3: Predict Total Score
    predicted_score = score_model.predict(new_student)[0]

    # Step 4: Predict Grade
    predicted_grade = grade_model.predict(new_student)[0]

    # Step 5: Predict Pass/Fail (needs scaling)
    scaled_student = result_scaler.transform(new_student)
    predicted_result = result_model.predict(scaled_student)[0]

    # Step 6: Risk level (same thresholds as training)
    fail_threshold = 63.4
    risk_threshold = 68.5

    if predicted_score < fail_threshold:
        risk_level = "Fail"
    elif predicted_score < risk_threshold:
        risk_level = "At Risk"
    else:
        risk_level = "Safe"

    # Step 7: Send results to HTML
    return render_template(
        'index.html',
        prediction=True,
        score=round(predicted_score, 2),
        grade=predicted_grade,
        result=predicted_result,
        risk=risk_level
    )
import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)