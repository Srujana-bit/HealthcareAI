import os
import sys
import joblib
from functools import wraps
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash
)

# Add current directory to module search path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Local module imports
from diagnosis import generate_diagnosis

from recommendations import generate_recommendations

# Load ML models (safe-load with fallback)
def load_model(path):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        return None

diabetes_model = load_model("models/diabetes_model.pkl")
heart_model    = load_model("models/heart_model.pkl")
stroke_model   = load_model("models/stroke_model.pkl")

# Flask setup
app = Flask(__name__)
app.secret_key = "supersecret"
USER_FILE = "users.txt"

# --------------------- User Utilities ---------------------
def load_users():
    if not os.path.exists(USER_FILE):
        return {}
    with open(USER_FILE, "r", encoding="utf-8") as f:
        return {
            line.split(",")[0]: line.split(",")[1]
            for line in f.read().splitlines() if "," in line
        }

def save_user(username, password):
    users = load_users()
    if username in users:
        return False
    with open(USER_FILE, "a", encoding="utf-8") as f:
        f.write(f"{username},{password}\n")
    return True

def verify_user(username, password):
    return load_users().get(username) == password

# ------------------- Route Protection --------------------
def login_required(view):
    @wraps(view)
    def wrapped_view(**kwargs):
        if "user" not in session:
            flash("Please log in first.", "warning")
            return redirect(url_for("login"))
        return view(**kwargs)
    return wrapped_view

# --------------------- Auth Routes ------------------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if not username or not password:
            flash("Both fields are required.", "danger")
        elif save_user(username, password):
            session["user"] = username
            flash("Signup successful!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Username already exists.", "danger")
    
    return render_template("signup.html")
# -------------------- Dashboard Page ----------------------
@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")

# ---------------------- Main Pages ------------------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/bmi", methods=["GET", "POST"])
@login_required
def bmi():
    result = None
    if request.method == "POST":
        weight = float(request.form["weight"])
        height = float(request.form["height"]) / 100
        result = round(weight / (height * height), 2)
    return render_template("bmi.html", result=result)

@app.route("/calorie", methods=["GET", "POST"])
@login_required
def calorie():
    result = None
    if request.method == "POST":
        age     = int(request.form["age"])
        gender  = request.form["gender"]
        activity= float(request.form["activity"])
        weight  = float(request.form["weight"])
        height  = float(request.form["height"])
        bmr     = 10 * weight + 6.25 * height - 5 * age + (5 if gender == "male" else -161)
        result  = round(bmr * activity, 2)
    return render_template("calorie.html", result=result)

@app.route("/symptom-checker", methods=["GET", "POST"])
@login_required
def symptom_checker():
    result = None
    if request.method == "POST":
        symptom = request.form["symptom"].strip().lower()
        result  = generate_diagnosis(symptom)
    return render_template("symptom.html", result=result)
@app.route("/diagnosis", methods=["GET", "POST"])
@login_required
def diagnosis():
    result = diabetes_result = heart_result = stroke_result = None
    if request.method == "POST":
        form = request.form
        if "text" in form:
            result = generate_diagnosis(form["text"])

        elif "pregnancies" in form and diabetes_model:
            try:
                features = [
                    float(form["pregnancies"]),
                    float(form["glucose"]),
                    float(form["blood_pressure"]),
                    float(form["skin_thickness"]),
                    float(form["insulin"]),
                    float(form["bmi"]),
                    float(form["diabetes_pedigree"]),
                    float(form["age"]),
                    
                ]
                diabetes_result = "Positive" if diabetes_model.predict([features])[0] else "Negative"
            except (KeyError, ValueError) as e:
                diabetes_result = f"Error: {e}"

        elif "cp" in form and heart_model:
            # unchanged
            try:
                features = [
                    float(form["age"]), float(form["sex"]), float(form["cp"]),
                    float(form["trestbps"]), float(form["chol"]), float(form["fbs"]),
                    float(form["restecg"]), float(form["thalach"]), float(form["exang"]),
                    float(form["oldpeak"]), float(form["slope"]), float(form["ca"]),
                    float(form["thal"])
                ]
                heart_result = "Positive" if heart_model.predict([features])[0] else "Negative"
            except (KeyError, ValueError) as e:
                heart_result = f"Error: {e}"

        elif "hypertension" in form and stroke_model:
            # unchanged
            try:
                features = [
                    float(form["gender"]), float(form["age"]), float(form["hypertension"]),
                    float(form["heart_disease"]), float(form["ever_married"]),
                    float(form["work_type"]), float(form["Residence_type"]),
                    float(form["avg_glucose_level"]), float(form["bmi"]),
                    float(form["smoking_status"])
                ]
                stroke_result = "Positive" if stroke_model.predict([features])[0] else "Negative"
            except (KeyError, ValueError) as e:
                stroke_result = f"Error: {e}"

    return render_template(
        "diagnosis.html",
        result=result,
        diabetes_result=diabetes_result,
        heart_result=heart_result,
        stroke_result=stroke_result
    )
@app.route("/recommendations", methods=["GET", "POST"])
@login_required
def recommendations():
    tips = error = None
    if request.method == "POST":
        condition = request.form.get("condition", "").strip()
        if condition:
            tips = generate_recommendations(condition)
        else:
            error = "Please enter a health condition."
    return render_template("recommendations.html", tips=tips, error=error)
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"].strip()
        if verify_user(username, password):
            session["user"] = username
            flash("Logged in successfully.", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials.", "danger")
    return render_template("login.html")
@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for("login"))
# --------------------- Run Flask App ----------------------
if __name__ == "__main__":
    app.run(debug=True)
