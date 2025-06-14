
def get_response(message):
    if "hello" in message or "hi" in message:
        return "Hello! I'm your Healthcare Assistant. Type 'bmi' or 'symptom' to begin."
    elif "your name" in message:
        return "I’m HealthBot, here to assist with your health queries."
    elif "thank" in message:
        return "You're welcome! Stay healthy."
    elif "help" in message:
        return "You can ask me things like:\n- bmi 70 170\n- symptom fever\n- What is BMI?"
    elif "what is bmi" in message:
        return "BMI (Body Mass Index) is a measure of body fat based on height and weight."
    elif "calorie" in message:
        return "On average, adults need 2000–2500 calories/day. It varies by age and activity."
    elif "bye" in message:
        return "Goodbye! Take care!"
    else:
        return "I didn't quite get that. Try asking about BMI or symptoms."
