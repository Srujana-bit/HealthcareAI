import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load BioGPT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT")
model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT")

# Device setup (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Symptom explanation map
explanation_map = {
    "Fever": "Your body is warmer than normal, usually because it’s fighting off an infection.",
    "Headache": "Pain or pressure in your head — it can be due to stress, lack of sleep, or illness.",
    "Vomiting": "When your stomach forces food or liquid out through your mouth, usually due to infection or bad food.",
    "Cold": "A common virus that causes sneezing, runny nose, and sometimes sore throat and coughing.",
    "Stomach Pain": "Pain in your tummy — can be from gas, indigestion, infection, or something you ate.",
    "Stomach Ache": "Another name for stomach pain; usually harmless but uncomfortable.",
    "Body Pains": "Your muscles or joints hurt — this can happen when you have a fever or flu.",
    "Cough": "A way your body clears your throat or lungs — it can be dry or with mucus.",
    "Runny Nose": "When your nose keeps dripping — often from a cold or allergy.",
    "Sore Throat": "Pain in your throat — usually from infection or overuse.",
    "Fatigue": "Feeling very tired or weak, even after resting.",
    "Diarrhea": "Loose or watery poop — often from infection, food, or stress.",
    "Constipation": "When it’s hard to poop or you don’t go often — usually from low fiber or water.",
    "Dizziness": "Feeling like you might fall or the room is spinning — can happen when you're tired or sick.",
    "Asthma": "A condition that makes it hard to breathe, especially when you're around dust or exercise too much.",
    "Allergic Rhinitis": "Sneezing and runny nose caused by things like dust or pollen.",
    "Sinusitis": "Blocked or swollen nose areas that cause pressure and stuffiness.",
    "Migraine": "A very strong headache that can make you feel sick or sensitive to light and sound.",
    "Gastroenteritis": "A stomach bug that causes vomiting and diarrhea — also called the stomach flu.",
    "UTI": "An infection in the pee area — makes it hurt when you go to the bathroom.",
    "COVID-19": "A virus that spreads easily — causes cough, fever, and sometimes loss of taste or smell.",
    "Flu (Influenza)": "A virus like a bad cold — makes you tired, feverish, and achy all over.",
    "Bronchitis": "Swelling in your airways — causes coughing and chest tightness.",
    "Pneumonia": "A serious infection in your lungs — makes breathing hard and causes chest pain.",
    "Sinusitis": "Swelling or infection in the sinuses (spaces in your face bones) that causes pressure, stuffy nose, headache, and sometimes thick mucus.",
    "Anxiety": "Feeling very worried or nervous, often about things that might happen in the future.",
    "Depression": "Feeling very sad or hopeless for a long time, losing interest in things you used to enjoy.",
    "Hypertension": "High blood pressure, which can make your heart work too hard and lead to other health problems.",
    "Diabetes": "A condition where your body has trouble using sugar for energy, leading to high blood sugar levels.",
    "Obesity": "Being very overweight, which can cause health problems like diabetes and heart disease.",
    "Heart Disease": "A condition that affects the heart's ability to pump blood, often due to blocked arteries.",
 }

# put this right after you declare explanation_map
explanation_map = {k.lower(): v for k, v in explanation_map.items()}

def clean_tip(tip: str) -> str:
    # Remove "the following" and numbered list patterns like (1), (2), etc.
    tip = re.sub(r"the following[:]*", "", tip, flags=re.IGNORECASE)
    tip = re.sub(r"\(\d+\)", "", tip)  # removes (1), (2), (3) etc.
    return tip.strip().capitalize()


# ────────────────────────────────────────────────────────────────
# 1. AI Diagnosis from Symptom Description
# ────────────────────────────────────────────────────────────────
def generate_recommendations(symptom: str):
    try:
        prompt = f"Patient presents with {symptom}. Provide three specific health recommendations to improve the condition:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs, 
            max_length=80, 
            num_beams=5, 
            early_stopping=True,
            no_repeat_ngram_size=2
        )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        result = result.replace(prompt, "").strip().lower()

        # Extract clean list of recommendations
        raw_items = [item.strip(" .") for item in re.split(r"[.,;]", result) if len(item.strip()) > 2]
        top_3 = raw_items[:3]

        recommendations = []
        for item in top_3:
            term = re.sub(r"\(.*?\)", "", item).strip()
            explanation = explanation_map.get(term.lower(), "No explanation available.")
            recommendations.append((term.title(), explanation))

        return recommendations

    except Exception as e:
        return [("Error", f"An error occurred: {str(e)}")]