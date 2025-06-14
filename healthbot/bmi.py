
def calculate_bmi(weight, height):
    height_m = height / 100  # cm to meters
    bmi = weight / (height_m ** 2)
    return round(bmi, 2)
