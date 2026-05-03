def generate_feedback(score):
    feedback = []

    def check(feature, value):
        if value < 50:
            return f"{feature} masih kurang natural"
        elif value < 75:
            return f"{feature} cukup baik tapi bisa ditingkatkan"
        else:
            return f"{feature} sudah mendekati native"

    for gender in ["male", "female"]:
        for feature, val in score[gender].items():
            feedback.append(f"{gender} - {check(feature, val)}")

    return feedback