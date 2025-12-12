

def improved_reward(BG_last_hour):
    G_prev = BG_last_hour[-2]
    G = BG_last_hour[-1]

    # Smooth symmetric reward around 110 mg/dL
    # Max reward at BG = 110, gradually decreases as moving away
    reward = -abs(G - 110) / 50.0

    # Small bonus for improving toward 110 
    delta = abs(G_prev - 110) - abs(G - 110)
    reward += 0.2 * (delta / 50.0)

    #  VERY mild safety penalties 
    if G < 70:     # mild hypo
        reward -= 0.2
    if G < 55:     # more hypo
        reward -= 0.2
    if G > 250:    # mild hyper
        reward -= 0.2

    return reward