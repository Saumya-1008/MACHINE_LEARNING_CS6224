import pandas as pd
import numpy as np
import io

# 1. Load the Dataset
csv_data = """Temperature_C,Humidity_pct,Wind_kmh,Play
22,45,12,No
24,50,15,No
25,55,18,No
27,60,20,Yes
29,65,22,Yes
31,70,25,Yes
33,75,28,Yes
35,80,30,Yes
21,40,10,No
23,48,13,No
26,52,16,No
28,58,19,Yes
30,62,21,Yes
32,68,24,Yes
34,72,27,Yes
36,78,31,Yes
20,42,11,No
22,47,14,No
24,53,17,No
27,57,20,Yes
29,63,23,Yes
31,67,26,Yes
33,73,29,Yes
35,77,32,Yes
19,38,9,No
21,44,12,No
23,49,15,No
26,56,18,Yes
28,61,21,Yes
30,66,24,Yes
32,71,27,Yes
34,76,30,Yes
18,35,8,No
20,41,10,No
22,46,13,No
25,54,16,No
27,59,19,Yes
29,64,22,Yes
31,69,25,Yes
33,74,28,Yes
17,32,7,No
19,39,9,No
21,43,11,No
24,51,14,No
26,55,17,Yes
28,60,20,Yes
30,65,23,Yes
32,70,26,Yes
34,75,29,Yes
36,82,33,Yes"""

df = pd.read_csv(io.StringIO(csv_data))

# 2. Extract Data by Class
df_yes = df[df['Play'] == 'Yes']
df_no = df[df['Play'] == 'No']

# 3. Calculate Class Priors with Laplace Smoothing 
# Using parameters from image_7f7ea7.png: alpha = 1, N = 50, K = 2
alpha = 1
N = len(df)
K = df['Play'].nunique()

N_yes = len(df_yes)
N_no = len(df_no)

P_yes = (N_yes + alpha) / (N + alpha * K)
P_no = (N_no + alpha) / (N + alpha * K)

# 4. Calculate Mean and Variance for Numerical Features per Class
# Using sample variance (ddof=1) standard for training data
features = ['Temperature_C', 'Humidity_pct', 'Wind_kmh']
stats = {
    'Yes': {f: {'mean': df_yes[f].mean(), 'var': df_yes[f].var(ddof=1)} for f in features},
    'No':  {f: {'mean': df_no[f].mean(), 'var': df_no[f].var(ddof=1)} for f in features}
}

# 5. Define Gaussian Likelihood Function
# Using formula from image_7f7ea7.png
def calculate_gaussian_prob(x, mean, var):
    exponent = np.exp(-((x - mean)**2) / (2 * var))
    return (1 / np.sqrt(2 * np.pi * var)) * exponent

# 6. Prediction Function
def predict_play_tennis(test_case):
    # Initialize scores with the prior probabilities
    score_yes = P_yes
    score_no = P_no
    
    print(f"--- Test Case: {test_case} ---")
    
    # Multiply by likelihoods for each feature
    for feature in features:
        x = test_case[feature]
        
        # P(Feature | Yes)
        prob_yes = calculate_gaussian_prob(x, stats['Yes'][feature]['mean'], stats['Yes'][feature]['var'])
        score_yes *= prob_yes
        
        # P(Feature | No)
        prob_no = calculate_gaussian_prob(x, stats['No'][feature]['mean'], stats['No'][feature]['var'])
        score_no *= prob_no
        
    print(f"Score(Yes) = P(Yes) * P(T|Yes) * P(H|Yes) * P(W|Yes) = {score_yes:e}")
    print(f"Score(No)  = P(No)  * P(T|No)  * P(H|No)  * P(W|No)  = {score_no:e}")
    
    # Compare Scores as required by image_7f7ea7.png and image_7f7ecb.png
    if score_yes > score_no:
        print("Prediction: Yes (Play Tennis)\n")
    else:
        print("Prediction: No (Do Not Play Tennis)\n")

# 7. Evaluate a Test Case
# Note: You can replace these values with any specific test case you need to predict.
sample_test_case = {
    'Temperature_C': 26, 
    'Humidity_pct': 58, 
    'Wind_kmh': 19
}

predict_play_tennis(sample_test_case)