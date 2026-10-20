import pandas as pd
import io

# 1. Load the Dataset
csv_data = """Subject,SenderType,Links,Class
Offer,Free,Many,Spam
Offer,Free,Many,Spam
Offer,Free,Few,Spam
Urgent,Free,Many,Spam
Urgent,Free,Many,Spam
Prize,Free,Many,Spam
Prize,Free,Few,Spam
Offer,Cheap,Many,Spam
Offer,Cheap,Few,Spam
Urgent,Cheap,Many,Spam
Urgent,Cheap,Few,Spam
Prize,Cheap,Many,Spam
Prize,Cheap,Few,Spam
Offer,Free,Many,Spam
Urgent,Free,Few,Spam
Prize,Free,Many,Spam
Offer,Cheap,Many,Spam
Urgent,Cheap,Many,Spam
Prize,Cheap,Few,Spam
Offer,Free,Few,Spam
Meeting,Work,Few,Not Spam
Meeting,Work,Many,Not Spam
Report,Work,Few,Not Spam
Report,Work,Many,Not Spam
Update,Work,Few,Not Spam
Update,Work,Many,Not Spam
Meeting,Free,Few,Not Spam
Report,Free,Few,Not Spam
Update,Free,Many,Not Spam
Meeting,Work,Few,Not Spam
Report,Work,Few,Not Spam
Update,Work,Many,Not Spam
Meeting,Work,Many,Not Spam
Report,Work,Few,Not Spam
Update,Work,Few,Not Spam
Meeting,Free,Many,Not Spam
Report,Free,Few,Not Spam
Update,Work,Many,Not Spam
Meeting,Work,Few,Not Spam
Report,Work,Many,Not Spam
Update,Work,Few,Not Spam
Meeting,Free,Few,Not Spam
Report,Free,Many,Not Spam
Update,Free,Few,Not Spam
Meeting,Work,Many,Not Spam
Report,Work,Few,Not Spam
Update,Work,Many,Not Spam
Meeting,Work,Few,Not Spam
Report,Work,Many,Not Spam
Update,Free,Few,Not Spam"""

df = pd.read_csv(io.StringIO(csv_data))

# 2. Define Feature Vocabularies based on image_7f8d8d.png
# This is used for the K (number of unique categorical values) in Laplace Smoothing
vocab_sizes = {
    'Subject': 6,      # Offer, Urgent, Prize, Meeting, Report, Update
    'SenderType': 3,   # Free, Cheap, Work
    'Links': 2         # Few, Many
}

# 3. Naive Bayes Classifier Function
def predict_spam(test_case):
    classes = df['Class'].unique()
    total_samples = len(df)
    
    scores = {}
    
    print(f"--- Predicting for Test Case: {test_case} ---")
    
    for c in classes:
        # P(C) - Prior Probability
        subset_c = df[df['Class'] == c]
        p_c = len(subset_c) / total_samples
        
        prob_conditional = 1.0
        
        # P(Feature | C)
        for feature, value in test_case.items():
            count_feature_in_c = len(subset_c[subset_c[feature] == value])
            
            # Apply Laplace Smoothing: (count + alpha) / (N_c + alpha * K)
            # Applied universally to ensure valid probability distributions if any zero-frequency is possible
            alpha = 1 
            k = vocab_sizes[feature]
            n_c = len(subset_c)
            
            smoothed_prob = (count_feature_in_c + alpha) / (n_c + alpha * k)
            prob_conditional *= smoothed_prob
            
        # Score(C) = P(C) * P(Subject|C) * P(SenderType|C) * P(Links|C)
        scores[c] = p_c * prob_conditional
        print(f"Score({c}): {scores[c]:.6e}")
        
    # Compare and Predict
    prediction = max(scores, key=scores.get)
    print(f"\nPrediction: {prediction}")
    return prediction

# 4. Evaluate a Sample Test Case
# You can modify these values to test different scenarios
sample_test = {
    'Subject': 'Offer',
    'SenderType': 'Work',
    'Links': 'Many'
}

predict_spam(sample_test)