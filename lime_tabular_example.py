
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- Configuration ---
RANDOM_STATE = 42

print("
--- Explainable AI Toolkit: LIME Tabular Example ---")

# --- Generate Synthetic Data ---
def generate_synthetic_data():
    print("Generating synthetic tabular data...")
    np.random.seed(RANDOM_STATE)
    data = {
        "feature_A": np.random.rand(1000) * 10,
        "feature_B": np.random.rand(1000) * 5,
        "feature_C": np.random.randint(0, 2, 1000),
        "feature_D": np.random.normal(5, 2, 1000),
    }
    df = pd.DataFrame(data)
    
    # Create a target variable with some dependency on features
    df["target"] = ((df["feature_A"] > 5) & (df["feature_B"] < 2.5) | (df["feature_C"] == 1)).astype(int)
    
    print("Synthetic data generated.")
    return df

# --- Train a RandomForestClassifier ---
def train_model(df):
    print("Training RandomForestClassifier...")
    X = df.drop("target", axis=1)
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print(f"Model trained with accuracy: {accuracy:.4f}")
    return model, X_train, X_test, y_test

# --- Explain a Prediction using LIME ---
def explain_prediction_with_lime(model, X_train, X_test, feature_names, class_names, instance_idx=0):
    print(f"
Explaining prediction for instance {instance_idx} using LIME...")
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification"
    )
    
    instance_to_explain = X_test.iloc[instance_idx].values
    explanation = explainer.explain_instance(
        data_row=instance_to_explain,
        predict_fn=model.predict_proba,
        num_features=len(feature_names)
    )
    
    print("LIME Explanation:")
    for feature, weight in explanation.as_list():
        print(f"  {feature}: {weight:.4f}")
    
    # You can also visualize the explanation
    # explanation.show_in_notebook(show_table=True, show_all=False)
    print("LIME explanation generated. (Visualization requires notebook environment)")


if __name__ == "__main__":
    df = generate_synthetic_data()
    model, X_train, X_test, y_test = train_model(df)
    
    feature_names = X_train.columns.tolist()
    class_names = ["Class 0", "Class 1"]
    
    explain_prediction_with_lime(model, X_train, X_test, feature_names, class_names, instance_idx=5)
    print("Explainable AI Toolkit script finished.")
