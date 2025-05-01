import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_data():
    """Load and prepare the data for classification."""
    print("Loading and preparing data...")
    # Load the data
    data = pd.read_csv('data.csv')
    
    # Extract month information from Sample Name for classification
    data['Month'] = data['Sample Name'].str.extract('(\d+)months').astype(int)
    
    # Create binary classification target based on month
    # We'll classify samples as early (≤12 months) or late (>12 months)
    data['Target'] = (data['Month'] > 12).astype(int)
    
    # Select features (excluding Sample Name and Month)
    features = [col for col in data.columns if col not in ['Sample Name', 'Month', 'Target']]
    X = data[features]
    y = data['Target']
    
    return X, y, features

def train_random_forest(X, y, features):
    """Train and evaluate the Random Forest model."""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train the model
    print("\nTraining Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test_scaled)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Perform cross-validation
    cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5)
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    return rf_model, X_test_scaled, y_test, y_pred

def plot_feature_importance(model, features):
    """Plot feature importance from the Random Forest model."""
    # Get feature importance
    importance = model.feature_importances_
    
    # Create DataFrame of features and their importance scores
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': importance
    })
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
    plt.title('Top 10 Most Important Features')
    plt.xlabel('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def plot_confusion_matrix(y_test, y_pred):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    # Prepare data
    X, y, features = prepare_data()
    
    # Train and evaluate model
    model, X_test_scaled, y_test, y_pred = train_random_forest(X, y, features)
    
    # Plot feature importance
    plot_feature_importance(model, features)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)
    
    print("\nVisualization files created:")
    print("1. feature_importance.png - Shows the most important features")
    print("2. confusion_matrix.png - Shows the model's prediction performance")

if __name__ == "__main__":
    main()
