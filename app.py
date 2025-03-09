from  flask  import  Flask, render_template... by Sravani M23:15Sravani M
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
import pandas as pd
import os
from datetime import datetime
import json
import io
import base64
import random
# Set matplotlib backend to non-interactive
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
 
# Create the instance directory if it doesn't exist
instance_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance')
if not os.path.exists(instance_path):
    os.makedirs(instance_path)
 
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(instance_path, "users.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
 
# Load the model and training data
try:
    with open('logistic_regressor.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    training_data = pd.read_csv('WineQT.csv')
except Exception as e:
    print(f"Error loading model or data: {e}")
    model = None
    training_data = None
 
# Define valid ranges for wine features
WINE_FEATURE_RANGES = {
    'fixed_acidity': (4.0, 15.0),
    'volatile_acidity': (0.1, 1.2),
    'citric_acid': (0.0, 1.0),
    'residual_sugar': (0.5, 15.0),
    'chlorides': (0.01, 0.2),
    'free_sulfur_dioxide': (1, 100),
    'total_sulfur_dioxide': (6, 300),
    'density': (0.99, 1.01),
    'pH': (2.8, 4.0),
    'sulphates': (0.3, 1.5),
    'alcohol': (8.0, 14.0)
}
 
FEATURE_DESCRIPTIONS = {
    'fixed_acidity': 'Non-volatile acids that contribute to the wine\'s tartness',
    'volatile_acidity': 'Amount of acetic acid, too high can lead to vinegar taste',
    'citric_acid': 'Adds freshness and flavor to wines',
    'residual_sugar': 'Amount of sugar remaining after fermentation',
    'chlorides': 'Amount of salt in the wine',
    'free_sulfur_dioxide': 'Prevents microbial growth and wine oxidation',
    'total_sulfur_dioxide': 'Total amount of SO2 (free + bound forms)',
    'density': 'Density relative to water',
    'pH': 'Describes how acidic or basic the wine is',
    'sulphates': 'Wine additive that contributes to SO2 levels',
    'alcohol': 'Percent alcohol content of the wine'
}
 
# Add these constants at the top with other constants
QUALITY_CATEGORIES = {
    'poor': (0, 4),
    'below_average': (4, 5),
    'average': (5, 6),
    'good': (6, 7),
    'excellent': (7, 10)
}
 
IDEAL_RANGES = {
    'alcohol': (11.0, 14.0),
    'volatile_acidity': (0.2, 0.5),
    'sulphates': (0.6, 1.0),
    'pH': (3.0, 3.4),
    'fixed_acidity': (6.0, 8.0),
    'citric_acid': (0.3, 0.8),
    'free_sulfur_dioxide': (30, 40),
    'total_sulfur_dioxide': (90, 130)
}
 
# Define weights for each feature's importance in quality
FEATURE_WEIGHTS = {
    'alcohol': 0.3,           # Alcohol has strong correlation with quality
    'volatile_acidity': 0.15, # Important for taste
    'sulphates': 0.15,        # Important for preservation
    'fixed_acidity': 0.1,
    'citric_acid': 0.1,
    'pH': 0.1,
    'free_sulfur_dioxide': 0.05,
    'total_sulfur_dioxide': 0.05
}
 
def calculate_weighted_quality(features):
    """Calculate wine quality based on how close each parameter is to its ideal range"""
    quality_score = 0
   
    # Calculate score based on alcohol content (higher alcohol generally means better quality)
    if features['alcohol'] >= IDEAL_RANGES['alcohol'][0]:
        alcohol_score = min((features['alcohol'] - IDEAL_RANGES['alcohol'][0]) /
                          (IDEAL_RANGES['alcohol'][1] - IDEAL_RANGES['alcohol'][0]), 1.0)
        quality_score += alcohol_score * FEATURE_WEIGHTS['alcohol'] * 10
   
    # Penalize high volatile acidity (lower is better)
    if features['volatile_acidity'] <= IDEAL_RANGES['volatile_acidity'][1]:
        va_score = 1 - (features['volatile_acidity'] - IDEAL_RANGES['volatile_acidity'][0]) / \
                      (IDEAL_RANGES['volatile_acidity'][1] - IDEAL_RANGES['volatile_acidity'][0])
        quality_score += va_score * FEATURE_WEIGHTS['volatile_acidity'] * 10
   
    # Score sulphates (higher is generally better within range)
    if IDEAL_RANGES['sulphates'][0] <= features['sulphates'] <= IDEAL_RANGES['sulphates'][1]:
        sulphates_score = (features['sulphates'] - IDEAL_RANGES['sulphates'][0]) / \
                         (IDEAL_RANGES['sulphates'][1] - IDEAL_RANGES['sulphates'][0])
        quality_score += sulphates_score * FEATURE_WEIGHTS['sulphates'] * 10
   
    # Score pH (closer to middle of range is better)
    ph_mid = sum(IDEAL_RANGES['pH']) / 2
    ph_range = IDEAL_RANGES['pH'][1] - IDEAL_RANGES['pH'][0]
    ph_score = 1 - abs(features['pH'] - ph_mid) / (ph_range / 2)
    quality_score += max(0, ph_score * FEATURE_WEIGHTS['pH'] * 10)
   
    # Score fixed acidity
    if IDEAL_RANGES['fixed_acidity'][0] <= features['fixed_acidity'] <= IDEAL_RANGES['fixed_acidity'][1]:
        acid_score = 1 - abs(features['fixed_acidity'] - sum(IDEAL_RANGES['fixed_acidity'])/2) / \
                        (IDEAL_RANGES['fixed_acidity'][1] - IDEAL_RANGES['fixed_acidity'][0])
        quality_score += acid_score * FEATURE_WEIGHTS['fixed_acidity'] * 10
   
    # Score sulfur dioxide levels
    if IDEAL_RANGES['free_sulfur_dioxide'][0] <= features['free_sulfur_dioxide'] <= IDEAL_RANGES['free_sulfur_dioxide'][1]:
        so2_score = 1 - abs(features['free_sulfur_dioxide'] - sum(IDEAL_RANGES['free_sulfur_dioxide'])/2) / \
                       (IDEAL_RANGES['free_sulfur_dioxide'][1] - IDEAL_RANGES['free_sulfur_dioxide'][0])
        quality_score += so2_score * FEATURE_WEIGHTS['free_sulfur_dioxide'] * 10
   
    # Add baseline quality
    base_quality = 5.0
    final_quality = base_quality + quality_score
   
    # Ensure quality is within bounds
    return max(3, min(9, final_quality))
 
# Function to check if input exists in training data
def is_in_training_data(features, tolerance=0.01):
    input_array = np.array([list(features.values())])
    for _, row in training_data.iloc[:, :-1].iterrows():
        if np.allclose(input_array[0], row.values, rtol=tolerance):
            return True
    return False
 
# Function to generate random realistic values
def get_random_suggestion():
    return {k: round(random.uniform(v[0], v[1]), 3) for k, v in WINE_FEATURE_RANGES.items()}
 
def get_quality_category(quality_score):
    for category, (min_val, max_val) in QUALITY_CATEGORIES.items():
        if min_val <= quality_score < max_val:
            return category
    return 'excellent' if quality_score >= 7 else 'poor'
 
def generate_recommendations(features, quality_score):
    recommendations = []
   
    if features['alcohol'] < IDEAL_RANGES['alcohol'][0]:
        recommendations.append(f"Increase alcohol content (currently {features['alcohol']:.1f}%, aim for {IDEAL_RANGES['alcohol'][0]}-{IDEAL_RANGES['alcohol'][1]}%)")
   
    if features['volatile_acidity'] > IDEAL_RANGES['volatile_acidity'][1]:
        recommendations.append(f"Reduce volatile acidity (currently {features['volatile_acidity']:.2f}, aim for {IDEAL_RANGES['volatile_acidity'][0]}-{IDEAL_RANGES['volatile_acidity'][1]})")
   
    if features['sulphates'] < IDEAL_RANGES['sulphates'][0]:
        recommendations.append(f"Increase sulphates (currently {features['sulphates']:.2f}, aim for {IDEAL_RANGES['sulphates'][0]}-{IDEAL_RANGES['sulphates'][1]})")
   
    if not (IDEAL_RANGES['pH'][0] <= features['pH'] <= IDEAL_RANGES['pH'][1]):
        recommendations.append(f"Adjust pH level (currently {features['pH']:.2f}, aim for {IDEAL_RANGES['pH'][0]}-{IDEAL_RANGES['pH'][1]})")
   
    if features['free_sulfur_dioxide'] < IDEAL_RANGES['free_sulfur_dioxide'][0]:
        recommendations.append(f"Increase free sulfur dioxide (currently {features['free_sulfur_dioxide']:.0f}, aim for {IDEAL_RANGES['free_sulfur_dioxide'][0]}-{IDEAL_RANGES['free_sulfur_dioxide'][1]})")
   
    return recommendations
 
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship('WinePrediction', backref='user', lazy=True)
 
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
 
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
 
class WinePrediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    prediction_date = db.Column(db.DateTime, default=datetime.utcnow)
    wine_features = db.Column(db.Text, nullable=False)  # JSON string of features
    predicted_quality = db.Column(db.Float, nullable=False)
 
    def set_features(self, features_dict):
        self.wine_features = json.dumps(features_dict)
 
    def get_features(self):
        return json.loads(self.wine_features)
 
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
 
@app.route('/')
def index():
    return render_template('index.html')
 
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
       
        if user and user.check_password(password):
            login_user(user)
            flash('Logged in successfully!', 'success')
            return redirect(url_for('dashboard'))
        flash('Invalid username or password', 'error')
    return render_template('login.html')
 
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
       
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return redirect(url_for('signup'))
           
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'error')
            return redirect(url_for('signup'))
           
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
       
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')
 
@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    prediction = None
    warning = None
    quality_category = None
    recommendations = None
   
    if request.method == 'POST':
        try:
            features = {
                'fixed_acidity': float(request.form['fixed_acidity']),
                'volatile_acidity': float(request.form['volatile_acidity']),
                'citric_acid': float(request.form['citric_acid']),
                'residual_sugar': float(request.form['residual_sugar']),
                'chlorides': float(request.form['chlorides']),
                'free_sulfur_dioxide': float(request.form['free_sulfur_dioxide']),
                'total_sulfur_dioxide': float(request.form['total_sulfur_dioxide']),
                'density': float(request.form['density']),
                'pH': float(request.form['pH']),
                'sulphates': float(request.form['sulphates']),
                'alcohol': float(request.form['alcohol'])
            }
           
            # Validate input ranges
            for feature, value in features.items():
                min_val, max_val = WINE_FEATURE_RANGES[feature]
                if not min_val <= value <= max_val:
                    flash(f'{feature} must be between {min_val} and {max_val}', 'error')
                    return redirect(url_for('dashboard'))
           
            # Check if input matches training data
            if is_in_training_data(features):
                warning = "These values are very similar to examples in our training dataset. Try adjusting the values for a more interesting prediction!"
           
            # Use our custom prediction logic instead of the model
            prediction = calculate_weighted_quality(features)
           
            # Get quality category and recommendations
            quality_category = get_quality_category(prediction)
            recommendations = generate_recommendations(features, prediction)
           
            # Save prediction to database
            wine_pred = WinePrediction(
                user_id=current_user.id,
                predicted_quality=prediction
            )
            wine_pred.set_features(features)
            db.session.add(wine_pred)
            db.session.commit()
           
        except Exception as e:
            flash(f'Error in prediction: {str(e)}', 'error')
   
    # Get user's prediction history
    predictions_history = WinePrediction.query.filter_by(user_id=current_user.id).order_by(WinePrediction.prediction_date.desc()).limit(5).all()
   
    return render_template('dashboard.html',
                         prediction=prediction,
                         predictions_history=predictions_history,
                         feature_ranges=WINE_FEATURE_RANGES,
                         feature_descriptions=FEATURE_DESCRIPTIONS,
                         warning=warning,
                         quality_category=quality_category,
                         recommendations=recommendations,
                         ideal_ranges=IDEAL_RANGES)
 
@app.route('/get_random_values')
@login_required
def get_random_values():
    return jsonify(get_random_suggestion())
 
@app.route('/profile')
@login_required
def profile():
    predictions_count = WinePrediction.query.filter_by(user_id=current_user.id).count()
    return render_template('profile.html', predictions_count=predictions_count)
 
@app.route('/export-predictions')
@login_required
def export_predictions():
    predictions = WinePrediction.query.filter_by(user_id=current_user.id).all()
    data = []
    for pred in predictions:
        row = pred.get_features()
        row['prediction_date'] = pred.prediction_date
        row['predicted_quality'] = pred.predicted_quality
        data.append(row)
   
    df = pd.DataFrame(data)
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
   
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'wine_predictions_{datetime.now().strftime("%Y%m%d")}.csv'
    )
 
@app.route('/wine-stats')
@login_required
def wine_stats():
    predictions = WinePrediction.query.filter_by(user_id=current_user.id).all()
    if not predictions:
        flash('No predictions available for analysis', 'info')
        return redirect(url_for('dashboard'))
   
    data = []
    for pred in predictions:
        features = pred.get_features()
        features['predicted_quality'] = pred.predicted_quality
        features['prediction_date'] = pred.prediction_date
        data.append(features)
   
    df = pd.DataFrame(data)
   
    # Generate statistics
    stats = {
        'avg_quality': df['predicted_quality'].mean(),
        'total_predictions': len(df),
        'best_quality': df['predicted_quality'].max(),
        'worst_quality': df['predicted_quality'].min(),
        'avg_alcohol': df['alcohol'].mean(),
        'avg_ph': df['pH'].mean()
    }
   
    try:
        # Create a figure with subplots
        fig = plt.figure(figsize=(15, 6))
       
        # 1. Quality Distribution Pie Chart
        plt.subplot(1, 2, 1)
        quality_counts = df['predicted_quality'].value_counts().sort_index()
        plt.pie(quality_counts.values, labels=[f'Quality {q:.1f}' for q in quality_counts.index],
                autopct='%1.1f%%', colors=plt.cm.viridis(np.linspace(0, 1, len(quality_counts))))
        plt.title('Wine Quality Distribution')
       
        # 2. Quality Trend Over Time
        plt.subplot(1, 2, 2)
        df['prediction_date'] = pd.to_datetime(df['prediction_date'])
        df.set_index('prediction_date')['predicted_quality'].plot(
            marker='o', linestyle='-', linewidth=2, markersize=8)
        plt.title('Wine Quality Trend Over Time')
        plt.xlabel('Prediction Date')
        plt.ylabel('Predicted Quality')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
       
        # Adjust layout
        plt.tight_layout()
       
        # Save plots to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        plt.close('all')
        buf.seek(0)
        plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
       
        # Calculate additional statistics
        recent_trend = 'improving' if df['predicted_quality'].iloc[-3:].mean() > df['predicted_quality'].iloc[:-3].mean() else 'declining'
        stats.update({
            'quality_trend': recent_trend,
            'most_common_quality': quality_counts.index[quality_counts.argmax()],
            'predictions_this_month': len(df[df['prediction_date'] > datetime.now().replace(day=1)])
        })
       
    except Exception as e:
        print(f"Error generating plots: {str(e)}")
        plot_data = None
        flash('Error generating visualizations', 'error')
   
    return render_template('wine_stats.html', stats=stats, visualization_plot=plot_data)
 
@app.route('/recommendations')
@login_required
def recommendations():
    # Get user's recent predictions
    recent_predictions = WinePrediction.query.filter_by(user_id=current_user.id)\
        .order_by(WinePrediction.prediction_date.desc())\
        .limit(5).all()
   
    if not recent_predictions:
        flash('Make some predictions first to get recommendations!', 'info')
        return redirect(url_for('dashboard'))
   
    # Calculate average characteristics of high-quality predictions (quality >= 6)
    high_quality_predictions = WinePrediction.query.filter_by(user_id=current_user.id)\
        .filter(WinePrediction.predicted_quality >= 6.0).all()
   
    if high_quality_predictions:
        high_quality_features = []
        for pred in high_quality_predictions:
            features = pred.get_features()
            high_quality_features.append(features)
       
        df = pd.DataFrame(high_quality_features)
        ideal_characteristics = {
            'alcohol': df['alcohol'].mean(),
            'pH': df['pH'].mean(),
            'sulphates': df['sulphates'].mean(),
            'fixed_acidity': df['fixed_acidity'].mean(),
            'volatile_acidity': df['volatile_acidity'].mean(),
            'citric_acid': df['citric_acid'].mean(),
            'residual_sugar': df['residual_sugar'].mean()
        }
       
        # Generate recommendations
        recommendations = {
            'ideal_characteristics': ideal_characteristics,
            'tips': [
                'Maintain alcohol content around {:.1f}%'.format(ideal_characteristics['alcohol']),
                'Keep pH levels near {:.2f}'.format(ideal_characteristics['pH']),
                'Aim for sulphates concentration of {:.2f} g/dm³'.format(ideal_characteristics['sulphates']),
                'Maintain fixed acidity around {:.1f} g/dm³'.format(ideal_characteristics['fixed_acidity'])
            ],
            'recent_quality': recent_predictions[0].predicted_quality if recent_predictions else None
        }
    else:
        recommendations = {
            'ideal_characteristics': None,
            'tips': [
                'Try increasing alcohol content slightly',
                'Maintain pH between 3.0 and 3.4',
                'Keep sulphates concentration balanced',
                'Monitor acidity levels carefully'
            ],
            'recent_quality': recent_predictions[0].predicted_quality if recent_predictions else None
        }
   
    return render_template('recommendations.html', recommendations=recommendations)
 
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('index'))
 
@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    if request.method == 'POST':
        email = request.form.get('email')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
 
        # Update email
        if email and email != current_user.email:
            if User.query.filter_by(email=email).first():
                flash('Email already registered', 'error')
                return redirect(url_for('profile'))
            current_user.email = email
 
        # Update password
        if new_password:
            if new_password != confirm_password:
                flash('Passwords do not match', 'error')
                return redirect(url_for('profile'))
            current_user.set_password(new_password)
 
        try:
            db.session.commit()
            flash('Profile updated successfully', 'success')
        except Exception as e:
            db.session.rollback()
            flash('Error updating profile', 'error')
 
    return redirect(url_for('profile'))
 
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
