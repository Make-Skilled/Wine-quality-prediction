from flask import Flask, render_template, request, redirect, url_for, flash, send_file
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

# Load the model
try:
    with open('logistic_regressor.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

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
            
            input_df = pd.DataFrame([features])
            prediction = model.predict(input_df)[0]
            
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
    
    return render_template('dashboard.html', prediction=prediction, predictions_history=predictions_history)

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