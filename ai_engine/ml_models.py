from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import joblib
from .data_processor import PerformanceDataProcessor
from .models import TrainedModel
import logging

logger = logging.getLogger(__name__)

class PerformanceScorer:
    """
    ML-based performance scoring model
    """

    def __init__(self):
        self.model = None
        self.data_processor = PerformanceDataProcessor()

    def train_model(self, X_train, y_train):
        """
        Train the performance scoring model
        """
        # Try Random Forest first
        rf_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )

        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10]
        }

        grid_search = GridSearchCV(
            rf_model, param_grid, cv=3, scoring='r2', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        self.model = grid_search.best_estimator_
        logger.info(f"Best parameters: {grid_search.best_params_}")

        return self.model

    def predict_score(self, X):
        """
        Predict performance score
        """
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict(X)

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model performance
        """
        if self.model is None:
            raise ValueError("Model not trained")

        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return {
            'mse': mse,
            'r2_score': r2,
            'rmse': np.sqrt(mse)
        }

    def save_model(self, name, version, metadata=None):
        """
        Save trained model to database
        """
        if self.model is None:
            raise ValueError("No model to save")

        trained_model = TrainedModel.objects.create(
            name=name,
            model_type='SCORING',
            algorithm='RandomForestRegressor',
            version=version,
            metadata=metadata or {}
        )

        trained_model.save_model(self.model)
        trained_model.save()

        return trained_model

    def load_model(self, model_id):
        """
        Load trained model from database
        """
        trained_model = TrainedModel.objects.get(id=model_id)
        self.model = trained_model.load_model()
        return self.model

class PerformancePredictor:
    """
    Predict future performance using time series and regression
    """

    def __init__(self):
        self.model = None
        self.data_processor = PerformanceDataProcessor()

    def train_predictive_model(self, X_train, y_train):
        """
        Train model for performance prediction
        """
        # Use Linear Regression for prediction
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

        return self.model

    def predict_future_performance(self, employee_data):
        """
        Predict future performance for an employee
        """
        if self.model is None:
            raise ValueError("Model not trained")

        prediction = self.model.predict(employee_data)
        return prediction[0]

    def save_model(self, name, version, metadata=None):
        """
        Save predictive model
        """
        if self.model is None:
            raise ValueError("No model to save")

        trained_model = TrainedModel.objects.create(
            name=name,
            model_type='PREDICTION',
            algorithm='LinearRegression',
            version=version,
            metadata=metadata or {}
        )

        trained_model.save_model(self.model)
        trained_model.save()

        return trained_model

class AnomalyDetector:
    """
    Detect performance anomalies using Isolation Forest
    """

    def __init__(self):
        self.model = None
        self.data_processor = PerformanceDataProcessor()

    def train_anomaly_model(self, X_train):
        """
        Train anomaly detection model
        """
        self.model = IsolationForest(
            n_estimators=100,
            contamination=0.1,  # Expected percentage of anomalies
            random_state=42
        )

        self.model.fit(X_train)
        return self.model

    def detect_anomalies(self, X):
        """
        Detect anomalies in performance data
        Returns: -1 for anomaly, 1 for normal
        """
        if self.model is None:
            raise ValueError("Model not trained")

        predictions = self.model.predict(X)
        scores = self.model.decision_function(X)

        return predictions, scores

    def save_model(self, name, version, metadata=None):
        """
        Save anomaly detection model
        """
        if self.model is None:
            raise ValueError("No model to save")

        trained_model = TrainedModel.objects.create(
            name=name,
            model_type='ANOMALY',
            algorithm='IsolationForest',
            version=version,
            metadata=metadata or {}
        )

        trained_model.save_model(self.model)
        trained_model.save()

        return trained_model

class EmployeeClusterer:
    """
    Cluster employees based on performance patterns
    """

    def __init__(self):
        self.model = None
        self.data_processor = PerformanceDataProcessor()

    def train_clustering_model(self, X_train, n_clusters=4):
        """
        Train K-means clustering model
        """
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )

        self.model.fit(X_train)
        return self.model

    def cluster_employees(self, X):
        """
        Assign employees to clusters
        """
        if self.model is None:
            raise ValueError("Model not trained")

        clusters = self.model.predict(X)
        return clusters

    def evaluate_clustering(self, X):
        """
        Evaluate clustering quality
        """
        if self.model is None:
            raise ValueError("Model not trained")

        clusters = self.model.predict(X)
        silhouette_avg = silhouette_score(X, clusters)

        return {
            'silhouette_score': silhouette_avg,
            'n_clusters': self.model.n_clusters,
            'cluster_centers': self.model.cluster_centers_.tolist()
        }

    def save_model(self, name, version, metadata=None):
        """
        Save clustering model
        """
        if self.model is None:
            raise ValueError("No model to save")

        trained_model = TrainedModel.objects.create(
            name=name,
            model_type='CLUSTERING',
            algorithm='KMeans',
            version=version,
            metadata=metadata or {}
        )

        trained_model.save_model(self.model)
        trained_model.save()

        return trained_model

class RecommendationEngine:
    """
    Generate personalized recommendations based on performance data
    """

    def __init__(self):
        self.data_processor = PerformanceDataProcessor()

    def generate_recommendations(self, employee_id, performance_score, recent_trends):
        """
        Generate recommendations based on performance analysis
        """
        recommendations = []

        # Analyze performance score
        if performance_score < 60:
            recommendations.append({
                'title': 'Performance Improvement Plan',
                'description': 'Your recent performance metrics indicate areas for improvement. Consider focusing on sales targets and customer engagement.',
                'priority': 'HIGH',
                'category': 'development',
                'actionable_steps': [
                    'Review sales techniques and strategies',
                    'Schedule training sessions',
                    'Set up regular performance check-ins'
                ]
            })

        elif performance_score < 80:
            recommendations.append({
                'title': 'Enhance Customer Base Growth',
                'description': 'Good performance overall, but customer base expansion could be accelerated.',
                'priority': 'MEDIUM',
                'category': 'sales',
                'actionable_steps': [
                    'Analyze successful customer acquisition strategies',
                    'Network with potential clients',
                    'Implement customer retention programs'
                ]
            })

        # Analyze trends
        if recent_trends and len(recent_trends) > 1:
            trend_direction = recent_trends.iloc[-1]['performance_trend'] - recent_trends.iloc[0]['performance_trend']

            if trend_direction < -5:
                recommendations.append({
                    'title': 'Address Declining Performance Trend',
                    'description': 'Your performance has been declining recently. Early intervention can help reverse this trend.',
                    'priority': 'CRITICAL',
                    'category': 'development',
                    'actionable_steps': [
                        'Identify factors contributing to decline',
                        'Seek mentorship or additional training',
                        'Adjust work strategies and priorities'
                    ]
                })

        # Engagement-based recommendations
        avg_engagement = recent_trends['team_engagement_score'].mean() if recent_trends is not None else 0

        if avg_engagement < 6:
            recommendations.append({
                'title': 'Boost Team Engagement',
                'description': 'Team engagement scores are below optimal levels. Focus on team collaboration and motivation.',
                'priority': 'MEDIUM',
                'category': 'engagement',
                'actionable_steps': [
                    'Participate in team-building activities',
                    'Provide feedback on workplace improvements',
                    'Seek opportunities for skill development'
                ]
            })

        return recommendations

    def get_performance_insights(self, employee_data):
        """
        Generate overall performance insights
        """
        insights = []

        # Department comparison
        dept_avg = employee_data.groupby('department')['overall_performance_score'].mean()
        employee_dept = employee_data['department'].iloc[0]
        employee_score = employee_data['overall_performance_score'].mean()

        dept_comparison = "above" if employee_score > dept_avg[employee_dept] else "below"

        insights.append({
            'title': f'Department Performance Comparison',
            'content': f'Your performance is {dept_comparison} the department average. Department avg: {dept_avg[employee_dept]:.1f}%, Your avg: {employee_score:.1f}%',
            'type': 'COMPARISON'
        })

        # Trend analysis
        if len(employee_data) > 3:
            recent_scores = employee_data['overall_performance_score'].tail(3)
            trend = "improving" if recent_scores.iloc[-1] > recent_scores.iloc[0] else "declining"

            insights.append({
                'title': 'Performance Trend Analysis',
                'content': f'Your performance has been {trend} over the last 3 periods. Latest score: {recent_scores.iloc[-1]:.1f}%',
                'type': 'TREND'
            })

        return insights