from django.db import models
from django.contrib.auth import get_user_model
from performance_app.models import EmployeeProfile, PerformanceRecord
import pickle
import json

User = get_user_model()

class TrainedModel(models.Model):
    """
    Stores trained ML models with metadata
    """
    MODEL_TYPES = [
        ('SCORING', 'Performance Scoring'),
        ('PREDICTION', 'Performance Prediction'),
        ('CLUSTERING', 'Employee Clustering'),
        ('ANOMALY', 'Anomaly Detection'),
    ]
    
    name = models.CharField(max_length=100)
    model_type = models.CharField(max_length=20, choices=MODEL_TYPES)
    algorithm = models.CharField(max_length=50)  # e.g., 'RandomForest', 'LinearRegression'
    version = models.CharField(max_length=20)
    model_data = models.BinaryField()  # Pickled model
    metadata = models.JSONField(default=dict)  # Training parameters, metrics
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.name} v{self.version} ({self.model_type})"
    
    def save_model(self, model):
        """Serialize and save the model"""
        self.model_data = pickle.dumps(model)
    
    def load_model(self):
        """Deserialize and return the model"""
        return pickle.loads(self.model_data)

class Prediction(models.Model):
    """
    Stores AI-generated predictions for employees
    """
    employee = models.ForeignKey(EmployeeProfile, on_delete=models.CASCADE)
    prediction_type = models.CharField(max_length=50)  # e.g., 'performance_score', 'engagement'
    predicted_value = models.FloatField()
    confidence_score = models.FloatField(null=True, blank=True)  # 0-1
    prediction_date = models.DateField()
    based_on_records = models.ManyToManyField(PerformanceRecord)
    model_used = models.ForeignKey(TrainedModel, on_delete=models.SET_NULL, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Prediction for {self.employee}: {self.prediction_type} = {self.predicted_value}"

class Recommendation(models.Model):
    """
    Stores personalized recommendations for employees
    """
    PRIORITY_LEVELS = [
        ('LOW', 'Low'),
        ('MEDIUM', 'Medium'),
        ('HIGH', 'High'),
        ('CRITICAL', 'Critical'),
    ]
    
    employee = models.ForeignKey(EmployeeProfile, on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    description = models.TextField()
    priority = models.CharField(max_length=10, choices=PRIORITY_LEVELS, default='MEDIUM')
    category = models.CharField(max_length=50)  # e.g., 'sales', 'engagement', 'development'
    actionable_steps = models.JSONField(default=list)  # List of steps to take
    expected_impact = models.TextField(blank=True)
    is_implemented = models.BooleanField(default=False)
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Recommendation for {self.employee}: {self.title}"

class Anomaly(models.Model):
    """
    Tracks unusual performance patterns
    """
    ANOMALY_TYPES = [
        ('PERFORMANCE_SPIKE', 'Performance Spike'),
        ('PERFORMANCE_DROP', 'Performance Drop'),
        ('ENGAGEMENT_CHANGE', 'Engagement Change'),
        ('TREND_BREAK', 'Trend Break'),
    ]
    
    employee = models.ForeignKey(EmployeeProfile, on_delete=models.CASCADE)
    anomaly_type = models.CharField(max_length=20, choices=ANOMALY_TYPES)
    description = models.TextField()
    severity_score = models.FloatField()  # 0-1
    detected_at = models.DateTimeField(auto_now_add=True)
    related_records = models.ManyToManyField(PerformanceRecord)
    model_used = models.ForeignKey(TrainedModel, on_delete=models.SET_NULL, null=True)
    is_acknowledged = models.BooleanField(default=False)
    
    def __str__(self):
        return f"Anomaly for {self.employee}: {self.anomaly_type}"

class PerformanceInsight(models.Model):
    """
    AI-generated analysis reports and insights
    """
    INSIGHT_TYPES = [
        ('TREND', 'Trend Analysis'),
        ('COMPARISON', 'Comparative Analysis'),
        ('FORECAST', 'Performance Forecast'),
        ('CLUSTER', 'Cluster Analysis'),
    ]
    
    title = models.CharField(max_length=200)
    insight_type = models.CharField(max_length=20, choices=INSIGHT_TYPES)
    content = models.TextField()
    key_findings = models.JSONField(default=list)
    affected_employees = models.ManyToManyField(EmployeeProfile)
    generated_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.insight_type}: {self.title}"
