import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from performance_app.models import PerformanceRecord, EmployeeProfile
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class PerformanceDataProcessor:
    """
    Handles data processing for AI models
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def get_performance_data(self, start_date=None, end_date=None, employee_ids=None):
        """
        Extract and prepare performance data for ML models
        """
        queryset = PerformanceRecord.objects.select_related('employee__user').all()

        if start_date:
            queryset = queryset.filter(performance_start_date__gte=start_date)
        if end_date:
            queryset = queryset.filter(performance_end_date__lte=end_date)
        if employee_ids:
            queryset = queryset.filter(employee_id__in=employee_ids)

        # Convert to DataFrame
        data = []
        for record in queryset:
            data.append({
                'employee_id': record.employee.id,
                'employee_name': str(record.employee),
                'department': record.employee.department,
                'job_title': record.employee.job_title,
                'sales_target': record.sales_target,
                'sales_volume': record.sales_volume,
                'sales_achieved_percent': record.sales_achieved_percent,
                'distribution_target': record.distribution_target,
                'distribution_volume': record.distribution_volume,
                'distribution_achieved_percent': record.distribution_achieved_percent,
                'revenue_target': record.revenue_target,
                'revenue_volume': record.revenue_volume,
                'revenue_achieved_percent': record.revenue_achieved_percent,
                'customer_base_target': record.customer_base_target,
                'customer_base_volume': record.customer_base_volume,
                'customer_base_achieved_percent': record.customer_base_achieved_percent,
                'team_engagement_score': record.team_engagement_score,
                'performance_start_date': record.performance_start_date,
                'performance_end_date': record.performance_end_date,
                'date_hired': record.employee.date_hired,
                'tenure_days': (datetime.now().date() - record.employee.date_hired).days if record.employee.date_hired else 0,
            })

        df = pd.DataFrame(data)
        return df

    def engineer_features(self, df):
        """
        Create additional features for ML models
        """
        # Overall performance score (weighted average)
        weights = {
            'sales': 0.25,
            'distribution': 0.20,
            'revenue': 0.30,
            'customer_base': 0.15,
            'engagement': 0.10
        }

        df['overall_performance_score'] = (
            df['sales_achieved_percent'] * weights['sales'] +
            df['distribution_achieved_percent'] * weights['distribution'] +
            df['revenue_achieved_percent'] * weights['revenue'] +
            df['customer_base_achieved_percent'] * weights['customer_base'] +
            df['team_engagement_score'] * weights['engagement'] / 10  # Normalize to 0-100
        )

        # Performance category
        df['performance_category'] = pd.cut(
            df['overall_performance_score'],
            bins=[0, 60, 80, 100],
            labels=['Poor', 'Good', 'Excellent']
        )

        # Tenure categories
        df['tenure_category'] = pd.cut(
            df['tenure_days'],
            bins=[0, 365, 1095, 2190, float('inf')],
            labels=['<1year', '1-3years', '3-6years', '6+years']
        )

        # Month and quarter features
        df['performance_month'] = pd.to_datetime(df['performance_start_date']).dt.month
        df['performance_quarter'] = pd.to_datetime(df['performance_start_date']).dt.quarter

        return df

    def encode_categorical_features(self, df):
        """
        Encode categorical variables for ML models
        """
        categorical_columns = ['department', 'job_title', 'performance_category', 'tenure_category']

        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].fillna('Unknown'))
            else:
                df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].fillna('Unknown'))

        return df

    def scale_numerical_features(self, df, fit=True):
        """
        Scale numerical features for ML models
        """
        numerical_columns = [
            'sales_achieved_percent', 'distribution_achieved_percent',
            'revenue_achieved_percent', 'customer_base_achieved_percent',
            'team_engagement_score', 'tenure_days'
        ]

        if fit:
            df[numerical_columns] = self.scaler.fit_transform(df[numerical_columns])
        else:
            df[numerical_columns] = self.scaler.transform(df[numerical_columns])

        return df

    def prepare_training_data(self, target_column='overall_performance_score', test_size=0.2):
        """
        Prepare complete training dataset
        """
        # Get raw data
        df = self.get_performance_data()

        if df.empty:
            logger.warning("No performance data available for training")
            return None, None, None, None

        # Engineer features
        df = self.engineer_features(df)

        # Encode categorical
        df = self.encode_categorical_features(df)

        # Scale numerical
        df = self.scale_numerical_features(df)

        # Prepare features and target
        feature_columns = [
            'sales_achieved_percent', 'distribution_achieved_percent',
            'revenue_achieved_percent', 'customer_base_achieved_percent',
            'team_engagement_score', 'tenure_days',
            'department_encoded', 'job_title_encoded', 'tenure_category_encoded',
            'performance_month', 'performance_quarter'
        ]

        X = df[feature_columns]
        y = df[target_column]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        return X_train, X_test, y_train, y_test

    def prepare_prediction_data(self, employee_id, recent_records=None):
        """
        Prepare data for making predictions on a specific employee
        """
        if recent_records is None:
            # Get recent performance records for the employee
            recent_records = PerformanceRecord.objects.filter(
                employee_id=employee_id
            ).order_by('-performance_end_date')[:5]  # Last 5 records

        if not recent_records:
            return None

        # Convert to DataFrame
        data = []
        for record in recent_records:
            data.append({
                'employee_id': record.employee.id,
                'employee_name': str(record.employee),
                'department': record.employee.department,
                'job_title': record.employee.job_title,
                'sales_target': record.sales_target,
                'sales_volume': record.sales_volume,
                'sales_achieved_percent': record.sales_achieved_percent,
                'distribution_target': record.distribution_target,
                'distribution_volume': record.distribution_volume,
                'distribution_achieved_percent': record.distribution_achieved_percent,
                'revenue_target': record.revenue_target,
                'revenue_volume': record.revenue_volume,
                'revenue_achieved_percent': record.revenue_achieved_percent,
                'customer_base_target': record.customer_base_target,
                'customer_base_volume': record.customer_base_volume,
                'customer_base_achieved_percent': record.customer_base_achieved_percent,
                'team_engagement_score': record.team_engagement_score,
                'performance_start_date': record.performance_start_date,
                'performance_end_date': record.performance_end_date,
                'date_hired': record.employee.date_hired,
                'tenure_days': (datetime.now().date() - record.employee.date_hired).days if record.employee.date_hired else 0,
            })

        df = pd.DataFrame(data)

        # Engineer features
        df = self.engineer_features(df)

        # Encode categorical (use existing encoders)
        df = self.encode_categorical_features(df)

        # Scale numerical (use existing scaler)
        df = self.scale_numerical_features(df, fit=False)

        # Average the features for prediction
        feature_columns = [
            'sales_achieved_percent', 'distribution_achieved_percent',
            'revenue_achieved_percent', 'customer_base_achieved_percent',
            'team_engagement_score', 'tenure_days',
            'department_encoded', 'job_title_encoded', 'tenure_category_encoded',
            'performance_month', 'performance_quarter'
        ]

        prediction_data = df[feature_columns].mean().values.reshape(1, -1)
        return prediction_data

    def get_employee_performance_trends(self, employee_id, months=12):
        """
        Get performance trends for trend analysis
        """
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=months*30)

        df = self.get_performance_data(
            start_date=start_date,
            end_date=end_date,
            employee_ids=[employee_id]
        )

        if df.empty:
            return None

        # Sort by date
        df = df.sort_values('performance_start_date')

        # Calculate rolling averages
        df['performance_trend'] = df['overall_performance_score'].rolling(window=3).mean()

        return df