from django.core.management.base import BaseCommand
from ai_engine.ml_models import PerformanceScorer, PerformancePredictor, AnomalyDetector, EmployeeClusterer
from ai_engine.data_processor import PerformanceDataProcessor
from ai_engine.models import TrainedModel
from performance_app.models import PerformanceRecord
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Train AI models for performance analysis'

    def add_arguments(self, parser):
        parser.add_argument(
            '--model-type',
            type=str,
            choices=['all', 'scoring', 'prediction', 'anomaly', 'clustering'],
            default='all',
            help='Type of model to train (default: all)'
        )
        parser.add_argument(
            '--version',
            type=str,
            default='1.0.0',
            help='Version number for the trained models'
        )
        parser.add_argument(
            '--test-size',
            type=float,
            default=0.2,
            help='Test size for train/test split (default: 0.2)'
        )

    def handle(self, *args, **options):
        model_type = options['model_type']
        version = options['version']
        test_size = options['test_size']

        # Check if there's enough data
        record_count = PerformanceRecord.objects.count()
        if record_count < 10:
            self.stdout.write(
                self.style.WARNING(f'Only {record_count} performance records found. Need at least 10 for training.')
            )
            return

        self.stdout.write(f'Found {record_count} performance records for training.')

        # Initialize components
        data_processor = PerformanceDataProcessor()

        # Prepare training data
        self.stdout.write('Preparing training data...')
        X_train, X_test, y_train, y_test = data_processor.prepare_training_data(test_size=test_size)

        if X_train is None:
            self.stdout.write(
                self.style.ERROR('Failed to prepare training data. Check data quality.')
            )
            return

        self.stdout.write(f'Training data prepared: {X_train.shape[0]} samples, {X_train.shape[1]} features')

        # Train models based on type
        if model_type in ['all', 'scoring']:
            self.train_scoring_model(X_train, X_test, y_train, y_test, version)

        if model_type in ['all', 'prediction']:
            self.train_prediction_model(X_train, X_test, y_train, y_test, version)

        if model_type in ['all', 'anomaly']:
            self.train_anomaly_model(X_train, X_test, version)

        if model_type in ['all', 'clustering']:
            self.train_clustering_model(X_train, version)

        self.stdout.write(
            self.style.SUCCESS('AI model training completed successfully!')
        )

    def train_scoring_model(self, X_train, X_test, y_train, y_test, version):
        """Train performance scoring model"""
        self.stdout.write('Training performance scoring model...')

        try:
            scorer = PerformanceScorer()
            model = scorer.train_model(X_train, y_train)

            # Evaluate model
            metrics = scorer.evaluate_model(X_test, y_test)
            self.stdout.write(f'Model evaluation - R²: {metrics["r2_score"]:.3f}, RMSE: {metrics["rmse"]:.3f}')

            # Save model
            metadata = {
                'training_samples': X_train.shape[0],
                'test_samples': X_test.shape[0],
                'features': X_train.shape[1],
                'evaluation_metrics': metrics
            }

            trained_model = scorer.save_model(
                name='Performance Scorer',
                version=version,
                metadata=metadata
            )

            self.stdout.write(
                self.style.SUCCESS(f'Performance scoring model saved: {trained_model.name} v{trained_model.version}')
            )

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error training scoring model: {e}')
            )

    def train_prediction_model(self, X_train, X_test, y_train, y_test, version):
        """Train performance prediction model"""
        self.stdout.write('Training performance prediction model...')

        try:
            predictor = PerformancePredictor()
            model = predictor.train_predictive_model(X_train, y_train)

            # Save model
            metadata = {
                'training_samples': X_train.shape[0],
                'features': X_train.shape[1]
            }

            trained_model = predictor.save_model(
                name='Performance Predictor',
                version=version,
                metadata=metadata
            )

            self.stdout.write(
                self.style.SUCCESS(f'Performance prediction model saved: {trained_model.name} v{trained_model.version}')
            )

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error training prediction model: {e}')
            )

    def train_anomaly_model(self, X_train, X_test, version):
        """Train anomaly detection model"""
        self.stdout.write('Training anomaly detection model...')

        # Combine train and test for unsupervised learning
        X_combined = X_train  # Use only training data for anomaly detection

        try:
            detector = AnomalyDetector()
            model = detector.train_anomaly_model(X_combined)

            # Save model
            metadata = {
                'training_samples': X_combined.shape[0],
                'features': X_combined.shape[1]
            }

            trained_model = detector.save_model(
                name='Anomaly Detector',
                version=version,
                metadata=metadata
            )

            self.stdout.write(
                self.style.SUCCESS(f'Anomaly detection model saved: {trained_model.name} v{trained_model.version}')
            )

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error training anomaly model: {e}')
            )

    def train_clustering_model(self, X_train, version):
        """Train employee clustering model"""
        self.stdout.write('Training employee clustering model...')

        try:
            clusterer = EmployeeClusterer()
            model = clusterer.train_clustering_model(X_train)

            # Evaluate clustering
            metrics = clusterer.evaluate_clustering(X_train)
            self.stdout.write(f'Clustering evaluation - Silhouette Score: {metrics["silhouette_score"]:.3f}')

            # Save model
            metadata = {
                'training_samples': X_train.shape[0],
                'features': X_train.shape[1],
                'n_clusters': metrics['n_clusters'],
                'evaluation_metrics': metrics
            }

            trained_model = clusterer.save_model(
                name='Employee Clusterer',
                version=version,
                metadata=metadata
            )

            self.stdout.write(
                self.style.SUCCESS(f'Employee clustering model saved: {trained_model.name} v{trained_model.version}')
            )

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error training clustering model: {e}')
            )