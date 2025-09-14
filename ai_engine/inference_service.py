from .ml_models import PerformanceScorer, PerformancePredictor, AnomalyDetector, EmployeeClusterer, RecommendationEngine
from .data_processor import PerformanceDataProcessor
from .sentiment_analyzer import SentimentAnalyzer
from .text_generator import AITextGenerator
from .models import TrainedModel, Prediction, Recommendation, Anomaly, PerformanceInsight
from performance_app.models import EmployeeProfile, PerformanceRecord, Evaluation, PeerReview
from django.utils import timezone
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class AIService:
    """
    Main service for AI-powered performance analysis
    """

    def __init__(self):
        self.data_processor = PerformanceDataProcessor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.text_generator = AITextGenerator()
        self.scorer = PerformanceScorer()
        self.predictor = PerformancePredictor()
        self.anomaly_detector = AnomalyDetector()
        self.clusterer = EmployeeClusterer()
        self.recommender = RecommendationEngine()

    def load_active_models(self):
        """
        Load the most recent active models for each type
        """
        try:
            # Load scoring model
            scoring_model = TrainedModel.objects.filter(
                model_type='SCORING',
                is_active=True
            ).order_by('-created_at').first()

            if scoring_model:
                self.scorer.load_model(scoring_model.id)
                logger.info(f"Loaded scoring model: {scoring_model.name}")

            # Load prediction model
            prediction_model = TrainedModel.objects.filter(
                model_type='PREDICTION',
                is_active=True
            ).order_by('-created_at').first()

            if prediction_model:
                self.predictor.load_model(prediction_model.id)
                logger.info(f"Loaded prediction model: {prediction_model.name}")

            # Load anomaly model
            anomaly_model = TrainedModel.objects.filter(
                model_type='ANOMALY',
                is_active=True
            ).order_by('-created_at').first()

            if anomaly_model:
                self.anomaly_detector.load_model(anomaly_model.id)
                logger.info(f"Loaded anomaly model: {anomaly_model.name}")

            # Load clustering model
            clustering_model = TrainedModel.objects.filter(
                model_type='CLUSTERING',
                is_active=True
            ).order_by('-created_at').first()

            if clustering_model:
                self.clusterer.load_model(clustering_model.id)
                logger.info(f"Loaded clustering model: {clustering_model.name}")

        except Exception as e:
            logger.error(f"Error loading models: {e}")

    def analyze_employee_performance(self, employee_id, user=None):
        """
        Comprehensive AI analysis for an employee
        """
        try:
            # Get employee data
            employee = EmployeeProfile.objects.get(id=employee_id)

            # Prepare prediction data
            prediction_data = self.data_processor.prepare_prediction_data(employee_id)

            if prediction_data is None:
                return {
                    'error': 'Insufficient performance data for analysis'
                }

            results = {
                'employee_id': employee_id,
                'employee_name': str(employee),
                'analysis_timestamp': timezone.now(),
                'performance_score': None,
                'ai_performance_summary': '',
                'ai_recommendations': [],
                'promotion_readiness': {},
                'predictions': [],
                'recommendations': [],
                'anomalies': [],
                'insights': [],
                'sentiment_analysis': {},
                'feedback_summary': {}
            }

            # Generate performance score
            if self.scorer.model:
                try:
                    score = self.scorer.predict_score(prediction_data)[0]
                    results['performance_score'] = round(score, 2)

                    # Save prediction
                    Prediction.objects.create(
                        employee=employee,
                        prediction_type='performance_score',
                        predicted_value=score,
                        prediction_date=timezone.now().date(),
                        model_used=self.scorer.model if hasattr(self.scorer, 'model') else None
                    )
                except Exception as e:
                    logger.error(f"Error generating performance score: {e}")

            # Generate predictions
            if self.predictor.model:
                try:
                    future_score = self.predictor.predict_future_performance(prediction_data)
                    results['predictions'].append({
                        'type': 'next_period_performance',
                        'value': round(future_score, 2),
                        'description': 'Predicted performance for next period'
                    })

                    # Save prediction
                    Prediction.objects.create(
                        employee=employee,
                        prediction_type='future_performance',
                        predicted_value=future_score,
                        prediction_date=timezone.now().date() + timedelta(days=30),
                        model_used=self.predictor.model if hasattr(self.predictor, 'model') else None
                    )
                except Exception as e:
                    logger.error(f"Error generating predictions: {e}")

            # Check for anomalies
            if self.anomaly_detector.model:
                try:
                    anomaly_predictions, anomaly_scores = self.anomaly_detector.detect_anomalies(prediction_data)

                    if anomaly_predictions[0] == -1:  # Anomaly detected
                        severity = abs(anomaly_scores[0])
                        anomaly = Anomaly.objects.create(
                            employee=employee,
                            anomaly_type='PERFORMANCE_ANOMALY',
                            description=f'Anomaly detected in performance metrics (severity: {severity:.2f})',
                            severity_score=severity,
                            model_used=self.anomaly_detector.model if hasattr(self.anomaly_detector, 'model') else None
                        )

                        results['anomalies'].append({
                            'type': anomaly.anomaly_type,
                            'description': anomaly.description,
                            'severity': anomaly.severity_score
                        })
                except Exception as e:
                    logger.error(f"Error detecting anomalies: {e}")

            # Generate recommendations
            try:
                trends = self.data_processor.get_employee_performance_trends(employee_id)
                recommendations = self.recommender.generate_recommendations(
                    employee_id,
                    results.get('performance_score', 0),
                    trends
                )

                for rec_data in recommendations:
                    recommendation = Recommendation.objects.create(
                        employee=employee,
                        title=rec_data['title'],
                        description=rec_data['description'],
                        priority=rec_data['priority'],
                        category=rec_data['category'],
                        actionable_steps=rec_data['actionable_steps'],
                        created_by=user
                    )

                    results['recommendations'].append({
                        'title': recommendation.title,
                        'description': recommendation.description,
                        'priority': recommendation.priority,
                        'category': recommendation.category
                    })
            except Exception as e:
                logger.error(f"Error generating recommendations: {e}")

            # Generate insights
            try:
                employee_df = self.data_processor.get_performance_data(employee_ids=[employee_id])
                if not employee_df.empty:
                    employee_df = self.data_processor.engineer_features(employee_df)
                    insights = self.recommender.get_performance_insights(employee_df)

                    for insight_data in insights:
                        insight = PerformanceInsight.objects.create(
                            title=insight_data['title'],
                            insight_type=insight_data['type'],
                            content=insight_data['content'],
                            affected_employees=[employee],
                            generated_by=user
                        )

                        results['insights'].append({
                            'title': insight.title,
                            'content': insight.content,
                            'type': insight.insight_type
                        })
            except Exception as e:
                logger.error(f"Error generating insights: {e}")

            # Perform sentiment analysis on feedback
            try:
                feedback_summary = self.sentiment_analyzer.get_employee_feedback_summary(employee_id)
                if feedback_summary:
                    results['feedback_summary'] = feedback_summary

                    # Analyze recent evaluations for detailed sentiment
                    recent_evaluations = Evaluation.objects.filter(
                        employee=employee
                    ).order_by('-date')[:5]

                    sentiment_details = []
                    for evaluation in recent_evaluations:
                        sentiment = self.sentiment_analyzer.analyze_evaluation_sentiment(evaluation.id)
                        if sentiment:
                            sentiment_details.append({
                                'evaluation_type': evaluation.evaluation_type,
                                'date': evaluation.date,
                                'sentiment': sentiment['sentiment'],
                                'polarity': sentiment['polarity'],
                                'confidence': sentiment['confidence']
                            })

                    results['sentiment_analysis'] = {
                        'overall_summary': feedback_summary,
                        'recent_evaluations': sentiment_details
                    }
            except Exception as e:
                logger.error(f"Error performing sentiment analysis: {e}")
                results['sentiment_analysis'] = {'error': str(e)}

            # Generate AI-powered performance summary
            try:
                ai_summary = self.text_generator.generate_performance_summary(employee_id)
                results['ai_performance_summary'] = ai_summary
            except Exception as e:
                logger.error(f"Error generating AI performance summary: {e}")
                results['ai_performance_summary'] = "AI summary generation failed."

            # Generate AI-powered recommendations
            try:
                ai_recommendations = self.text_generator.generate_recommendations(employee_id)
                results['ai_recommendations'] = ai_recommendations
            except Exception as e:
                logger.error(f"Error generating AI recommendations: {e}")
                results['ai_recommendations'] = []

            # Assess promotion readiness
            try:
                promotion_assessment = self.text_generator.assess_promotion_readiness(employee_id)
                results['promotion_readiness'] = promotion_assessment
            except Exception as e:
                logger.error(f"Error assessing promotion readiness: {e}")
                results['promotion_readiness'] = {
                    'readiness_score': 0,
                    'assessment': 'Unable to assess promotion readiness',
                    'factors': []
                }

            return results

        except Exception as e:
            logger.error(f"Error in employee performance analysis: {e}")
            return {
                'error': f'Analysis failed: {str(e)}'
            }

    def analyze_department_performance(self, department, user=None):
        """
        AI analysis for department-level performance
        """
        try:
            # Get department employees
            employees = EmployeeProfile.objects.filter(department=department)

            if not employees:
                return {'error': 'No employees found in department'}

            department_results = {
                'department': department,
                'analysis_timestamp': timezone.now(),
                'total_employees': employees.count(),
                'performance_distribution': {},
                'top_performers': [],
                'needs_attention': [],
                'department_insights': []
            }

            employee_scores = []

            for employee in employees:
                analysis = self.analyze_employee_performance(employee.id, user)
                if 'performance_score' in analysis and analysis['performance_score']:
                    employee_scores.append({
                        'employee': str(employee),
                        'score': analysis['performance_score']
                    })

            if employee_scores:
                scores_only = [emp['score'] for emp in employee_scores]

                # Performance distribution
                department_results['performance_distribution'] = {
                    'excellent': len([s for s in scores_only if s >= 80]),
                    'good': len([s for s in scores_only if 60 <= s < 80]),
                    'needs_improvement': len([s for s in scores_only if s < 60]),
                    'average_score': round(sum(scores_only) / len(scores_only), 2)
                }

                # Top performers
                top_performers = sorted(employee_scores, key=lambda x: x['score'], reverse=True)[:3]
                department_results['top_performers'] = top_performers

                # Needs attention
                needs_attention = [emp for emp in employee_scores if emp['score'] < 60]
                department_results['needs_attention'] = needs_attention

            # Department insights
            try:
                dept_data = self.data_processor.get_performance_data()
                if not dept_data.empty:
                    dept_data = dept_data[dept_data['department'] == department]
                    if not dept_data.empty:
                        dept_data = self.data_processor.engineer_features(dept_data)

                        # Department vs company average
                        company_avg = self.data_processor.get_performance_data()['overall_performance_score'].mean()
                        dept_avg = dept_data['overall_performance_score'].mean()

                        insight = PerformanceInsight.objects.create(
                            title=f'{department} Department Analysis',
                            insight_type='COMPARISON',
                            content=f'Department performance average: {dept_avg:.1f}% vs Company average: {company_avg:.1f}%',
                            affected_employees=list(employees),
                            generated_by=user
                        )

                        department_results['department_insights'].append({
                            'title': insight.title,
                            'content': insight.content
                        })
            except Exception as e:
                logger.error(f"Error generating department insights: {e}")

            return department_results

        except Exception as e:
            logger.error(f"Error in department performance analysis: {e}")
            return {
                'error': f'Department analysis failed: {str(e)}'
            }

    def get_employee_cluster(self, employee_id):
        """
        Get employee's performance cluster
        """
        try:
            if not self.clusterer.model:
                return None

            prediction_data = self.data_processor.prepare_prediction_data(employee_id)
            if prediction_data is None:
                return None

            cluster = self.clusterer.cluster_employees(prediction_data)[0]
            return cluster

        except Exception as e:
            logger.error(f"Error getting employee cluster: {e}")
            return None

    def get_performance_trends(self, employee_id, months=12):
        """
        Get performance trends for an employee
        """
        try:
            trends = self.data_processor.get_employee_performance_trends(employee_id, months)
            return trends

        except Exception as e:
            logger.error(f"Error getting performance trends: {e}")
            return None