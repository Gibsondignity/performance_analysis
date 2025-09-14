import random
from performance_app.models import EmployeeProfile, PerformanceRecord, Evaluation, PeerReview
from .sentiment_analyzer import SentimentAnalyzer
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class AITextGenerator:
    """
    Generates AI-powered text summaries and recommendations
    """

    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()

    def generate_performance_summary(self, employee_id):
        """
        Generate an AI-powered performance summary
        """
        try:
            employee = EmployeeProfile.objects.get(id=employee_id)

            # Get recent performance data
            recent_records = PerformanceRecord.objects.filter(
                employee=employee
            ).order_by('-performance_end_date')[:3]

            if not recent_records:
                return "Insufficient performance data available for comprehensive analysis."

            # Get evaluations
            evaluations = Evaluation.objects.filter(
                employee=employee
            ).order_by('-date')[:5]

            # Get peer reviews
            peer_reviews = PeerReview.objects.filter(
                reviewee=employee
            ).order_by('-review_date')[:3]

            # Calculate key metrics
            avg_score = sum(record.get_overall_kpi_score() for record in recent_records) / len(recent_records)
            avg_score = round(avg_score, 1)

            # Analyze sentiment
            sentiment_summary = self.sentiment_analyzer.get_employee_feedback_summary(employee_id)

            # Generate summary text
            summary_parts = []

            # Performance level assessment
            if avg_score >= 85:
                performance_level = "exceptional"
                summary_parts.append(f"{employee.first_name} has demonstrated {performance_level} performance with an average score of {avg_score}%.")
            elif avg_score >= 75:
                performance_level = "strong"
                summary_parts.append(f"{employee.first_name} has shown {performance_level} performance with an average score of {avg_score}%.")
            elif avg_score >= 65:
                performance_level = "solid"
                summary_parts.append(f"{employee.first_name} has delivered {performance_level} performance with an average score of {avg_score}%.")
            elif avg_score >= 55:
                performance_level = "adequate"
                summary_parts.append(f"{employee.first_name} has achieved {performance_level} performance with an average score of {avg_score}%.")
            else:
                performance_level = "developing"
                summary_parts.append(f"{employee.first_name} is in a {performance_level} phase with an average score of {avg_score}%.")

            # Trend analysis
            if len(recent_records) >= 2:
                scores = [record.get_overall_kpi_score() for record in recent_records]
                if scores[0] > scores[-1] + 5:
                    summary_parts.append("Performance has shown a declining trend that requires attention.")
                elif scores[-1] > scores[0] + 5:
                    summary_parts.append("Performance has demonstrated positive growth and improvement.")
                else:
                    summary_parts.append("Performance has remained relatively stable over recent periods.")

            # Feedback sentiment
            if sentiment_summary and sentiment_summary['total_feedbacks'] > 0:
                sentiment = sentiment_summary['average_sentiment']
                if sentiment == 'positive':
                    summary_parts.append("Feedback from colleagues and supervisors has been predominantly positive.")
                elif sentiment == 'negative':
                    summary_parts.append("Feedback indicates areas that need focused improvement.")
                else:
                    summary_parts.append("Feedback has been generally balanced with room for growth.")

            # Evaluation participation
            eval_count = evaluations.count()
            peer_count = peer_reviews.count()

            if eval_count > 0 or peer_count > 0:
                participation_parts = []
                if eval_count > 0:
                    participation_parts.append(f"{eval_count} formal evaluation{'s' if eval_count != 1 else ''}")
                if peer_count > 0:
                    participation_parts.append(f"{peer_count} peer review{'s' if peer_count != 1 else ''}")

                if participation_parts:
                    summary_parts.append(f"The employee has participated in {', '.join(participation_parts)}.")

            # Combine all parts
            summary = " ".join(summary_parts)

            return summary

        except Exception as e:
            logger.error(f"Error generating performance summary: {e}")
            return "Unable to generate performance summary due to insufficient data."

    def generate_recommendations(self, employee_id):
        """
        Generate AI-powered recommendations for training and development
        """
        try:
            employee = EmployeeProfile.objects.get(id=employee_id)

            # Get recent performance data
            recent_records = PerformanceRecord.objects.filter(
                employee=employee
            ).order_by('-performance_end_date')[:3]

            if not recent_records:
                return []

            avg_score = sum(record.get_overall_kpi_score() for record in recent_records) / len(recent_records)

            recommendations = []

            # Performance-based recommendations
            if avg_score >= 85:
                recommendations.extend(self._get_high_performer_recommendations(employee))
            elif avg_score >= 70:
                recommendations.extend(self._get_good_performer_recommendations(employee))
            elif avg_score >= 55:
                recommendations.extend(self._get_average_performer_recommendations(employee))
            else:
                recommendations.extend(self._get_developing_performer_recommendations(employee))

            # Role-specific recommendations
            recommendations.extend(self._get_role_specific_recommendations(employee))

            # Randomize order slightly for variety
            random.shuffle(recommendations)

            return recommendations[:5]  # Return top 5 recommendations

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []

    def _get_high_performer_recommendations(self, employee):
        """Recommendations for high performers"""
        return [
            {
                'title': 'Leadership Development Program',
                'description': 'Consider enrolling in advanced leadership training to prepare for senior roles.',
                'priority': 'HIGH',
                'category': 'development',
                'reasoning': 'High performers are often ready for increased responsibilities.'
            },
            {
                'title': 'Mentorship Program',
                'description': 'Participate as a mentor for junior team members to develop leadership skills.',
                'priority': 'MEDIUM',
                'category': 'development',
                'reasoning': 'Sharing expertise helps develop coaching and leadership abilities.'
            },
            {
                'title': 'Cross-functional Project Assignment',
                'description': 'Consider assignments in other departments to broaden experience.',
                'priority': 'MEDIUM',
                'category': 'development',
                'reasoning': 'High performers benefit from diverse experiences that prepare them for advancement.'
            }
        ]

    def _get_good_performer_recommendations(self, employee):
        """Recommendations for good performers"""
        return [
            {
                'title': 'Advanced Technical Training',
                'description': 'Pursue advanced training in your technical specialty.',
                'priority': 'MEDIUM',
                'category': 'training',
                'reasoning': 'Building deeper expertise in your field will enhance performance further.'
            },
            {
                'title': 'Professional Certification',
                'description': 'Consider obtaining relevant industry certifications.',
                'priority': 'HIGH',
                'category': 'certification',
                'reasoning': 'Certifications validate expertise and open new opportunities.'
            },
            {
                'title': 'Project Leadership Role',
                'description': 'Take on project leadership responsibilities to develop management skills.',
                'priority': 'MEDIUM',
                'category': 'development',
                'reasoning': 'Good performers are ready to take on more complex responsibilities.'
            }
        ]

    def _get_average_performer_recommendations(self, employee):
        """Recommendations for average performers"""
        return [
            {
                'title': 'Performance Enhancement Workshop',
                'description': 'Attend workshops focused on productivity and goal achievement.',
                'priority': 'HIGH',
                'category': 'training',
                'reasoning': 'Targeted training can help improve key performance areas.'
            },
            {
                'title': 'Mentoring Session',
                'description': 'Schedule regular mentoring sessions with supervisor.',
                'priority': 'HIGH',
                'category': 'development',
                'reasoning': 'Regular guidance can help address performance gaps.'
            },
            {
                'title': 'Skill Assessment',
                'description': 'Complete a comprehensive skills assessment to identify development areas.',
                'priority': 'MEDIUM',
                'category': 'assessment',
                'reasoning': 'Understanding specific skill gaps enables targeted improvement.'
            }
        ]

    def _get_developing_performer_recommendations(self, employee):
        """Recommendations for developing performers"""
        return [
            {
                'title': 'Fundamental Skills Training',
                'description': 'Focus on building core competencies required for the role.',
                'priority': 'CRITICAL',
                'category': 'training',
                'reasoning': 'Addressing fundamental skill gaps is essential for performance improvement.'
            },
            {
                'title': 'Performance Improvement Plan',
                'description': 'Develop a structured improvement plan with clear milestones.',
                'priority': 'CRITICAL',
                'category': 'development',
                'reasoning': 'Structured guidance is crucial for significant performance improvement.'
            },
            {
                'title': 'Daily Check-ins',
                'description': 'Implement daily progress check-ins with supervisor.',
                'priority': 'HIGH',
                'category': 'supervision',
                'reasoning': 'Regular feedback and support are essential during development phases.'
            }
        ]

    def _get_role_specific_recommendations(self, employee):
        """Role-specific recommendations"""
        recommendations = []

        role = employee.user.role

        if role == 'EMPLOYEE':
            recommendations.append({
                'title': 'Supervisor Communication Training',
                'description': 'Improve communication skills for better collaboration with supervisors.',
                'priority': 'MEDIUM',
                'category': 'soft_skills',
                'reasoning': 'Effective communication is crucial for junior roles.'
            })
        elif role in ['MANAGER', 'MIDDLE_MANAGER']:
            recommendations.append({
                'title': 'Team Management Certification',
                'description': 'Pursue certification in team leadership and management.',
                'priority': 'HIGH',
                'category': 'certification',
                'reasoning': 'Management roles require specific leadership competencies.'
            })
        elif role == 'HR':
            recommendations.append({
                'title': 'HR Analytics Training',
                'description': 'Develop skills in HR data analysis and reporting.',
                'priority': 'MEDIUM',
                'category': 'technical',
                'reasoning': 'HR professionals benefit from analytical skills.'
            })

        return recommendations

    def assess_promotion_readiness(self, employee_id):
        """
        Assess employee's readiness for promotion
        """
        try:
            employee = EmployeeProfile.objects.get(id=employee_id)

            # Get performance data
            records = PerformanceRecord.objects.filter(
                employee=employee
            ).order_by('-performance_end_date')[:6]  # Last 6 months

            if len(records) < 3:
                return {
                    'readiness_score': 0,
                    'assessment': 'Insufficient data for promotion assessment',
                    'factors': []
                }

            # Calculate readiness factors
            avg_score = sum(record.get_overall_kpi_score() for record in records) / len(records)
            consistency = self._calculate_performance_consistency(records)
            tenure_months = (datetime.now().date() - employee.date_hired).days / 30

            # Weighted readiness score
            readiness_score = (
                (avg_score / 100) * 0.5 +      # Performance: 50%
                consistency * 0.3 +             # Consistency: 30%
                min(tenure_months / 24, 1) * 0.2  # Tenure: 20% (max 2 years)
            ) * 100

            readiness_score = round(readiness_score, 1)

            # Generate assessment
            if readiness_score >= 80:
                assessment = "Strong candidate for promotion. Demonstrates consistent high performance and readiness for increased responsibilities."
            elif readiness_score >= 65:
                assessment = "Good potential for promotion. Shows solid performance with some development areas to address."
            elif readiness_score >= 50:
                assessment = "Moderate readiness for promotion. Additional development and consistent performance needed."
            else:
                assessment = "Not ready for promotion. Focus on performance improvement and skill development first."

            factors = [
                f"Average Performance Score: {avg_score:.1f}%",
                f"Performance Consistency: {consistency:.1f}/1.0",
                f"Tenure: {int(tenure_months)} months",
                f"Evaluations Completed: {Evaluation.objects.filter(employee=employee).count()}",
                f"Peer Reviews Received: {PeerReview.objects.filter(reviewee=employee).count()}"
            ]

            return {
                'readiness_score': readiness_score,
                'assessment': assessment,
                'factors': factors,
                'recommendations': self._get_promotion_recommendations(readiness_score)
            }

        except Exception as e:
            logger.error(f"Error assessing promotion readiness: {e}")
            return {
                'readiness_score': 0,
                'assessment': 'Unable to assess promotion readiness',
                'factors': []
            }

    def _calculate_performance_consistency(self, records):
        """Calculate performance consistency score (0-1)"""
        if len(records) < 2:
            return 0.5

        scores = [record.get_overall_kpi_score() for record in records]
        avg_score = sum(scores) / len(scores)

        # Calculate variance (lower variance = higher consistency)
        variance = sum((score - avg_score) ** 2 for score in scores) / len(scores)
        std_dev = variance ** 0.5

        # Convert to consistency score (0-1, where 1 is most consistent)
        consistency = max(0, 1 - (std_dev / 20))  # Assume 20% std dev is very inconsistent

        return round(consistency, 2)

    def _get_promotion_recommendations(self, readiness_score):
        """Get promotion recommendations based on readiness score"""
        if readiness_score >= 80:
            return [
                "Ready for immediate promotion consideration",
                "Consider fast-track development programs",
                "Prepare for increased leadership responsibilities"
            ]
        elif readiness_score >= 65:
            return [
                "Ready for promotion within 3-6 months with development",
                "Focus on leadership skill development",
                "Seek mentorship from senior leaders"
            ]
        elif readiness_score >= 50:
            return [
                "Promotion possible within 6-12 months",
                "Address performance gaps identified",
                "Complete additional training and certifications"
            ]
        else:
            return [
                "Focus on performance improvement first",
                "Complete required training programs",
                "Demonstrate consistent performance over time"
            ]