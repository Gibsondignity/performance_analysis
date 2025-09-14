from textblob import TextBlob
from transformers import pipeline
import logging
from performance_app.models import Evaluation, PeerReview
import numpy as np

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Analyzes sentiment in feedback and evaluation text using TextBlob and Hugging Face
    """

    def __init__(self):
        self.textblob_analyzer = None
        self.huggingface_analyzer = None
        self._initialize_analyzers()

    def _initialize_analyzers(self):
        """
        Initialize sentiment analysis models
        """
        try:
            # Initialize TextBlob (rule-based, fast)
            self.textblob_analyzer = TextBlob

            # Initialize Hugging Face sentiment analysis pipeline
            # Using a lightweight model for efficiency
            self.huggingface_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            logger.info("Sentiment analyzers initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing sentiment analyzers: {e}")
            # Fallback to TextBlob only
            self.huggingface_analyzer = None

    def analyze_text_sentiment(self, text, use_huggingface=True):
        """
        Analyze sentiment of a text using both TextBlob and Hugging Face
        Returns: dict with polarity, subjectivity, and detailed scores
        """
        if not text or not text.strip():
            return {
                'polarity': 0.0,
                'subjectivity': 0.0,
                'sentiment': 'neutral',
                'confidence': 0.0,
                'scores': {}
            }

        results = {}

        # TextBlob analysis (fast, rule-based)
        blob = self.textblob_analyzer(text)
        results['polarity'] = blob.sentiment.polarity  # -1 to 1
        results['subjectivity'] = blob.sentiment.subjectivity  # 0 to 1

        # Determine sentiment category
        if results['polarity'] > 0.1:
            results['sentiment'] = 'positive'
        elif results['polarity'] < -0.1:
            results['sentiment'] = 'negative'
        else:
            results['sentiment'] = 'neutral'

        # Hugging Face analysis (more accurate, slower)
        if use_huggingface and self.huggingface_analyzer:
            try:
                hf_results = self.huggingface_analyzer(text)
                if hf_results and len(hf_results[0]) > 0:
                    scores = {item['label']: item['score'] for item in hf_results[0]}
                    results['scores'] = scores

                    # Use Hugging Face for final sentiment determination
                    if 'LABEL_2' in scores:  # Positive
                        results['sentiment'] = 'positive'
                        results['confidence'] = scores['LABEL_2']
                    elif 'LABEL_0' in scores:  # Negative
                        results['sentiment'] = 'negative'
                        results['confidence'] = scores['LABEL_0']
                    else:  # Neutral
                        results['sentiment'] = 'neutral'
                        results['confidence'] = scores.get('LABEL_1', 0.0)
                else:
                    results['confidence'] = abs(results['polarity'])
                    results['scores'] = {}
            except Exception as e:
                logger.error(f"Error in Hugging Face analysis: {e}")
                results['confidence'] = abs(results['polarity'])
                results['scores'] = {}
        else:
            results['confidence'] = abs(results['polarity'])
            results['scores'] = {}

        return results

    def analyze_evaluation_sentiment(self, evaluation_id):
        """
        Analyze sentiment in an evaluation's feedback
        """
        try:
            evaluation = Evaluation.objects.get(id=evaluation_id)

            # Combine all text fields for analysis
            text_to_analyze = ""
            if evaluation.remarks:
                text_to_analyze += evaluation.remarks + " "
            if evaluation.strengths:
                text_to_analyze += evaluation.strengths + " "
            if evaluation.areas_for_improvement:
                text_to_analyze += evaluation.areas_for_improvement + " "
            if evaluation.goals_achieved:
                text_to_analyze += evaluation.goals_achieved + " "
            if evaluation.development_needs:
                text_to_analyze += evaluation.development_needs + " "

            if not text_to_analyze.strip():
                return None

            sentiment_result = self.analyze_text_sentiment(text_to_analyze)

            # Add evaluation context
            sentiment_result.update({
                'evaluation_id': evaluation_id,
                'evaluation_type': evaluation.evaluation_type,
                'employee_name': str(evaluation.employee),
                'evaluator_name': str(evaluation.evaluator) if evaluation.evaluator else 'Self'
            })

            return sentiment_result

        except Evaluation.DoesNotExist:
            logger.error(f"Evaluation {evaluation_id} not found")
            return None
        except Exception as e:
            logger.error(f"Error analyzing evaluation sentiment: {e}")
            return None

    def analyze_peer_review_sentiment(self, review_id):
        """
        Analyze sentiment in a peer review
        """
        try:
            review = PeerReview.objects.get(id=review_id)

            # Combine feedback text
            text_to_analyze = ""
            if review.strengths:
                text_to_analyze += review.strengths + " "
            if review.areas_for_improvement:
                text_to_analyze += review.areas_for_improvement + " "
            if review.feedback:
                text_to_analyze += review.feedback + " "

            if not text_to_analyze.strip():
                return None

            sentiment_result = self.analyze_text_sentiment(text_to_analyze)

            # Add review context
            sentiment_result.update({
                'review_id': review_id,
                'reviewer_name': str(review.reviewer),
                'reviewee_name': str(review.reviewee),
                'overall_rating': review.overall_rating,
                'is_anonymous': review.is_anonymous
            })

            return sentiment_result

        except PeerReview.DoesNotExist:
            logger.error(f"Peer review {review_id} not found")
            return None
        except Exception as e:
            logger.error(f"Error analyzing peer review sentiment: {e}")
            return None

    def get_employee_feedback_summary(self, employee_id):
        """
        Get sentiment summary for all feedback about an employee
        """
        try:
            # Get all evaluations for the employee
            evaluations = Evaluation.objects.filter(employee_id=employee_id)
            peer_reviews = PeerReview.objects.filter(reviewee_id=employee_id)

            all_sentiments = []

            # Analyze evaluation sentiments
            for evaluation in evaluations:
                sentiment = self.analyze_evaluation_sentiment(evaluation.id)
                if sentiment:
                    all_sentiments.append(sentiment)

            # Analyze peer review sentiments
            for review in peer_reviews:
                sentiment = self.analyze_peer_review_sentiment(review.id)
                if sentiment:
                    all_sentiments.append(sentiment)

            if not all_sentiments:
                return {
                    'total_feedbacks': 0,
                    'average_sentiment': 'neutral',
                    'sentiment_distribution': {},
                    'key_themes': []
                }

            # Calculate summary statistics
            polarities = [s['polarity'] for s in all_sentiments]
            sentiments = [s['sentiment'] for s in all_sentiments]

            sentiment_counts = {
                'positive': sentiments.count('positive'),
                'neutral': sentiments.count('neutral'),
                'negative': sentiments.count('negative')
            }

            # Determine overall sentiment
            avg_polarity = np.mean(polarities)
            if avg_polarity > 0.1:
                overall_sentiment = 'positive'
            elif avg_polarity < -0.1:
                overall_sentiment = 'negative'
            else:
                overall_sentiment = 'neutral'

            return {
                'total_feedbacks': len(all_sentiments),
                'average_sentiment': overall_sentiment,
                'average_polarity': round(avg_polarity, 3),
                'sentiment_distribution': sentiment_counts,
                'sentiment_trend': self._calculate_sentiment_trend(all_sentiments)
            }

        except Exception as e:
            logger.error(f"Error getting employee feedback summary: {e}")
            return None

    def _calculate_sentiment_trend(self, sentiments):
        """
        Calculate sentiment trend over time
        """
        if len(sentiments) < 2:
            return 'stable'

        # Sort by date if available
        try:
            sorted_sentiments = sorted(sentiments, key=lambda x: x.get('date', ''))
            polarities = [s['polarity'] for s in sorted_sentiments]

            # Simple trend calculation
            first_half = polarities[:len(polarities)//2]
            second_half = polarities[len(polarities)//2:]

            first_avg = np.mean(first_half)
            second_avg = np.mean(second_half)

            if second_avg > first_avg + 0.1:
                return 'improving'
            elif second_avg < first_avg - 0.1:
                return 'declining'
            else:
                return 'stable'
        except:
            return 'stable'

    def extract_key_phrases(self, text):
        """
        Extract key phrases from text using TextBlob
        """
        if not text:
            return []

        blob = self.textblob_analyzer(text)
        noun_phrases = list(blob.noun_phrases)

        # Filter and rank phrases
        key_phrases = []
        for phrase in noun_phrases:
            if len(phrase.split()) >= 2:  # At least 2 words
                key_phrases.append(phrase)

        return key_phrases[:10]  # Return top 10