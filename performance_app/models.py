from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin
from django.core.validators import MinValueValidator, MaxValueValidator

class CustomUserManager(BaseUserManager):
    def create_user(self, employee_id=None, password=None, **extra_fields):
        if not employee_id:
            role = extra_fields.get('role')
            if role:
                employee_id = User.generate_employee_id(role)
            else:
                raise ValueError('Role must be provided if employee_id is not')
        user = self.model(employee_id=employee_id, **extra_fields)
        user.set_password(password)
        user.save()
        return user

    def create_superuser(self, employee_id=None, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        return self.create_user(employee_id, password, **extra_fields)


class User(AbstractBaseUser, PermissionsMixin):
    ROLE_CHOICES = [
        ('HR', 'HR'),
        ('HIGH_MANAGER', 'High Level Manager'),
        ('MIDDLE_MANAGER', 'Middle Level Manager'),
        ('MANAGER', 'Low Level Manager'),
        ('EMPLOYEE', 'Employee'),
    ]

    employee_id = models.CharField(max_length=20, unique=True)
    role = models.CharField(max_length=20, choices=ROLE_CHOICES)
    branch = models.CharField(max_length=100, blank=True, null=True)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)

    USERNAME_FIELD = 'employee_id'
    REQUIRED_FIELDS = []

    objects = CustomUserManager()

    def __str__(self):
        return f"{self.employee_id} - {self.role}"

    @staticmethod
    def generate_employee_id(role):
        prefix = {
            'HR': 'HR',
            'HIGH_MANAGER': 'HM',
            'MIDDLE_MANAGER': 'MM',
            'MANAGER': 'MG',
            'EMPLOYEE': 'EM'
        }.get(role, 'EM')

        # Find the last employee_id with this prefix
        last_id = User.objects.filter(employee_id__startswith=prefix).order_by('-employee_id').first()
        if last_id:
            # Extract the number part after the prefix
            num_str = last_id.employee_id[len(prefix):]
            try:
                num = int(num_str) + 1
            except ValueError:
                # If parsing fails, start from 1
                num = 1
        else:
            num = 1
        return f"{prefix}{num:04d}"



class EmployeeProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    first_name = models.CharField(max_length=100, blank=True, null=True)
    last_name = models.CharField(max_length=100, blank=True, null=True)
    department = models.CharField(max_length=100)
    job_title = models.CharField(max_length=100)
    date_hired = models.DateField()
    phone = models.CharField(max_length=15, blank=True, null=True)
    address = models.TextField(null=True, blank=True)
    emergency_contact_name = models.CharField(max_length=100, null=True, blank=True)
    emergency_contact_phone = models.CharField(max_length=15, null=True, blank=True)
    emergency_relationship = models.CharField(max_length=50, null=True, blank=True)
    data_created = models.DateTimeField(auto_now_add=True, blank=True, null=True)
    data_updated = models.DateTimeField(auto_now=True, blank=True, null=True)
    created_by = models.ForeignKey(User, related_name='created_employee', on_delete=models.CASCADE, null=True, blank=True)
    updated_by = models.ForeignKey(User, related_name='updated_employee', on_delete=models.CASCADE, null=True, blank=True)

    def __str__(self):
        return f"{self.user.employee_id} - {self.first_name} {self.last_name}"

class KPI(models.Model):
    """
    Defines KPIs for different roles
    """
    KPI_TYPES = [
        ('QUANTITATIVE', 'Quantitative'),
        ('QUALITATIVE', 'Qualitative'),
        ('BOOLEAN', 'Boolean'),
    ]

    role = models.CharField(max_length=20, choices=User.ROLE_CHOICES)
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    kpi_type = models.CharField(max_length=20, choices=KPI_TYPES, default='QUANTITATIVE')
    unit = models.CharField(max_length=50, blank=True)  # e.g., 'USD', 'hours', 'percentage'
    target_direction = models.CharField(max_length=10, choices=[('HIGHER', 'Higher is Better'), ('LOWER', 'Lower is Better')], default='HIGHER')
    weight = models.DecimalField(max_digits=5, decimal_places=2, default=1.00, validators=[MinValueValidator(0), MaxValueValidator(100)])
    is_active = models.BooleanField(default=True)
    data_created = models.DateTimeField(auto_now_add=True)
    data_updated = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(User, related_name='created_kpis', on_delete=models.CASCADE, null=True, blank=True)
    updated_by = models.ForeignKey(User, related_name='updated_kpis', on_delete=models.CASCADE, null=True, blank=True)

    class Meta:
        unique_together = ('role', 'name')
        verbose_name = 'KPI'
        verbose_name_plural = 'KPIs'

    def __str__(self):
        return f"{self.role} - {self.name} ({self.weight}%)"


class PerformanceRecord(models.Model):
    employee = models.ForeignKey(EmployeeProfile, on_delete=models.CASCADE)

    # Legacy fields for backward compatibility (can be removed later)
    sales_target = models.FloatField(default=0.00, blank=True, null=True)
    sales_volume = models.FloatField(default=0.00, blank=True, null=True)
    distribution_target = models.FloatField(default=0.00, blank=True, null=True)
    distribution_volume = models.FloatField(default=0.00, blank=True, null=True)
    revenue_target = models.FloatField(default=0.00, blank=True, null=True)
    revenue_volume = models.FloatField(default=0.00, blank=True, null=True)
    customer_base_target = models.FloatField(default=0.00, blank=True, null=True)
    customer_base_volume = models.FloatField(default=0.00, blank=True, null=True)
    team_engagement_score = models.IntegerField(default=0, blank=True, null=True)  # e.g., from 1 to 10

    # NEW: Flexible KPI values stored as JSON
    kpi_values = models.JSONField(default=dict)  # {'kpi_name': {'target': value, 'actual': value}}

    # Date range of performance period
    performance_start_date = models.DateField(blank=True, null=True)
    performance_end_date = models.DateField(blank=True, null=True)

    # Existing timestamps
    data_created = models.DateTimeField(auto_now_add=True, blank=True, null=True)
    data_updated = models.DateTimeField(auto_now=True, blank=True, null=True)
    created_by = models.ForeignKey(User, related_name='created_records', on_delete=models.CASCADE, null=True, blank=True)
    updated_by = models.ForeignKey(User, related_name='updated_records', on_delete=models.CASCADE, null=True, blank=True)

    def get_kpi_score(self, kpi_name):
        """Calculate score for a specific KPI"""
        if kpi_name in self.kpi_values:
            kpi_data = self.kpi_values[kpi_name]
            target = kpi_data.get('target', 0)
            actual = kpi_data.get('actual', 0)

            if target == 0:
                return 0

            score = (actual / target * 100) if target else 0
            return min(100, max(0, score))  # Clamp between 0-100
        return 0

    def get_overall_kpi_score(self):
        """Calculate overall KPI score based on role-specific weights"""
        role_kpis = KPI.objects.filter(role=self.employee.user.role, is_active=True)
        if not role_kpis.exists():
            # Fallback to legacy calculation
            return self.get_legacy_overall_score()

        total_weighted_score = 0
        total_weight = 0

        for kpi in role_kpis:
            score = self.get_kpi_score(kpi.name)
            total_weighted_score += score * (kpi.weight / 100)
            total_weight += kpi.weight / 100

        return total_weighted_score / total_weight * 100 if total_weight > 0 else 0

    def get_legacy_overall_score(self):
        """Fallback calculation using legacy fields"""
        # Calculate overall performance score using weighted average
        weights = {
            'sales': 0.25,
            'distribution': 0.20,
            'revenue': 0.30,
            'customer_base': 0.15,
            'engagement': 0.10
        }

        sales_percent = (self.sales_volume / self.sales_target * 100) if self.sales_target else 0
        distribution_percent = (self.distribution_volume / self.distribution_target * 100) if self.distribution_target else 0
        revenue_percent = (self.revenue_volume / self.revenue_target * 100) if self.revenue_target else 0
        customer_base_percent = (self.customer_base_volume / self.customer_base_target * 100) if self.customer_base_target else 0
        engagement_score = self.team_engagement_score or 0

        weighted_score = (
            (sales_percent / 100) * weights['sales'] +
            (distribution_percent / 100) * weights['distribution'] +
            (revenue_percent / 100) * weights['revenue'] +
            (customer_base_percent / 100) * weights['customer_base'] +
            (engagement_score / 10) * weights['engagement']
        ) * 100

        return weighted_score

    # Legacy properties for backward compatibility
    @property
    def sales_achieved_percent(self):
        return (self.sales_volume / self.sales_target * 100) if self.sales_target else 0

    @property
    def distribution_achieved_percent(self):
        return (self.distribution_volume / self.distribution_target * 100) if self.distribution_target else 0

    @property
    def revenue_achieved_percent(self):
        return (self.revenue_volume / self.revenue_target * 100) if self.revenue_target else 0

    @property
    def customer_base_achieved_percent(self):
        return (self.customer_base_volume / self.customer_base_target * 100) if self.customer_base_target else 0



class Evaluation(models.Model):
    EVALUATION_TYPES = [
        ('SELF', 'Self Evaluation'),
        ('MANAGER', 'Manager Evaluation'),
        ('PEER', 'Peer Review'),
        ('360', '360-Degree Feedback'),
        ('AI', 'AI Generated'),
    ]

    employee = models.ForeignKey(EmployeeProfile, on_delete=models.CASCADE)
    evaluator = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)  # Can be null for self-evaluations
    date = models.DateField(auto_now_add=True)
    remarks = models.TextField()
    performance_score = models.IntegerField()  # 1–10 scale or similar
    evaluation_type = models.CharField(max_length=10, choices=EVALUATION_TYPES, default='MANAGER')

    # Additional fields for different evaluation types
    strengths = models.TextField(blank=True)
    areas_for_improvement = models.TextField(blank=True)
    goals_achieved = models.TextField(blank=True)
    development_needs = models.TextField(blank=True)

    # Weighted criteria scores (JSON field for flexibility)
    criteria_scores = models.JSONField(default=dict)  # e.g., {'sales': 8, 'engagement': 7}

    # Link to evaluation criteria for this role
    evaluation_criteria = models.ManyToManyField('EvaluationCriteria', blank=True)

    # For 360-degree feedback
    is_anonymous = models.BooleanField(default=False)

    data_created = models.DateTimeField(auto_now_add=True, blank=True, null=True)
    data_updated = models.DateTimeField(auto_now=True, blank=True, null=True)
    created_by = models.ForeignKey(User, related_name='created_evaluations', on_delete=models.CASCADE, null=True, blank=True)
    updated_by = models.ForeignKey(User, related_name='updated_evaluations', on_delete=models.CASCADE, null=True, blank=True)

    def __str__(self):
        return f"Evaluation for {self.employee.user} by {self.evaluator}"

    def calculate_weighted_score(self):
        """
        Calculate overall performance score based on weighted criteria
        """
        if not self.criteria_scores or not self.evaluation_criteria.exists():
            return self.performance_score

        total_weight = 0
        weighted_sum = 0

        for criteria in self.evaluation_criteria.all():
            if criteria.criteria_name in self.criteria_scores:
                score = self.criteria_scores[criteria.criteria_name]
                weight = criteria.weight / 100.0  # Convert percentage to decimal
                weighted_sum += score * weight
                total_weight += weight

        if total_weight > 0:
            return round(weighted_sum / total_weight * 10, 1)  # Scale to 0-10
        return self.performance_score

    def get_criteria_summary(self):
        """
        Get a summary of criteria scores
        """
        if not self.criteria_scores:
            return {}

        summary = {}
        for criteria_name, score in self.criteria_scores.items():
            criteria_obj = self.evaluation_criteria.filter(criteria_name=criteria_name).first()
            if criteria_obj:
                summary[criteria_name] = {
                    'score': score,
                    'weight': criteria_obj.weight,
                    'description': criteria_obj.description
                }
        return summary


def generate_ai_evaluation(performance_record):
    """
    Generate an AI evaluation based on performance record data.
    Returns a dictionary with evaluation details.
    """
    from ai_engine.sentiment_analyzer import SentimentAnalyzer

    # Calculate performance percentages
    sales_percent = performance_record.sales_achieved_percent
    distribution_percent = performance_record.distribution_achieved_percent
    revenue_percent = performance_record.revenue_achieved_percent
    customer_base_percent = performance_record.customer_base_achieved_percent
    engagement_score = performance_record.team_engagement_score

    # Calculate overall performance score (weighted average)
    # Weights can be adjusted based on business priorities
    weights = {
        'sales': 0.25,
        'distribution': 0.20,
        'revenue': 0.30,
        'customer_base': 0.15,
        'engagement': 0.10
    }

    # Calculate weighted score (scale 0-10)
    weighted_score = (
        (sales_percent / 100) * weights['sales'] +
        (distribution_percent / 100) * weights['distribution'] +
        (revenue_percent / 100) * weights['revenue'] +
        (customer_base_percent / 100) * weights['customer_base'] +
        (engagement_score / 10) * weights['engagement']
    ) * 10

    # Generate remarks based on performance
    remarks = generate_ai_remarks(
        sales_percent,
        distribution_percent,
        revenue_percent,
        customer_base_percent,
        engagement_score
    )

    # Analyze sentiment of the generated remarks
    sentiment_analyzer = SentimentAnalyzer()
    sentiment_analysis = sentiment_analyzer.analyze_text_sentiment(remarks, use_huggingface=False)  # Use only TextBlob for speed

    # Generate strengths and areas for improvement based on performance
    strengths = []
    areas_for_improvement = []

    if sales_percent >= 90:
        strengths.append("Excellent sales performance")
    elif sales_percent < 70:
        areas_for_improvement.append("Sales target achievement needs improvement")

    if distribution_percent >= 90:
        strengths.append("Outstanding distribution management")
    elif distribution_percent < 70:
        areas_for_improvement.append("Distribution efficiency requires attention")

    if revenue_percent >= 90:
        strengths.append("Superior revenue generation")
    elif revenue_percent < 70:
        areas_for_improvement.append("Revenue growth strategies need enhancement")

    if customer_base_percent >= 90:
        strengths.append("Exceptional customer base expansion")
    elif customer_base_percent < 70:
        areas_for_improvement.append("Customer acquisition and retention needs focus")

    if engagement_score >= 8:
        strengths.append("High team engagement and collaboration")
    elif engagement_score < 6:
        areas_for_improvement.append("Team engagement and motivation needs improvement")

    return {
        'performance_score': round(weighted_score),
        'remarks': remarks,
        'strengths': '; '.join(strengths),
        'areas_for_improvement': '; '.join(areas_for_improvement),
        'sentiment_analysis': sentiment_analysis,
        'goals_achieved': f"Achieved {weighted_score:.1f}/10 overall performance score",
        'development_needs': '; '.join(areas_for_improvement) if areas_for_improvement else "Continue current performance level"
    }


def generate_ai_remarks(sales_percent, distribution_percent, revenue_percent, customer_base_percent, engagement_score):
    """
    Generate AI remarks based on performance metrics.
    """
    remarks = "AI-Generated Evaluation:\n\n"
    
    # Sales performance comment
    if sales_percent >= 100:
        remarks += f"✅ Excellent sales performance ({sales_percent:.1f}% of target achieved).\n"
    elif sales_percent >= 80:
        remarks += f"👍 Good sales performance ({sales_percent:.1f}% of target achieved).\n"
    elif sales_percent >= 60:
        remarks += f"⚠️ Sales performance needs improvement ({sales_percent:.1f}% of target achieved).\n"
    else:
        remarks += f"❌ Poor sales performance ({sales_percent:.1f}% of target achieved).\n"
    
    # Distribution performance comment
    if distribution_percent >= 100:
        remarks += f"✅ Excellent distribution performance ({distribution_percent:.1f}% of target achieved).\n"
    elif distribution_percent >= 80:
        remarks += f"👍 Good distribution performance ({distribution_percent:.1f}% of target achieved).\n"
    elif distribution_percent >= 60:
        remarks += f"⚠️ Distribution performance needs improvement ({distribution_percent:.1f}% of target achieved).\n"
    else:
        remarks += f"❌ Poor distribution performance ({distribution_percent:.1f}% of target achieved).\n"
    
    # Revenue performance comment
    if revenue_percent >= 100:
        remarks += f"✅ Excellent revenue performance ({revenue_percent:.1f}% of target achieved).\n"
    elif revenue_percent >= 80:
        remarks += f"👍 Good revenue performance ({revenue_percent:.1f}% of target achieved).\n"
    elif revenue_percent >= 60:
        remarks += f"⚠️ Revenue performance needs improvement ({revenue_percent:.1f}% of target achieved).\n"
    else:
        remarks += f"❌ Poor revenue performance ({revenue_percent:.1f}% of target achieved).\n"
    
    # Customer base performance comment
    if customer_base_percent >= 100:
        remarks += f"✅ Excellent customer base growth ({customer_base_percent:.1f}% of target achieved).\n"
    elif customer_base_percent >= 80:
        remarks += f"👍 Good customer base growth ({customer_base_percent:.1f}% of target achieved).\n"
    elif customer_base_percent >= 60:
        remarks += f"⚠️ Customer base growth needs improvement ({customer_base_percent:.1f}% of target achieved).\n"
    else:
        remarks += f"❌ Poor customer base growth ({customer_base_percent:.1f}% of target achieved).\n"
    
    # Engagement score comment
    if engagement_score >= 8:
        remarks += f"✅ Excellent team engagement score ({engagement_score}/10).\n"
    elif engagement_score >= 6:
        remarks += f"👍 Good team engagement score ({engagement_score}/10).\n"
    elif engagement_score >= 4:
        remarks += f"⚠️ Team engagement needs improvement ({engagement_score}/10).\n"
    else:
        remarks += f"❌ Poor team engagement score ({engagement_score}/10).\n"
    
    # Overall assessment
    avg_performance = (sales_percent + distribution_percent + revenue_percent + customer_base_percent) / 4
    if avg_performance >= 90 and engagement_score >= 8:
        remarks += "\n🏆 Overall: Outstanding performance! Keep up the excellent work.\n"
    elif avg_performance >= 75 and engagement_score >= 6:
        remarks += "\n🌟 Overall: Strong performance with good potential for growth.\n"
    elif avg_performance >= 60 and engagement_score >= 5:
        remarks += "\n📈 Overall: Satisfactory performance with areas for improvement.\n"
    else:
        remarks += "\n🔧 Overall: Performance requires significant improvement and focused attention.\n"
    
    return remarks


class Attendance(models.Model):
    """
    Tracks employee attendance and punctuality
    """
    employee = models.ForeignKey(EmployeeProfile, on_delete=models.CASCADE)
    date = models.DateField()
    check_in_time = models.TimeField(null=True, blank=True)
    check_out_time = models.TimeField(null=True, blank=True)
    is_present = models.BooleanField(default=True)
    is_late = models.BooleanField(default=False)
    is_early_departure = models.BooleanField(default=False)
    hours_worked = models.DecimalField(max_digits=4, decimal_places=2, null=True, blank=True)
    notes = models.TextField(blank=True)
    data_created = models.DateTimeField(auto_now_add=True)
    data_updated = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(User, related_name='created_attendance', on_delete=models.CASCADE, null=True, blank=True)
    updated_by = models.ForeignKey(User, related_name='updated_attendance', on_delete=models.CASCADE, null=True, blank=True)

    class Meta:
        unique_together = ('employee', 'date')

    def __str__(self):
        return f"{self.employee} - {self.date} - {'Present' if self.is_present else 'Absent'}"


class Task(models.Model):
    """
    Tracks employee tasks and project progress
    """
    STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('IN_PROGRESS', 'In Progress'),
        ('COMPLETED', 'Completed'),
        ('OVERDUE', 'Overdue'),
    ]

    PRIORITY_CHOICES = [
        ('LOW', 'Low'),
        ('MEDIUM', 'Medium'),
        ('HIGH', 'High'),
        ('CRITICAL', 'Critical'),
    ]

    employee = models.ForeignKey(EmployeeProfile, on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    priority = models.CharField(max_length=20, choices=PRIORITY_CHOICES, default='MEDIUM')
    assigned_date = models.DateField(auto_now_add=True)
    due_date = models.DateField(null=True, blank=True)
    completed_date = models.DateField(null=True, blank=True)
    progress_percentage = models.IntegerField(default=0, validators=[MinValueValidator(0), MaxValueValidator(100)])
    estimated_hours = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    actual_hours = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    project_name = models.CharField(max_length=100, blank=True)
    data_created = models.DateTimeField(auto_now_add=True)
    data_updated = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(User, related_name='created_tasks', on_delete=models.CASCADE, null=True, blank=True)
    updated_by = models.ForeignKey(User, related_name='updated_tasks', on_delete=models.CASCADE, null=True, blank=True)

    def __str__(self):
        return f"{self.employee} - {self.title} ({self.status})"


class PeerReview(models.Model):
    """
    Stores peer review feedback
    """
    reviewer = models.ForeignKey(EmployeeProfile, related_name='reviews_given', on_delete=models.CASCADE)
    reviewee = models.ForeignKey(EmployeeProfile, related_name='reviews_received', on_delete=models.CASCADE)
    review_date = models.DateField(auto_now_add=True)
    strengths = models.TextField(blank=True)
    areas_for_improvement = models.TextField(blank=True)
    overall_rating = models.IntegerField(validators=[MinValueValidator(1), MaxValueValidator(10)])
    collaboration_score = models.IntegerField(validators=[MinValueValidator(1), MaxValueValidator(10)])
    communication_score = models.IntegerField(validators=[MinValueValidator(1), MaxValueValidator(10)])
    technical_skills_score = models.IntegerField(validators=[MinValueValidator(1), MaxValueValidator(10)])
    feedback = models.TextField(blank=True)
    is_anonymous = models.BooleanField(default=False)
    data_created = models.DateTimeField(auto_now_add=True)
    data_updated = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(User, related_name='created_peer_reviews', on_delete=models.CASCADE, null=True, blank=True)
    updated_by = models.ForeignKey(User, related_name='updated_peer_reviews', on_delete=models.CASCADE, null=True, blank=True)

    def __str__(self):
        return f"Peer Review: {self.reviewer} -> {self.reviewee} ({self.overall_rating}/10)"


class Training(models.Model):
    """
    Tracks employee training and skill development
    """
    STATUS_CHOICES = [
        ('PLANNED', 'Planned'),
        ('IN_PROGRESS', 'In Progress'),
        ('COMPLETED', 'Completed'),
        ('CANCELLED', 'Cancelled'),
    ]

    employee = models.ForeignKey(EmployeeProfile, on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    training_type = models.CharField(max_length=100)  # e.g., 'Technical', 'Soft Skills', 'Certification'
    provider = models.CharField(max_length=100, blank=True)
    start_date = models.DateField(null=True, blank=True)
    end_date = models.DateField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PLANNED')
    completion_percentage = models.IntegerField(default=0, validators=[MinValueValidator(0), MaxValueValidator(100)])
    skills_gained = models.JSONField(default=list)  # List of skills acquired
    certification_earned = models.CharField(max_length=100, blank=True)
    cost = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    notes = models.TextField(blank=True)
    data_created = models.DateTimeField(auto_now_add=True)
    data_updated = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(User, related_name='created_trainings', on_delete=models.CASCADE, null=True, blank=True)
    updated_by = models.ForeignKey(User, related_name='updated_trainings', on_delete=models.CASCADE, null=True, blank=True)

    def __str__(self):
        return f"{self.employee} - {self.title} ({self.status})"


class EvaluationCriteria(models.Model):
    """
    Customizable evaluation criteria per role
    """
    role = models.CharField(max_length=20, choices=User.ROLE_CHOICES)
    criteria_name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    weight = models.DecimalField(max_digits=5, decimal_places=2, validators=[MinValueValidator(0), MaxValueValidator(100)])
    is_active = models.BooleanField(default=True)
    data_created = models.DateTimeField(auto_now_add=True)
    data_updated = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(User, related_name='created_criteria', on_delete=models.CASCADE, null=True, blank=True)
    updated_by = models.ForeignKey(User, related_name='updated_criteria', on_delete=models.CASCADE, null=True, blank=True)

    class Meta:
        unique_together = ('role', 'criteria_name')

    def __str__(self):
        return f"{self.role} - {self.criteria_name} ({self.weight}%)"
