from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin

class CustomUserManager(BaseUserManager):
    def create_user(self, employee_id, password=None, **extra_fields):
        if not employee_id:
            raise ValueError('The Employee ID must be set')
        user = self.model(employee_id=employee_id, **extra_fields)
        user.set_password(password)
        user.save()
        return user

    def create_superuser(self, employee_id, password=None, **extra_fields):
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

class PerformanceRecord(models.Model):
    employee = models.ForeignKey(EmployeeProfile, on_delete=models.CASCADE)
    sales_target = models.FloatField(default=0.00)
    sales_volume = models.FloatField(default=0.00)
    distribution_target = models.FloatField(default=0.00)
    distribution_volume = models.FloatField(default=0.00)
    revenue_target = models.FloatField(default=0.00)
    revenue_volume = models.FloatField(default=0.00)
    customer_base_target = models.FloatField(default=0.00)
    customer_base_volume = models.FloatField(default=0.00)
    team_engagement_score = models.IntegerField(default=0)  # e.g., from 1 to 10

    # NEW: Date range of performance period
    performance_start_date = models.DateField(blank=True, null=True)
    performance_end_date = models.DateField(blank=True, null=True)

    # Existing timestamps
    data_created = models.DateTimeField(auto_now_add=True, blank=True, null=True)
    data_updated = models.DateTimeField(auto_now=True, blank=True, null=True)
    created_by = models.ForeignKey(User, related_name='created_records', on_delete=models.CASCADE, null=True, blank=True)
    updated_by = models.ForeignKey(User, related_name='updated_records', on_delete=models.CASCADE, null=True, blank=True)

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
        ('MANUAL', 'Manual'),
        ('AI', 'AI Generated'),
    ]
    
    employee = models.ForeignKey(EmployeeProfile, on_delete=models.CASCADE)
    evaluator = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)  # Typically a manager
    date = models.DateField(auto_now_add=True)
    remarks = models.TextField()
    performance_score = models.IntegerField()  # 1–10 scale or similar
    evaluation_type = models.CharField(max_length=10, choices=EVALUATION_TYPES, default='MANUAL')
    data_created = models.DateTimeField(auto_now_add=True, blank=True, null=True)
    data_updated = models.DateTimeField(auto_now=True, blank=True, null=True)
    created_by = models.ForeignKey(User, related_name='created_profiles', on_delete=models.CASCADE, null=True, blank=True)
    updated_by = models.ForeignKey(User, related_name='updated_profiles', on_delete=models.CASCADE, null=True, blank=True)

    def __str__(self):
        return f"Evaluation for {self.employee.user} by {self.evaluator}"


def generate_ai_evaluation(performance_record):
    """
    Generate an AI evaluation based on performance record data.
    Returns a dictionary with evaluation details.
    """
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
    
    return {
        'performance_score': round(weighted_score),
        'remarks': remarks
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
