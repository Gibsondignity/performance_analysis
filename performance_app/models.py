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
        return self.employee_id



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
    # manager = models.ForeignKey(User, related_name='subordinates', null=True, blank=True, on_delete=models.SET_NULL)

    def __str__(self):
        return f"{self.user.employee_id} - {self.first_name} {self.last_name}"


# class ContactInfo(models.Model):
#     employee = models.OneToOneField(EmployeeProfile, on_delete=models.CASCADE)
#     phone = models.CharField(max_length=15)
#     address = models.TextField()
#     emergency_contact_name = models.CharField(max_length=100)
#     emergency_contact_phone = models.CharField(max_length=15)
#     emergency_relationship = models.CharField(max_length=50)


class PerformanceRecord(models.Model):
    employee = models.ForeignKey(EmployeeProfile, on_delete=models.CASCADE)
    date_recorded = models.DateField(default=0.00)
    sales_target = models.FloatField(default=0.00)
    sales_volume = models.FloatField(default=0.00)
    distribution_target = models.FloatField(default=0.00)
    distribution_volume = models.FloatField(default=0.00)
    revenue_target = models.FloatField(default=0.00)
    revenue_volume = models.FloatField(default=0.00)
    customer_base_target = models.FloatField(default=0.00)
    customer_base_volume = models.FloatField(default=0.00)
    team_engagement_score = models.IntegerField(default=0)  # e.g., from 1 to 10

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
    employee = models.ForeignKey(EmployeeProfile, on_delete=models.CASCADE)
    evaluator = models.ForeignKey(User, on_delete=models.CASCADE)  # Typically a manager
    date = models.DateField(auto_now_add=True)
    remarks = models.TextField()
    performance_score = models.IntegerField()  # 1–10 scale or similar

    def __str__(self):
        return f"Evaluation for {self.employee.user} by {self.evaluator.get_full_name()}"
