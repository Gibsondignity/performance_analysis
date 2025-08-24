from django.contrib import admin
from .models import *
from .forms import *

# Register your models here.
admin.site.site_header = "EDPMS ADMIN PORTAL"

admin.site.site_title = "EDPMS ADMIN PORTAL"
admin.site.index_title = "WELCOME TO THE EDPMS ADMIN PORTAL"
# admin.site.register(User)
admin.site.register(EmployeeProfile)
# admin.site.register(ContactInfo)
class PerformanceRecordAdmin(admin.ModelAdmin):
    list_display = ('employee', 'sales_achieved_percent', 'revenue_achieved_percent',
                    'distribution_achieved_percent', 'customer_base_achieved_percent',
                    'team_engagement_score', 'performance_start_date', 'performance_end_date')
    list_filter = ('performance_start_date', 'performance_end_date', 'employee__department')
    search_fields = ('employee__user__employee_id', 'employee__first_name', 'employee__last_name')
    
    def sales_achieved_percent(self, obj):
        return f"{obj.sales_achieved_percent:.1f}%"
    sales_achieved_percent.short_description = 'Sales %'
    
    def revenue_achieved_percent(self, obj):
        return f"{obj.revenue_achieved_percent:.1f}%"
    revenue_achieved_percent.short_description = 'Revenue %'
    
    def distribution_achieved_percent(self, obj):
        return f"{obj.distribution_achieved_percent:.1f}%"
    distribution_achieved_percent.short_description = 'Distribution %'
    
    def customer_base_achieved_percent(self, obj):
        return f"{obj.customer_base_achieved_percent:.1f}%"
    customer_base_achieved_percent.short_description = 'Customer Base %'

admin.site.register(PerformanceRecord, PerformanceRecordAdmin)
class EvaluationAdmin(admin.ModelAdmin):
    list_display = ('employee', 'evaluator', 'performance_score', 'date', 'evaluation_type')
    list_filter = ('evaluation_type', 'date', 'performance_score')
    search_fields = ('employee__user__employee_id', 'employee__first_name', 'employee__last_name')
    readonly_fields = ('evaluation_type',)  # Make evaluation_type readonly in admin
    
    def get_readonly_fields(self, request, obj=None):
        # Make all fields readonly for AI evaluations
        if obj and obj.evaluation_type == 'AI':
            return [f.name for f in self.model._meta.fields]
        return self.readonly_fields

admin.site.register(Evaluation, EvaluationAdmin)


class CustomUserAdmin(BaseUserAdmin):
    form = UserChangeForm
    add_form = UserCreationForm

    list_display = ('employee_id', 'role', 'is_staff')
    list_filter = ('is_staff', 'role')
    fieldsets = (
        (None, {'fields': ('employee_id', 'password')}),
        ('Personal info', {'fields': ('role', 'branch')}),
        ('Permissions', {'fields': ('is_staff', 'is_superuser', 'is_active', 'groups', 'user_permissions')}),
    )
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('employee_id', 'role', 'branch', 'password1', 'password2'),
        }),
    )
    search_fields = ('employee_id',)
    ordering = ('employee_id',)
    filter_horizontal = ('groups', 'user_permissions',)

# Register your custom User model with the custom admin
admin.site.register(User, CustomUserAdmin)
