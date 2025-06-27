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
admin.site.register(PerformanceRecord)
admin.site.register(Evaluation)


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
