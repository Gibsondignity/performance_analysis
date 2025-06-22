from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django import forms
from django.contrib.auth.forms import ReadOnlyPasswordHashField
from django.conf import settings
from django.forms.widgets import TextInput, PasswordInput, Select, DateInput, CheckboxInput, NumberInput, Textarea

from .models import *



class UserCreationForm(forms.ModelForm):
    password1 = forms.CharField(label='Password', widget=forms.PasswordInput)
    password2 = forms.CharField(label='Confirm Password', widget=forms.PasswordInput)

    class Meta:
        model = User  # ✅ FIXED LINE
        fields = ('employee_id', 'role', 'branch')

    def clean_password2(self):
        password1 = self.cleaned_data.get('password1')
        password2 = self.cleaned_data.get('password2')
        if password1 and password2 and password1 != password2:
            raise forms.ValidationError("Passwords do not match")
        return password2

    def save(self, commit=True):
        user = super().save(commit=False)
        user.set_password(self.cleaned_data['password1'])  # Hash the password
        if commit:
            user.save()
        return user


class UserChangeForm(forms.ModelForm):
    password = ReadOnlyPasswordHashField()

    class Meta:
        model = User
        fields = ('employee_id', 'password', 'role', 'branch', 'is_active', 'is_staff', 'is_superuser')

    def clean_password(self):
        return self.initial["password"]




class UserForm(forms.ModelForm):
    password = forms.CharField(
        required=False,
        widget=PasswordInput(attrs={
            'class': 'w-full px-3 py-2 p-4 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500',
            'placeholder': 'Enter password'
        })
    )

    class Meta:
        model = User
        fields = ['employee_id', 'password', 'role', 'branch', 'is_active']
        widgets = {
            'employee_id': TextInput(attrs={
                'class': 'w-full px-3 py-2 p-4 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500',
                'placeholder': 'Employee ID'
            }),
            'role': Select(attrs={
                'class': 'w-full px-3 py-2 p-4 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500'
            }),
            'branch': TextInput(attrs={
                'class': 'w-full px-3 py-2 p-4 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500'
            }),
            'is_active': CheckboxInput(attrs={
                'class': 'form-checkbox h-5 w-5 text-blue-600'
            }),
        }

    def save(self, commit=True):
        user = super().save(commit=False)
        if self.cleaned_data['password']:
            user.set_password(self.cleaned_data['password'])
        if commit:
            user.save()
        return user


class EmployeeProfileForm(forms.ModelForm):
    class Meta:
        model = EmployeeProfile
        fields = ['first_name', 'last_name', 'department', 'job_title', 'date_hired', 'phone', 'address', 'emergency_contact_name', 'emergency_contact_phone', 'emergency_relationship']
        widgets = {
            'first_name': TextInput(attrs={
                'class': 'w-full px-3 py-2 p-4 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500',
                'placeholder': 'First Name'
            }),
            'last_name': TextInput(attrs={
                'class': 'w-full px-3 py-2 p-4 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500',
                'placeholder': 'Last Name'
            }),
            'department': TextInput(attrs={
                'class': 'w-full px-3 py-2 p-4 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500',
                'placeholder': 'Department'
            }),
            'job_title': TextInput(attrs={
                'class': 'w-full px-3 py-2 p-4 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500',
                'placeholder': 'Job Title'
            }),
            'date_hired': DateInput(attrs={
                'class': 'w-full px-3 py-2 p-4 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500',
                'type': 'date'
            }),
           'phone': TextInput(attrs={
                'class': 'w-full px-3 py-2 p-4 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500',
                'placeholder': 'Phone number'
            }),
            'address': TextInput(attrs={
                'class': 'w-full px-3 py-2 p-4 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500',
                'placeholder': 'Home address'
            }),
            'emergency_contact_name': TextInput(attrs={
                'class': 'w-full px-3 py-2 p-4 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500',
                'placeholder': 'Emergency Contact Name'
            }),
            'emergency_contact_phone': TextInput(attrs={
                'class': 'w-full px-3 py-2 p-4 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500',
                'placeholder': 'Emergency Phone'
            }),
            'emergency_relationship': TextInput(attrs={
                'class': 'w-full px-3 py-2 p-4 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500',
                'placeholder': 'Relationship'
            }),
        }




class EditUserForm(forms.ModelForm):
    password = forms.CharField(
        required=False,
        widget=PasswordInput(attrs={
            'class': 'w-full px-3 py-2 p-4 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500',
            'placeholder': 'Enter password'
        })
    )

    class Meta:
        model = User
        fields = ['employee_id', 'role', 'branch', 'is_active']
        widgets = {
            'employee_id': TextInput(attrs={
                'class': 'w-full px-3 py-2 p-4 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500',
                'placeholder': 'Employee ID'
            }),
            'role': Select(attrs={
                'class': 'w-full px-3 py-2 p-4 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500'
            }),
            'branch': TextInput(attrs={
                'class': 'w-full px-3 py-2 p-4 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500'
            }),
            'is_active': CheckboxInput(attrs={
                'class': 'form-checkbox h-5 w-5 text-blue-600'
            }),
        }

    def save(self, commit=True):
        user = super().save(commit=False)
        if self.cleaned_data['password']:
            user.set_password(self.cleaned_data['password'])
        if commit:
            user.save()
        return user


class EditEmployeeProfileForm(forms.ModelForm):
    class Meta:
        model = EmployeeProfile
        fields = ['first_name', 'last_name', 'department', 'job_title', 'date_hired', 'phone', 'address', 'emergency_contact_name', 'emergency_contact_phone', 'emergency_relationship']
        widgets = {
            'first_name': TextInput(attrs={
                'class': 'w-full px-3 py-2 p-4 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500',
                'placeholder': 'First Name'
            }),
            'last_name': TextInput(attrs={
                'class': 'w-full px-3 py-2 p-4 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500',
                'placeholder': 'Last Name'
            }),
            'department': TextInput(attrs={
                'class': 'w-full px-3 py-2 p-4 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500',
                'placeholder': 'Department'
            }),
            'job_title': TextInput(attrs={
                'class': 'w-full px-3 py-2 p-4 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500',
                'placeholder': 'Job Title'
            }),
            'date_hired': DateInput(attrs={
                'class': 'w-full px-3 py-2 p-4 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500',
                'type': 'date'
            }),
           'phone': TextInput(attrs={
                'class': 'w-full px-3 py-2 p-4 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500',
                'placeholder': 'Phone number'
            }),
            'address': TextInput(attrs={
                'class': 'w-full px-3 py-2 p-4 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500',
                'placeholder': 'Home address'
            }),
            'emergency_contact_name': TextInput(attrs={
                'class': 'w-full px-3 py-2 p-4 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500',
                'placeholder': 'Emergency Contact Name'
            }),
            'emergency_contact_phone': TextInput(attrs={
                'class': 'w-full px-3 py-2 p-4 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500',
                'placeholder': 'Emergency Phone'
            }),
            'emergency_relationship': TextInput(attrs={
                'class': 'w-full px-3 py-2 p-4 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500',
                'placeholder': 'Relationship'
            }),
        }





class PerformanceRecordForm(forms.ModelForm):
    class Meta:
        model = PerformanceRecord
        fields = ['employee', 
                'sales_target', 
                'sales_volume', 
                'distribution_target', 
                'distribution_volume', 
                'revenue_target', 
                'revenue_volume',
                'customer_base_target',
                'customer_base_volume',
                'team_engagement_score',
                'performance_start_date',
                'performance_end_date'
            ]
        
        widgets = {
            'employee': forms.Select(attrs={
                'class': 'w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500'
            }),
            'sales_target': NumberInput(attrs={
                'class': 'w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500',
                'step': '0.01',
                'placeholder': 'Sales Target'
            }),
            'sales_volume': NumberInput(attrs={
                'class': 'w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500',
                'step': '0.01',
                'placeholder': 'Sales Volume'
            }),
            'distribution_target': NumberInput(attrs={
                'class': 'w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500',
                'step': '0.01',
                'placeholder': 'Distribution Target'
            }),
            'distribution_volume': NumberInput(attrs={
                'class': 'w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500',
                'step': '0.01',
                'placeholder': 'Distribution Volume'
            }),
            'revenue_target': NumberInput(attrs={
                'class': 'w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500',
                'step': '0.01',
                'placeholder': 'Revenue Target'
            }),
            'revenue_volume': NumberInput(attrs={
                'class': 'w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500',
                'step': '0.01',
                'placeholder': 'Revenue Volume'
            }),
            'customer_base_target': NumberInput(attrs={
                'class': 'w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500',
                'step': '0.01',
                'placeholder': 'Customer Base Target'
            }),
            'customer_base_volume': NumberInput(attrs={
                'class': 'w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500',
                'step': '0.01',
                'placeholder': 'Customer Base Volume'
            }),
            'team_engagement_score': NumberInput(attrs={
                'class': 'w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500',
                'min': 0,
                'max': 10,
                'placeholder': 'Score (0-10)'
            }),
            'performance_start_date': DateInput(attrs={
                'class': 'w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500',
                'type': 'date'
            }),
            'performance_end_date': DateInput(attrs={
                'class': 'w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500',
                'type': 'date'
            }),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['employee'].queryset = EmployeeProfile.objects.select_related('user')



class EvaluationForm(forms.ModelForm):
    class Meta:
        model = Evaluation
        fields = ['employee', 'remarks', 'performance_score']
        widgets = {
            'employee': forms.Select(attrs={
                'class': 'w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500'
            }),
            'evaluator': forms.Select(attrs={
                'class': 'w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500'
            }),
            'date': DateInput(attrs={
                'type': 'date',
                'class': 'w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500'
            }),
            'performance_score': forms.NumberInput(attrs={
                'min': 1,
                'max': 10,
                'class': 'w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500'
            }),
            'remarks': Textarea(attrs={
                'rows': 4,
                'class': 'w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500',
                'placeholder': 'Write evaluation remarks here...'
            }),
        }