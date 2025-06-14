from django.contrib.auth import authenticate, login, logout
from django.shortcuts import render, redirect
from datetime import datetime
from django.contrib.auth.decorators import login_required
import json
from django.shortcuts import render
from .models import *
from django.core.serializers.json import DjangoJSONEncoder
from django.utils.timezone import now
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from .forms import UserForm, EmployeeProfileForm, EditUserForm
from django.contrib import messages
from django.db import IntegrityError



def custom_login_view(request):
    error = None
    if request.method == 'POST':
        employee_id = request.POST.get('employee_id')
        password = request.POST.get('password')
        user = authenticate(request, username=employee_id, password=password)

        if user:
            login(request, user)
            profile = EmployeeProfile.objects.get(user=user)
            # Fetch employee profile details
            try:
                
                print(f"Profile found: {profile}")
                request.session['employee_id'] = user.employee_id
                request.session['first_name'] = profile.first_name
                request.session['last_name'] = profile.last_name
            except EmployeeProfile.DoesNotExist:
                # You could still store the ID, in case profile is missing
                request.session['employee_id'] = user.employee_id
                request.session['first_name'] = ''
                request.session['last_name'] = ''

            return redirect('dashboard')

        error = "Invalid Employee ID or password"

    return render(request, 'login.html', {'error': error, 'year': now().year})



def logout_view(request):
    if request.method == 'POST':
        logout(request)
        request.session.flush()
        return redirect('login')
    return render(request, 'login.html', {'year': datetime.now().year})


@login_required
def analytical_dashboard(request):
    records = PerformanceRecord.objects.all()

    sales_data = {
        'labels': [r.employee.user.employee_id for r in records],
        'percentages': [round(r.sales_achieved_percent, 2) for r in records]
    }

    revenue_data = {
        'labels': [r.employee.user.employee_id for r in records],
        'percentages': [round(r.revenue_achieved_percent, 2) for r in records]
    }

    engagement_data = {
        'labels': [str(r.date_recorded) for r in records],
        'scores': [r.team_engagement_score for r in records]
    }

    return render(request, 'dashboard/dashboard.html', {
        'sales_data': json.dumps(sales_data, cls=DjangoJSONEncoder),
        'revenue_data': json.dumps(revenue_data, cls=DjangoJSONEncoder),
        'engagement_data': json.dumps(engagement_data, cls=DjangoJSONEncoder),
    })





# HR MANAGER DASHBOARD
@login_required
def employee_management(request):
    if request.method == 'POST':
        user_id = request.POST.get('user_id')
        if user_id:
            user = get_object_or_404(User, pk=user_id)
            user_form = UserForm(request.POST, instance=user)
            profile_form = EmployeeProfileForm(request.POST, instance=user.employeeprofile)
        else:
            user_form = UserForm(request.POST)
            profile_form = EmployeeProfileForm(request.POST)
        print(user_form.errors, profile_form.errors)
        # contact_form = ContactInfoForm(request.POST)
        if user_form.is_valid() and profile_form.is_valid():
            user = user_form.save()
            profile = profile_form.save(commit=False)
            profile.user = user
            profile.save()
            messages.success(request, 'Employee saved successfully.')
            return redirect('employee_management')
        else:
            messages.error(request, 'There was an error saving the employee.')

    user_form = UserForm()
    profile_form = EmployeeProfileForm()
    # contact_form = ContactInfoForm()

    employees = EmployeeProfile.objects.select_related('user')
    return render(request, 'dashboard/hr/employee_mgmt.html', {
        'employees': employees,
        'user_form': user_form,
        'profile_form': profile_form,
        # 'contact_form': contact_form,
    })



@login_required
def add_employee(request):
    if request.method == 'POST':
        user_form = UserForm(request.POST)
        profile_form = EmployeeProfileForm(request.POST)

        if user_form.is_valid() and profile_form.is_valid():
            try:
                user = user_form.save()
                profile = profile_form.save(commit=False)
                profile.user = user
                profile.save()
                messages.success(request, '✅ Employee added successfully.')
                return redirect('employee_management')
            except IntegrityError as e:
                if 'unique constraint' in str(e).lower() or 'UNIQUE constraint failed' in str(e):
                    messages.error(request, '❌ Employee ID already exists.')
                else:
                    messages.error(request, f'❌ An error occurred: {e}')
        else:
            # Forms not valid; let the template display field errors
            messages.error(request, '❌ Please correct the errors below.')
    else:
        user_form = UserForm()
        profile_form = EmployeeProfileForm()

    return render(request, 'dashboard/hr/add_employee.html', {
        'user_form': user_form,
        'profile_form': profile_form,
    })


@login_required
def view_employee(request, user_id):
    employee = get_object_or_404(EmployeeProfile, user__id=user_id)
    return render(request, 'dashboard/hr/view_employee.html', {'employee': employee})



@login_required
def edit_employee(request, user_id):
    user = get_object_or_404(User, id=user_id)
    profile = get_object_or_404(EmployeeProfile, user=user)

    if request.method == 'POST':
        user_form = EditUserForm(request.POST, instance=user)
        profile_form = EmployeeProfileForm(request.POST, instance=profile)

        if user_form.is_valid() and profile_form.is_valid():
            user_form.save()
            profile_form.save()
            messages.success(request, 'Employee updated successfully.')
            return redirect('employee_management')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        user_form = EditUserForm(instance=user)
        profile_form = EmployeeProfileForm(instance=profile)

    return render(request, 'dashboard/hr/edit_employee.html', {
        'user_form': user_form,
        'profile_form': profile_form,
        'employee': profile
    })



@login_required
def delete_employee(request, user_id):
    user = get_object_or_404(User, id=user_id)

    if request.method == 'POST':
        user.delete()
        messages.success(request, 'Employee deleted successfully.')
        return redirect('employee_management')

    messages.error(request, 'Invalid request method.')
    return redirect('employee_management')


@login_required
def performance_records(request):
    records = PerformanceRecord.objects.all()
    return render(request, 'dashboard/hr/performance_records.html', {'records': records})