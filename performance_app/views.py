from django.contrib.auth import authenticate, login, logout
from django.shortcuts import render, redirect
from datetime import datetime
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
import json
from django.shortcuts import render
from .models import *
from django.core.serializers.json import DjangoJSONEncoder
from django.db.models.functions import TruncMonth
from collections import defaultdict
from django.utils.timezone import now
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from .forms import UserForm, EmployeeProfileForm, EditUserForm, PerformanceRecordForm, EvaluationForm
from django.contrib import messages
from django.db import IntegrityError
from django.db.models import Avg, ExpressionWrapper, F, FloatField
from datetime import datetime
from django.utils.timezone import localtime
import csv



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

            if user.roles == 'EMPLOYEE':
                return redirect('employee')
            else:
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
    selected_month = request.GET.get('month')  # e.g., "June 2025"
    records = PerformanceRecord.objects.select_related('employee__user').all()

    # Group records by month-year
    grouped_data = defaultdict(list)
    month_options = set()

    for record in records:
        date = localtime(record.data_created)
        month_label = date.strftime("%B %Y")
        grouped_data[month_label].append(record)
        month_options.add(month_label)

    month_options = sorted(list(month_options))  # for dropdown

    # Filter by selected month or show all months
    filtered_months = [selected_month] if selected_month and selected_month in grouped_data else grouped_data.keys()

    sales_data = {'labels': [], 'datasets': []}
    revenue_data = {'labels': [], 'datasets': []}
    engagement_data = {'labels': [], 'scores': []}

    for month in filtered_months:
        recs = grouped_data[month]
        sales_avg = sum(r.sales_achieved_percent for r in recs) / len(recs)
        revenue_avg = sum(r.revenue_achieved_percent for r in recs) / len(recs)
        engagement_avg = sum(r.team_engagement_score for r in recs) / len(recs)

        sales_data['labels'].append(month)
        sales_data['datasets'].append(round(sales_avg, 2))
        revenue_data['labels'].append(month)
        revenue_data['datasets'].append(round(revenue_avg, 2))
        engagement_data['labels'].append(month)
        engagement_data['scores'].append(round(engagement_avg, 2))

    return render(request, 'dashboard/dashboard.html', {
        'sales_data': json.dumps(sales_data, cls=DjangoJSONEncoder),
        'revenue_data': json.dumps(revenue_data, cls=DjangoJSONEncoder),
        'engagement_data': json.dumps(engagement_data, cls=DjangoJSONEncoder),
        'month_options': month_options,
        'selected_month': selected_month,
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







# LOW LEVEL VIEWS FOR PERFORMANCE RECORDS
@login_required
def performance_record_list(request):
    """
    List all performance records with optional filtering by employee and performance date range.
    """
    records = PerformanceRecord.objects.select_related('employee', 'employee__user').order_by('-performance_start_date')
    employees = EmployeeProfile.objects.select_related('user').all()

    # Get filter values from GET request
    employee_id = request.GET.get('employee')
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')

    # Apply filters
    if employee_id:
        records = records.filter(employee_id=employee_id)
    if start_date and end_date:
        records = records.filter(
            performance_start_date__gte=start_date,
            performance_end_date__lte=end_date
        )

    return render(request, 'dashboard/low_level_manager/performance_records.html', {
        'records': records,
        'employees': employees,
        'selected_employee': int(employee_id) if employee_id else None,
        'start_date': start_date,
        'end_date': end_date,
    })



@login_required
def add_performance_record(request):
    """
    Add a new performance record.
    """
    if request.method == 'POST':
        form = PerformanceRecordForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, '✅ Performance record added successfully.')
            return redirect('performance_record_list')
        else:
            messages.error(request, '❌ Please correct the errors below.')
    else:
        form = PerformanceRecordForm()

    return render(request, 'dashboard/low_level_manager/add_performance.html', {'form': form})


@login_required
def edit_performance_record(request, pk):
    """
    Edit an existing performance record.
    """
    record = get_object_or_404(PerformanceRecord, pk=pk)

    if request.method == 'POST':
        form = PerformanceRecordForm(request.POST, instance=record)
        if form.is_valid():
            form.save()
            messages.success(request, '✅ Performance record updated successfully.')
            return redirect('performance_record_list')
        else:
            messages.error(request, '❌ Please correct the errors below.')
    else:
        form = PerformanceRecordForm(instance=record)

    return render(request, 'dashboard/low_level_manager/edit_performance_records.html', {'form': form, 'record': record})


@login_required
def delete_performance_record(request, pk):
    """
    Delete a performance record.
    """
    record = get_object_or_404(PerformanceRecord, pk=pk)

    if request.method == 'POST':
        record.delete()
        messages.success(request, '🗑️ Performance record deleted successfully.')
        return redirect('performance_record_list')
    
    messages.error(request, '❌ Invalid request method.')

    return redirect('performance_record_list')







def analytics_view(request):
    employees = EmployeeProfile.objects.select_related('user')
    selected_employee = request.GET.get('employee')
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')

    records = PerformanceRecord.objects.select_related('employee').all()

    if selected_employee:
        records = records.filter(employee_id=selected_employee)
    if start_date:
        records = records.filter(data_created__gte=start_date)
    if end_date:
        records = records.filter(data_created__lte=end_date)

    # Manually compute achieved percentages since they're not DB fields
    def safe_percent(numerator, denominator):
        return (numerator / denominator * 100) if denominator else 0

    sales_percentages = [safe_percent(r.sales_volume, r.sales_target) for r in records]
    dist_percentages = [safe_percent(r.distribution_volume, r.distribution_target) for r in records]
    revenue_percentages = [safe_percent(r.revenue_volume, r.revenue_target) for r in records]
    engagement_scores = [r.team_engagement_score for r in records]

    avg_sales = round(sum(sales_percentages) / len(sales_percentages), 2) if sales_percentages else 0
    avg_distribution = round(sum(dist_percentages) / len(dist_percentages), 2) if dist_percentages else 0
    avg_revenue = round(sum(revenue_percentages) / len(revenue_percentages), 2) if revenue_percentages else 0
    avg_engagement = round(sum(engagement_scores) / len(engagement_scores), 1) if engagement_scores else 0

    # For chart display
    chart_labels = [r.data_created.strftime('%Y-%m-%d') for r in records]
    chart_sales = sales_percentages
    chart_revenue = revenue_percentages
    chart_engagement = engagement_scores

    context = {
        'employees': employees,
        'selected_employee': int(selected_employee) if selected_employee else '',
        'start_date': start_date,
        'end_date': end_date,
        'avg_sales': avg_sales,
        'avg_distribution': avg_distribution,
        'avg_revenue': avg_revenue,
        'avg_engagement': avg_engagement,
        'chart_labels': chart_labels,
        'chart_sales': chart_sales,
        'chart_revenue': chart_revenue,
        'chart_engagement': chart_engagement,
    }

    return render(request, 'dashboard/low_level_manager/analytics.html', context)





def manager_report_dashboard(request):
    records = PerformanceRecord.objects.select_related('employee__user')

    # Filters
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    department = request.GET.get('department')

    if start_date:
        records = records.filter(performance_start_date__gte=start_date)
    if end_date:
        records = records.filter(performance_end_date__lte=end_date)
    if department:
        records = records.filter(employee__department=department)

    departments = EmployeeProfile.objects.values_list('department', flat=True).distinct()

    # Summary Metrics
    avg_sales = round(sum(r.sales_volume / r.sales_target * 100 for r in records if r.sales_target) / len(records), 2) if records else 0
    avg_revenue = round(sum(r.revenue_volume / r.revenue_target * 100 for r in records if r.revenue_target) / len(records), 2) if records else 0
    avg_engagement = round(sum(r.team_engagement_score for r in records) / len(records), 2) if records else 0

    # Sales Bar Chart Data by Department
    sales_chart = defaultdict(list)
    revenue_chart = defaultdict(float)
    engagement_chart = defaultdict(list)

    for r in records:
        dept = r.employee.department
        if r.sales_target:
            sales_chart[dept].append(r.sales_volume / r.sales_target * 100)
        if r.revenue_target:
            revenue_chart[dept] += r.revenue_volume / r.revenue_target * 100
        engagement_chart[str(r.performance_start_date)].append(r.team_engagement_score)

    sales_chart_data = {
        'labels': list(sales_chart.keys()),
        'data': [round(sum(v)/len(v), 2) for v in sales_chart.values()]
    }

    revenue_chart_data = {
        'labels': list(revenue_chart.keys()),
        'data': [round(v, 2) for v in revenue_chart.values()]
    }

    engagement_chart_data = {
        'labels': list(engagement_chart.keys()),
        'data': [round(sum(v)/len(v), 2) for v in engagement_chart.values()]
    }

    context = {
        'avg_sales': avg_sales,
        'avg_revenue': avg_revenue,
        'avg_engagement': avg_engagement,
        'departments': departments,
        'sales_chart_data': json.dumps(sales_chart_data),
        'revenue_chart_data': json.dumps(revenue_chart_data),
        'engagement_chart_data': json.dumps(engagement_chart_data)
    }

    return render(request, 'dashboard/low_level_manager/reports.html', context)


def export_manager_report(request):
    if request.method == 'POST':
        records = PerformanceRecord.objects.select_related('employee__user')

        # Apply filters
        start_date = request.POST.get('start_date')
        end_date = request.POST.get('end_date')
        department = request.POST.get('department')

        if start_date:
            records = records.filter(performance_start_date__gte=start_date)
        if end_date:
            records = records.filter(performance_end_date__lte=end_date)
        if department:
            records = records.filter(employee__department=department)

        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="performance_report.csv"'

        writer = csv.writer(response)
        writer.writerow(['Employee ID', 'Department', 'Sales (%)', 'Revenue (%)', 'Engagement', 'Start Date', 'End Date'])

        for r in records:
            sales = round(r.sales_volume / r.sales_target * 100, 2) if r.sales_target else 0
            revenue = round(r.revenue_volume / r.revenue_target * 100, 2) if r.revenue_target else 0
            writer.writerow([
                r.employee.user.employee_id,
                r.employee.department,
                sales,
                revenue,
                r.team_engagement_score,
                r.performance_start_date,
                r.performance_end_date
            ])

        return response





@login_required
def evaluation_list(request):
    evaluations = Evaluation.objects.select_related('employee', 'evaluator').order_by('-date')
    employees = EmployeeProfile.objects.all()
    evaluators = User.objects.filter(is_staff=True)  # assuming managers are staff users

    employee_id = request.GET.get('employee')
    evaluator_id = request.GET.get('evaluator')
    date = request.GET.get('date')

    if employee_id:
        evaluations = evaluations.filter(employee_id=employee_id)
    if evaluator_id:
        evaluations = evaluations.filter(evaluator_id=evaluator_id)
    if date:
        evaluations = evaluations.filter(date=date)

    return render(request, 'dashboard/low_level_manager/evaluation.html', {
        'evaluations': evaluations,
        'employees': employees,
        'evaluators': evaluators,
        'selected_employee': int(employee_id) if employee_id else None,
        'selected_evaluator': int(evaluator_id) if evaluator_id else None,
        'selected_date': date,
    })


@login_required
def add_evaluation(request):
    
    if request.method == 'POST':
        try:
            form = EvaluationForm(request.POST)
            if form.is_valid():
                evaluation = form.save(commit=False)
                evaluation.created_by = request.user
                evaluation.evaluator = request.user 
                evaluation.save()
                messages.success(request, '✅ Evaluation added successfully.')
                return redirect('evaluation_list')
            else:
                messages.error(request, '❌ Please correct the errors below.')
        except EmployeeProfile.DoesNotExist:
            messages.error(request, '❌ Employee profile not found. Please contact HR.')
    else:
        form = EvaluationForm()

    return render(request, 'dashboard/low_level_manager/add_evaluation.html', {'form': form})


@login_required
def edit_evaluation(request, pk):
    evaluation = get_object_or_404(Evaluation, pk=pk)

    if request.method == 'POST':
        form = EvaluationForm(request.POST, instance=evaluation)
        if form.is_valid():
            evaluation = form.save(commit=False)
            evaluation.updated_by = request.user
            evaluation.save()
            messages.success(request, '✅ Evaluation updated successfully.')
            return redirect('evaluation_list')
        else:
            messages.error(request, '❌ Please correct the errors below.')
    else:
        form = EvaluationForm(instance=evaluation)

    return render(request, 'dashboard/low_level_manager/edit_evaluation.html', {'form': form, 'evaluation': evaluation})


@login_required
def delete_evaluation(request, pk):
    evaluation = get_object_or_404(Evaluation, pk=pk)

    if request.method == 'POST':
        evaluation.delete()
        messages.success(request, '🗑️ Evaluation deleted successfully.')
        return redirect('evaluation_list')

    messages.error(request, '❌ Invalid request method.')
    return redirect('evaluation_list')





# HIGH LEVEL MANAGER VIEWS
def deep_analytics(request):
    records = PerformanceRecord.objects.select_related('employee__user')

    # Filters
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    department = request.GET.get('department')

    if start_date:
        records = records.filter(performance_start_date__gte=start_date)
    if end_date:
        records = records.filter(performance_end_date__lte=end_date)
    if department:
        records = records.filter(employee__department=department)

    departments = EmployeeProfile.objects.values_list('department', flat=True).distinct()

    # Summary Metrics
    avg_sales = round(sum(r.sales_volume / r.sales_target * 100 for r in records if r.sales_target) / len(records), 2) if records else 0
    avg_revenue = round(sum(r.revenue_volume / r.revenue_target * 100 for r in records if r.revenue_target) / len(records), 2) if records else 0
    avg_engagement = round(sum(r.team_engagement_score for r in records) / len(records), 2) if records else 0

    # Charts
    sales_chart = defaultdict(list)
    revenue_chart = defaultdict(float)
    engagement_chart = defaultdict(list)

    for r in records:
        dept = r.employee.department
        if r.sales_target:
            sales_chart[dept].append(r.sales_volume / r.sales_target * 100)
        if r.revenue_target:
            revenue_chart[dept] += r.revenue_volume / r.revenue_target * 100
        engagement_chart[str(r.performance_start_date)].append(r.team_engagement_score)

    sales_chart_data = {
        'labels': list(sales_chart.keys()),
        'data': [round(sum(v)/len(v), 2) for v in sales_chart.values()]
    }

    revenue_chart_data = {
        'labels': list(revenue_chart.keys()),
        'data': [round(v, 2) for v in revenue_chart.values()]
    }

    engagement_chart_data = {
        'labels': list(engagement_chart.keys()),
        'data': [round(sum(v)/len(v), 2) for v in engagement_chart.values()]
    }

    # Evaluation Metrics
    evaluations = Evaluation.objects.select_related('employee__user')
    if department:
        evaluations = evaluations.filter(employee__department=department)

    evaluation_chart = defaultdict(list)
    for e in evaluations:
        full_name = str(e.employee)  # Ensure it's a string
        evaluation_chart[full_name].append(e.performance_score)

    evaluation_chart_data = {
        'labels': list(evaluation_chart.keys()),
        'data': [round(sum(v) / len(v), 2) for v in evaluation_chart.values()]
    }

    avg_evaluation = round(sum(e.performance_score for e in evaluations) / len(evaluations), 2) if evaluations else 0

    context = {
        'avg_sales': avg_sales,
        'avg_revenue': avg_revenue,
        'avg_engagement': avg_engagement,
        'avg_evaluation': avg_evaluation,
        'departments': departments,
        'sales_chart_data': json.dumps(sales_chart_data),
        'revenue_chart_data': json.dumps(revenue_chart_data),
        'engagement_chart_data': json.dumps(engagement_chart_data),
        'evaluation_chart_data': json.dumps(evaluation_chart_data),
    }

    return render(request, 'dashboard/high_level_manager/reports.html', context)





@login_required
def employee(request):

    employee = EmployeeProfile.objects.get(user=request.user)
    return render(request, 'dashboard/employee/employee_info.html', {'employee': employee})




@login_required
def my_analytics_view(request):
    user = request.user
    employee = getattr(user, 'employeeprofile', None)

    if not employee:
        return render(request, 'errors/403.html')

    records = PerformanceRecord.objects.filter(employee=employee).order_by('performance_start_date')
    evaluations = Evaluation.objects.filter(employee=employee).order_by('date')

    # Filters
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')

    if start_date:
        records = records.filter(performance_start_date__gte=start_date)
        evaluations = evaluations.filter(date__gte=start_date)
    if end_date:
        records = records.filter(performance_end_date__lte=end_date)
        evaluations = evaluations.filter(date__lte=end_date)

    # Summary Stats
    avg_sales = round(sum(r.sales_achieved_percent for r in records) / len(records), 2) if records else 0
    avg_revenue = round(sum(r.revenue_achieved_percent for r in records) / len(records), 2) if records else 0
    avg_engagement = round(sum(r.team_engagement_score for r in records) / len(records), 2) if records else 0

    # Chart Data
    performance_chart = {
        'labels': [str(r.performance_start_date) for r in records],
        'sales': [round(r.sales_achieved_percent, 2) for r in records],
        'revenue': [round(r.revenue_achieved_percent, 2) for r in records]
    }

    evaluation_chart = {
        'labels': [str(e.date) for e in evaluations],
        'scores': [e.performance_score for e in evaluations]
    }

    context = {
        'avg_sales': avg_sales,
        'avg_revenue': avg_revenue,
        'avg_engagement': avg_engagement,
        'performance_chart_data': json.dumps(performance_chart),
        'evaluation_chart_data': json.dumps(evaluation_chart),
    }

    return render(request, 'dashboard/employee/emplyee_analytics.html', context)
