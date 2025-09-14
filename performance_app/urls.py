from django.urls import path
from .import views


urlpatterns = [
    path('', views.custom_login_view, name='login'),
    path('logout_view?logged-out-of-the-dashboard', views.logout_view, name='logout'),
    path('dashboard/', views.analytical_dashboard, name='dashboard'),

    # HR MANAGER DASHBOARD
    path('employees/', views.employee_management, name='employee_management'),
    path('performance_records/', views.performance_records, name='performance_records'),
    path('employees/view/<int:user_id>/', views.view_employee, name='view_employee'),
    path('employees/delete/<int:user_id>/', views.delete_employee, name='delete_employee'),
    path('employees/add/', views.add_employee, name='add_employee'),
    path('employees/edit/<int:user_id>/', views.edit_employee, name='edit_employee'),

    # PERFORMANCE RECORDS
    path('performance/', views.performance_record_list, name='performance_record_list'),
    path('performance/add/', views.add_performance_record, name='add_performance_record'),
    path('performance/<int:pk>/edit/', views.edit_performance_record, name='edit_performance_record'),
    path('performance/<int:pk>/delete/', views.delete_performance_record, name='delete_performance_record'),

    path('analytics/', views.analytics_view, name='analytics'),
    path('manager-analytics/', views.manager_report_dashboard, name='manager_report_dashboard'),
    path('export_manager_report/', views.export_manager_report, name='export_manager_report'),

    path('evaluations/', views.evaluation_list, name='evaluation_list'),
    path('evaluations/add/', views.add_evaluation, name='add_evaluation'),
    path('evaluations/self/add/', views.add_self_evaluation, name='add_self_evaluation'),
    path('evaluations/manager/<int:employee_id>/add/', views.add_manager_evaluation, name='add_manager_evaluation'),
    path('evaluations/360/<int:employee_id>/add/', views.add_360_evaluation, name='add_360_evaluation'),
    path('evaluations/<int:pk>/edit/', views.edit_evaluation, name='edit_evaluation'),
    path('evaluations/<int:pk>/delete/', views.delete_evaluation, name='delete_evaluation'),


    # HIGH LEVEL MANAGER DASHBOARD
    path('deep_analytics/', views.deep_analytics, name='deep_analytics'),


    # EMPLOYEE DASHBOARD
    path('employee', views.employee, name='employee'),
    path('my-analytics/', views.my_analytics_view, name='my_analytics'),
    path('my_evaluation_list/', views.my_evaluation_list, name='my_evaluation_list'),

    # AI-POWERED FEATURES
    path('ai/employee/<int:employee_id>/', views.ai_employee_analysis, name='ai_employee_analysis'),
    path('ai/department/', views.ai_department_analysis, name='ai_department_analysis'),
    path('ai/recommendations/', views.ai_recommendations_dashboard, name='ai_recommendations'),
    path('ai/insights/', views.ai_insights_dashboard, name='ai_insights'),
    path('ai/anomalies/', views.ai_anomalies_dashboard, name='ai_anomalies'),

    # ATTENDANCE MANAGEMENT
    path('attendance/', views.attendance_list, name='attendance_list'),
    path('attendance/add/', views.add_attendance, name='add_attendance'),

    # TASK MANAGEMENT
    path('tasks/', views.task_list, name='task_list'),
    path('tasks/add/', views.add_task, name='add_task'),
    path('tasks/<int:task_id>/update/', views.update_task_status, name='update_task_status'),

    # PEER REVIEWS
    path('peer-reviews/', views.peer_review_list, name='peer_review_list'),
    path('peer-reviews/add/', views.add_peer_review, name='add_peer_review'),

    # TRAINING MANAGEMENT
    path('training/', views.training_list, name='training_list'),
    path('training/add/', views.add_training, name='add_training'),
    path('training/<int:training_id>/update/', views.update_training_progress, name='update_training_progress'),

    # KPI MANAGEMENT
    path('kpis/', views.kpi_list, name='kpi_list'),
    path('kpis/add/', views.add_kpi, name='add_kpi'),
    path('kpis/<int:kpi_id>/edit/', views.edit_kpi, name='edit_kpi'),
    path('kpis/<int:kpi_id>/delete/', views.delete_kpi, name='delete_kpi'),

    # EVALUATION CRITERIA
    path('evaluation-criteria/', views.evaluation_criteria_list, name='evaluation_criteria_list'),
    path('evaluation-criteria/add/', views.add_evaluation_criteria, name='add_evaluation_criteria'),

    # ENHANCED ANALYTICS
    path('analytics/attendance/', views.attendance_analytics, name='attendance_analytics'),
    path('analytics/tasks/', views.task_analytics, name='task_analytics'),
    path('analytics/training/', views.training_analytics, name='training_analytics'),

    # COMPREHENSIVE REPORTING
    path('reports/generate/', views.generate_performance_report, name='generate_performance_report'),

    # AI DEMONSTRATION
    path('ai/demo/', views.ai_demo_dashboard, name='ai_demo_dashboard'),

    # HR DASHBOARD
    path('hr/dashboard/', views.hr_dashboard, name='hr_dashboard'),

    # ENHANCED REPORTS & EXPORTS
    path('reports/organization/', views.export_organization_report, name='export_organization_report'),
]