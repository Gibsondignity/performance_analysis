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
    path('evaluations/<int:pk>/edit/', views.edit_evaluation, name='edit_evaluation'),
    path('evaluations/<int:pk>/delete/', views.delete_evaluation, name='delete_evaluation'),


    # HIGH LEVEL MANAGER DASHBOARD
    path('deep_analytics/', views.deep_analytics, name='deep_analytics'),
]