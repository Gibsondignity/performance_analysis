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




]