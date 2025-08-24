from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from performance_app.models import PerformanceRecord, Evaluation, generate_ai_evaluation

User = get_user_model()

class Command(BaseCommand):
    help = 'Generate AI evaluations for all existing performance records that do not have AI evaluations'

    def add_arguments(self, parser):
        parser.add_argument(
            '--user-id',
            type=str,
            help='Employee ID of the user to use as the evaluator for AI evaluations',
        )

    def handle(self, *args, **options):
        # Get the evaluator user
        evaluator = None
        if options['user_id']:
            try:
                evaluator = User.objects.get(employee_id=options['user_id'])
            except User.DoesNotExist:
                self.stdout.write(
                    self.style.ERROR(f'User with employee ID "{options["user_id"]}" does not exist')
                )
                return
        
        # If no specific user provided, try to get the first admin user
        if not evaluator:
            try:
                evaluator = User.objects.filter(is_staff=True).first()
                if not evaluator:
                    # If no admin user exists, get the first user
                    evaluator = User.objects.first()
            except User.DoesNotExist:
                self.stdout.write(
                    self.style.ERROR('No users found in the system')
                )
                return

        # Get all performance records that don't have AI evaluations
        performance_records = PerformanceRecord.objects.all()
        
        created_count = 0
        skipped_count = 0
        
        for record in performance_records:
            # Check if an AI evaluation already exists for this record
            existing_ai_eval = Evaluation.objects.filter(
                employee=record.employee,
                evaluation_type='AI'
            ).first()
            
            if not existing_ai_eval:
                # Generate AI evaluation
                ai_evaluation_data = generate_ai_evaluation(record)
                Evaluation.objects.create(
                    employee=record.employee,
                    performance_score=ai_evaluation_data['performance_score'],
                    remarks=ai_evaluation_data['remarks'],
                    evaluation_type='AI',
                    created_by=evaluator
                )
                created_count += 1
                self.stdout.write(
                    self.style.SUCCESS(f'Created AI evaluation for {record.employee}')
                )
            else:
                skipped_count += 1
                self.stdout.write(
                    self.style.WARNING(f'Skipped {record.employee} - AI evaluation already exists')
                )
        
        self.stdout.write(
            self.style.SUCCESS(
                f'Successfully created {created_count} AI evaluations, skipped {skipped_count} records'
            )
        )