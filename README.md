# Build and start
docker-compose up --build

# In another terminal, run migrations
docker-compose exec web python manage.py makemigrations
docker-compose exec web python manage.py migrate

# Create superuser
docker-compose exec web python manage.py createsuperuser


