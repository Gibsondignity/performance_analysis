from django import template

register = template.Library()

@register.filter
def divide(value, arg):
    try:
        return float(value) / float(arg) if arg != 0 else 0
    except:
        return 0



@register.filter
def add_class(field, css_class):
    return field.as_widget(attrs={"class": css_class})