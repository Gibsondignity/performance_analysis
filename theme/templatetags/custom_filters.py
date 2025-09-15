from django import template

register = template.Library()

@register.filter
def mul(value, arg):
    try:
        return float(value) * float(arg)
    except:
        return 0

@register.filter
def div(value, arg):
    try:
        return float(value) / float(arg) if arg != 0 else 0
    except:
        return 0


@register.filter
def divide(value, arg):
    try:
        return float(value) / float(arg) if arg != 0 else 0
    except:
        return 0

@register.filter
def add_class(field, css_class):
    return field.as_widget(attrs={"class": css_class})


@register.filter
def replace(value, arg):
    """
    Replace occurrences of a substring with another substring.
    Usage: {{ value|replace:"old:new" }}
    """
    if not value:
        return value

    try:
        old, new = arg.split(':', 1)
        return str(value).replace(old, new)
    except (ValueError, AttributeError):
        return value