from django.urls import path
from .views import stock_plot

urlpatterns = [
    path("stock/", stock_plot, name="stock_plot"),
]
