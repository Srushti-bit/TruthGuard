from django.contrib import admin
from .models import NewsSubmission

@admin.register(NewsSubmission)
class NewsSubmissionAdmin(admin.ModelAdmin):
    list_display = ['user', 'prediction', 'confidence_score', 'submitted_at']
    list_filter = ['prediction', 'submitted_at']
    search_fields = ['user__username', 'news_text']
    readonly_fields = ['submitted_at']