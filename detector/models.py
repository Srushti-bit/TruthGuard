from django.db import models
from django.contrib.auth.models import User

class NewsSubmission(models.Model):
    PREDICTION_CHOICES = [
        ('REAL', 'Real'),
        ('FAKE', 'Fake'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    news_text = models.TextField()
    source_url = models.URLField(blank=True, null=True)
    prediction = models.CharField(max_length=10, choices=PREDICTION_CHOICES)
    confidence_score = models.FloatField()
    submitted_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.prediction} - {self.submitted_at.strftime('%Y-%m-%d')}"

    class Meta:
        ordering = ['-submitted_at']