from django.db import models
from django.contrib.auth.models import User

class NewsSubmission(models.Model):
    PREDICTION_CHOICES = [
        ('REAL', 'Real'),
        ('FAKE', 'Fake'),
        ('UNCERTAIN', 'Uncertain'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    news_text = models.TextField()
    source_url = models.URLField(blank=True, null=True)
    
    # Veracity Metrics
    prediction = models.CharField(max_length=15, choices=PREDICTION_CHOICES)
    confidence_score = models.FloatField()  # e.g., 0.9420
    uncertainty_score = models.FloatField() # From MC Dropout passes
    
    # Hybrid Model Components
    bert_semantic_score = models.FloatField(null=True, blank=True)
    gcn_propagation_score = models.FloatField(null=True, blank=True)
    
    shares_analyzed = models.IntegerField(default=500)
    submitted_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.prediction}"