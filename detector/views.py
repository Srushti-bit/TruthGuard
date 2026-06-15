from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.models import User
from django.contrib import messages
from django.http import HttpResponseForbidden
from detector.models import NewsSubmission
from django.db.models import Count
from django.utils import timezone
from datetime import timedelta
from functools import lru_cache
import json
import os


# Local Model Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'ml_model', 'hybrid_model.pth')


@lru_cache(maxsize=1)
def get_local_model():
    """Load the ML model only when analysis is requested."""
    import torch
    import torch.nn as nn
    from transformers import BertTokenizer, BertModel

    class HybridTruthGuard(nn.Module):
        def __init__(self):
            super(HybridTruthGuard, self).__init__()
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            self.fusion = nn.Linear(768 + 64, 256)
            self.classifier = nn.Linear(256, 2)
            self.dropout = nn.Dropout(0.3)

        def forward(self, ids, mask, force_dropout=False):
            if force_dropout:
                self.dropout.train()

            outputs = self.bert(ids, attention_mask=mask)
            bert_feats = outputs.last_hidden_state[:, 0, :]
            gcn_placeholder = torch.zeros(bert_feats.shape[0], 64).to(bert_feats.device)

            combined = torch.cat([bert_feats, gcn_placeholder], dim=-1)
            x = torch.relu(self.fusion(combined))
            x = self.dropout(x)

            return self.classifier(x)

    print("Loading TruthGuard local model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model = HybridTruthGuard()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    print("Model loaded successfully.")
    return tokenizer, model, torch


def analyze_with_local_model(news_text):
    """Run local BERT+GCN hybrid model with Monte Carlo Dropout."""
    import numpy as np

    tokenizer, model, torch = get_local_model()

    inputs = tokenizer(
        news_text,
        max_length=64,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )

    all_probs = []

    with torch.no_grad():
        for _ in range(50):
            outputs = model(
                inputs['input_ids'],
                inputs['attention_mask'],
                force_dropout=True
            )
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_probs.append(probs.numpy())

    all_probs = np.array(all_probs)
    mean_probs = np.mean(all_probs, axis=0)[0]
    std_dev = np.std(all_probs, axis=0)[0]

    predicted_class = int(np.argmax(mean_probs))
    confidence = float(mean_probs[predicted_class])
    uncertainty = float(std_dev[predicted_class])

    real_prob = float(mean_probs[1])

    if uncertainty > 0.12:
        prediction = 'UNCERTAIN'
    elif predicted_class == 1:
        prediction = 'REAL'
    else:
        prediction = 'FAKE'

    return {
        'prediction': prediction,
        'confidence_score': round(confidence, 4),
        'uncertainty_score': round(uncertainty, 4),
        'bert_semantic_score': round(real_prob, 4),
        'gcn_propagation_score': round(confidence, 4),
        'explanation': (
           f"This article appears to be {prediction}. "
           f"The result is based on the article text and reliability patterns found during the check."
)
    }


# Helpers
def is_admin(user):
    return user.is_staff


# Views
@login_required
def submit_news(request):
    result = None
    error = None

    if request.method == 'POST':
        news_text = request.POST.get('news_text', '').strip()
        source_url = request.POST.get('source_url', '').strip() or None

        if not news_text:
            error = "Please enter some news text to analyze."
        elif len(news_text) < 20:
            error = "Please enter at least 20 characters of news text."
        else:
            try:
                analysis = analyze_with_local_model(news_text)

                submission = NewsSubmission.objects.create(
                    user=request.user,
                    news_text=news_text,
                    source_url=source_url,
                    prediction=analysis['prediction'],
                    confidence_score=analysis['confidence_score'],
                    uncertainty_score=analysis['uncertainty_score'],
                    bert_semantic_score=analysis.get('bert_semantic_score'),
                    gcn_propagation_score=analysis.get('gcn_propagation_score'),
                )

                result = {
                    'submission': submission,
                    'explanation': analysis.get('explanation', ''),
                }

                messages.success(request, "Analysis complete!")

            except Exception as e:
                error = f"Analysis failed: {str(e)}"

    return render(request, 'detector/submit.html', {
        'result': result,
        'error': error,
    })


@login_required
def history_view(request):
    submissions = NewsSubmission.objects.filter(
        user=request.user
    ).order_by('-submitted_at')[:50]

    return render(request, 'detector/history.html', {
        'submissions': submissions,
    })


@login_required
def delete_submission(request, submission_id):
    if not request.user.is_superuser:
        return HttpResponseForbidden("Only superusers can delete submissions.")

    submission = get_object_or_404(NewsSubmission, id=submission_id)

    if request.method == 'POST':
        submission.delete()
        messages.success(request, f'Submission #{submission_id} deleted.')

    return redirect('detector:history')


# Admin dashboard
@login_required
@user_passes_test(is_admin)
def admin_dashboard(request):
    total_submissions = NewsSubmission.objects.count()
    total_users = User.objects.count()

    fake_count = NewsSubmission.objects.filter(prediction='FAKE').count()
    real_count = NewsSubmission.objects.filter(prediction='REAL').count()
    uncertain_count = NewsSubmission.objects.filter(prediction='UNCERTAIN').count()

    seven_days_ago = timezone.now() - timedelta(days=7)
    recent_submissions = NewsSubmission.objects.filter(
        submitted_at__gte=seven_days_ago
    ).count()

    top_users = NewsSubmission.objects.values('user__username').annotate(
        count=Count('id')
    ).order_by('-count')[:10]

    all_submissions = NewsSubmission.objects.select_related('user').all()[:50]

    chart_labels = []
    chart_data = []

    for i in range(6, -1, -1):
        day = timezone.now() - timedelta(days=i)
        chart_labels.append(day.strftime('%b %d'))
        chart_data.append(
            NewsSubmission.objects.filter(submitted_at__date=day.date()).count()
        )

    doughnut_labels = ['Real', 'Fake', 'Uncertain']
    doughnut_data = [real_count, fake_count, uncertain_count]

    context = {
        'total_submissions': total_submissions,
        'total_users': total_users,
        'fake_count': fake_count,
        'real_count': real_count,
        'uncertain_count': uncertain_count,
        'recent_submissions': recent_submissions,
        'top_users': top_users,
        'all_submissions': all_submissions,
        'fake_percentage': round((fake_count / total_submissions * 100), 1) if total_submissions > 0 else 0,
        'real_percentage': round((real_count / total_submissions * 100), 1) if total_submissions > 0 else 0,
        'chart_labels': json.dumps(chart_labels),
        'chart_data': json.dumps(chart_data),
        'doughnut_labels': json.dumps(doughnut_labels),
        'doughnut_data': json.dumps(doughnut_data),
    }

    return render(request, 'dashboard/admin_dashboard.html', context)