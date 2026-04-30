from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.models import User
from django.contrib import messages
from django.http import HttpResponseForbidden
from detector.models import NewsSubmission
from django.db.models import Count
from django.utils import timezone
from datetime import timedelta
import json
import os
import re
import google.generativeai as genai


# ── Gemini client ──────────────────────────────────────────────
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

ANALYSIS_PROMPT = """You are a misinformation detection AI. Analyze the following news text and determine if it is REAL, FAKE, or UNCERTAIN.

Respond ONLY in this exact JSON format, no extra text, no markdown fences:
{
  "prediction": "REAL" or "FAKE" or "UNCERTAIN",
  "confidence_score": <float 0.0-1.0>,
  "uncertainty_score": <float 0.0-1.0>,
  "bert_semantic_score": <float 0.0-1.0>,
  "gcn_propagation_score": <float 0.0-1.0>,
  "explanation": "<2-3 sentence plain English explanation>"
}

Rules:
- confidence_score: how confident you are in the prediction
- uncertainty_score: how uncertain/ambiguous the content is (higher = more uncertain)
- bert_semantic_score: semantic credibility score based on language patterns
- gcn_propagation_score: estimated virality/propagation risk score
- Be calibrated and honest. Satire/clickbait = FAKE. Verifiable facts = REAL. Ambiguous = UNCERTAIN.
"""


# ── Helpers ───────────────────────────────────────────────────
def is_admin(user):
    return user.is_staff


def analyze_news_with_ai(news_text):
    prompt = f"{ANALYSIS_PROMPT}\n\nNEWS TEXT:\n{news_text}"
    response = gemini_model.generate_content(prompt)
    raw = response.text.strip()
    # Strip accidental markdown fences
    raw = re.sub(r'^```json\s*|^```\s*|```$', '', raw, flags=re.MULTILINE).strip()
    return json.loads(raw)


# ── Views ─────────────────────────────────────────────────────

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
                analysis = analyze_news_with_ai(news_text)

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

            except json.JSONDecodeError:
                error = "AI returned an unexpected response. Please try again."
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


# ── Admin dashboard ───────────────────────────────────────────

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

    chart_labels, chart_data = [], []
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