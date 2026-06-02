from django.shortcuts import render
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.models import User
from detector.models import NewsSubmission
from django.db.models import Count, Avg
from django.utils import timezone
from datetime import timedelta
import json

def is_admin(user):
    return user.is_staff

@login_required
@user_passes_test(is_admin)
def admin_dashboard(request):
    total_submissions = NewsSubmission.objects.count()
    total_users       = User.objects.count()
    fake_count        = NewsSubmission.objects.filter(prediction='FAKE').count()
    real_count        = NewsSubmission.objects.filter(prediction='REAL').count()
    uncertain_count   = NewsSubmission.objects.filter(prediction='UNCERTAIN').count()

    seven_days_ago = timezone.now() - timedelta(days=7)
    recent_submissions = NewsSubmission.objects.filter(
        submitted_at__gte=seven_days_ago
    ).count()

    # Averages
    averages = NewsSubmission.objects.aggregate(
        avg_confidence=Avg('confidence_score'),
        avg_uncertainty=Avg('uncertainty_score'),
    )
    avg_confidence  = round((averages['avg_confidence']  or 0) * 100, 1)
    avg_uncertainty = round((averages['avg_uncertainty'] or 0) * 100, 1)

    top_users = NewsSubmission.objects.values('user__username').annotate(
        count=Count('id')
    ).order_by('-count')[:10]

    all_submissions = NewsSubmission.objects.select_related('user').all()[:50]

    # Chart data — last 7 days
    chart_labels, chart_data = [], []
    for i in range(6, -1, -1):
        day = timezone.now() - timedelta(days=i)
        chart_labels.append(day.strftime('%b %d'))
        chart_data.append(
            NewsSubmission.objects.filter(submitted_at__date=day.date()).count()
        )

    context = {
        'total_submissions': total_submissions,
        'total_users':       total_users,
        'fake_count':        fake_count,
        'real_count':        real_count,
        'uncertain_count':   uncertain_count,
        'recent_submissions': recent_submissions,
        'avg_confidence':    avg_confidence,
        'avg_uncertainty':   avg_uncertainty,
        'top_users':         top_users,
        'all_submissions':   all_submissions,
        'fake_percentage':   round((fake_count / total_submissions * 100), 1) if total_submissions else 0,
        'real_percentage':   round((real_count / total_submissions * 100), 1) if total_submissions else 0,
        'chart_labels':      json.dumps(chart_labels),
        'chart_data':        json.dumps(chart_data),
        'doughnut_labels':   json.dumps(['Real', 'Fake', 'Uncertain']),
        'doughnut_data':     json.dumps([real_count, fake_count, uncertain_count]),
    }
    return render(request, 'dashboard/admin_dashboard.html', context)