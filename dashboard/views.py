from django.shortcuts import render
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.models import User
from detector.models import NewsSubmission
from django.db.models import Count
from django.utils import timezone
from datetime import timedelta

def is_admin(user):
    return user.is_staff

@login_required
@user_passes_test(is_admin)
def admin_dashboard(request):
    total_submissions = NewsSubmission.objects.count()
    total_users = User.objects.count()
    fake_count = NewsSubmission.objects.filter(prediction='FAKE').count()
    real_count = NewsSubmission.objects.filter(prediction='REAL').count()

    seven_days_ago = timezone.now() - timedelta(days=7)
    recent_submissions = NewsSubmission.objects.filter(
        submitted_at__gte=seven_days_ago
    ).count()

    top_users = NewsSubmission.objects.values('user__username').annotate(
        count=Count('id')
    ).order_by('-count')[:10]

    all_submissions = NewsSubmission.objects.select_related('user').all()[:50]

    context = {
        'total_submissions': total_submissions,
        'total_users': total_users,
        'fake_count': fake_count,
        'real_count': real_count,
        'recent_submissions': recent_submissions,
        'top_users': top_users,
        'all_submissions': all_submissions,
        'fake_percentage': round((fake_count / total_submissions * 100), 1) if total_submissions > 0 else 0,
        'real_percentage': round((real_count / total_submissions * 100), 1) if total_submissions > 0 else 0,
    }
    return render(request, 'dashboard/admin_dashboard.html', context)