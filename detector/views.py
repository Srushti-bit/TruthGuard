from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
import requests
from bs4 import BeautifulSoup
from .models import NewsSubmission

def scrape_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        return text[:5000]
    except Exception:
        return None

@login_required
def submit_news(request):
    if request.method == 'POST':
        news_text = request.POST.get('news_text', '').strip()
        source_url = request.POST.get('source_url', '').strip()

        if source_url and not news_text:
            news_text = scrape_url(source_url)
            if not news_text:
                messages.error(request, 'Could not extract text from that URL. Please paste the text manually.')
                return render(request, 'detector/submit.html')

        if not news_text or len(news_text) < 50:
            messages.error(request, 'Please provide at least 50 characters of news text.')
            return render(request, 'detector/submit.html')

        from ml_model.predictor import predict_news
        result = predict_news(news_text)

        submission = NewsSubmission.objects.create(
            user=request.user,
            news_text=news_text,
            source_url=source_url if source_url else None,
            prediction=result['prediction'],
            confidence_score=result['confidence'],
        )

        return render(request, 'detector/result.html', {
            'submission': submission,
            'result': result,
            'news_preview': news_text[:500],
        })

    return render(request, 'detector/submit.html')

@login_required
def history_view(request):
    submissions = NewsSubmission.objects.filter(user=request.user)
    return render(request, 'detector/history.html', {'submissions': submissions})