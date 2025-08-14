from pathlib import Path
from django.shortcuts import render
from django.conf import settings
from .forms import PromptForm
from .pipeline import orchestrate


def home(request):
    context = {}
    if request.method == "POST":
        form = PromptForm(request.POST, request.FILES)
        if form.is_valid():
            prompt = form.cleaned_data.get("prompt")
            media_dir = Path(settings.MEDIA_ROOT)
            media_dir.mkdir(parents=True, exist_ok=True)
            output = orchestrate(prompt, media_dir)
            context.update({
                "form": form,
                "story": output.story,
                "character_description": output.character_description,
                "background_description": output.background_description,
                "combined_image_url": settings.MEDIA_URL + Path(output.combined_image_path).name,
            })
        else:
            context["form"] = form
    else:
        context["form"] = PromptForm()
    return render(request, "creative/home.html", context)
