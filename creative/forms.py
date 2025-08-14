from django import forms


class PromptForm(forms.Form):
    prompt = forms.CharField(
        label="Your prompt",
        widget=forms.Textarea(attrs={"rows": 4, "placeholder": "Write a seed for the short story..."}),
        required=False,
        help_text="Leave empty to use a default creative seed.",
    )
    audio = forms.FileField(
        label="Optional audio (WAV/MP3)",
        required=False,
    )


