import os
import html

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
from django.conf import settings
from twilio.rest import Client


def get_client():
    return Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)


def _escape(text: str) -> str:
    """
    Escape text for safe inclusion inside TwiML XML.
    Prevents '&', '<', '>' and quotes from breaking the response and causing
    Twilio to say 'Sorry, an application error has occurred'.
    """
    if text is None:
        return ""
    return html.escape(str(text), quote=True)


def initiate_call(to_number: str, session_id: str) -> str:
    client = get_client()
    call = client.calls.create(
        to=to_number,
        from_=settings.TWILIO_PHONE_NUMBER,
        url=f"{settings.PUBLIC_BASE_URL}/calls/answer/?session_id={session_id}",
        method="POST",
        status_callback=f"{settings.PUBLIC_BASE_URL}/calls/status/",
        status_callback_method="POST",
        status_callback_event=["completed", "failed", "busy", "no-answer"],
        timeout=30,
    )
    return call.sid


def twiml_gather(prompt: str, session_id: str) -> str:
    """
    Build TwiML that gathers user speech and forwards it back to our /calls/respond/ endpoint.
    The prompt is XML-escaped so any characters from the LLM cannot break TwiML.
    """
    url = f"{settings.PUBLIC_BASE_URL}/calls/respond/"
    safe_prompt = _escape(prompt)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Gather input="speech" action="{url}?session_id={session_id}" method="POST" speechTimeout="auto" language="en-US">
    <Say voice="Polly.Joanna">{safe_prompt}</Say>
  </Gather>
  <Say voice="Polly.Joanna">I didn't catch that. Goodbye!</Say>
  <Hangup/>
</Response>"""


def twiml_hangup(text: str) -> str:
    """
    Build a TwiML hangup response with escaped text so Twilio always receives valid XML.
    """
    safe_text = _escape(text)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="Polly.Joanna">{safe_text}</Say>
  <Hangup/>
</Response>"""
