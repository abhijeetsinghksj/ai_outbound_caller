#!/usr/bin/env python3
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
import django
django.setup()
from django.conf import settings
from colorama import Fore, Style, init as colorama_init
from calls.models import CallSession
from calls.twilio_service import initiate_call
from knowledge_base.kb_service import build_index
colorama_init(autoreset=True)

MODEL_OPTIONS = list(settings.MODEL_CONFIGS.keys())

def select_model():
    print(f"\n{Fore.YELLOW}Available Models:{Style.RESET_ALL}")
    for i, k in enumerate(MODEL_OPTIONS, 1):
        cfg = settings.MODEL_CONFIGS[k]
        print(f"  [{i}] {cfg['display_name']} ({cfg['model_id']})")
    c = input(f"\n{Fore.CYAN}Select [1-{len(MODEL_OPTIONS)}] or Enter for default: {Style.RESET_ALL}").strip()
    try:
        return MODEL_OPTIONS[int(c) - 1]
    except:
        return settings.ACTIVE_MODEL

def validate_number(raw):
    n = raw.strip()
    if not n.startswith("+"):
        n = "+" + n
    if not n[1:].isdigit() or len(n) < 8:
        raise ValueError(f"Invalid: {n}. Use E.164 e.g. +919876543210")
    return n

def create_session(to_number, model_key):
    cfg = settings.MODEL_CONFIGS[model_key]
    s = CallSession(
        to_number=to_number,
        from_number=settings.TWILIO_PHONE_NUMBER,
        model_key=model_key,
        model_id=cfg["model_id"],
        model_provider=cfg["provider"],
        status="initiated",
    )
    s.save()
    return s

def watch(session_id, poll=4.0, timeout=600):
    print(f"\n{Fore.YELLOW}Monitoring call... (Ctrl+C to stop){Style.RESET_ALL}\n")
    deadline, last = time.time() + timeout, 0
    while time.time() < deadline:
        try:
            s = CallSession.objects.get(id=session_id)
        except:
            time.sleep(poll)
            continue
        if len(s.turns) > last:
            for t in s.turns[last:]:
                print(f"{Fore.BLUE}{'─'*50}{Style.RESET_ALL}")
                print(f"  {Fore.GREEN}Turn {t.turn_index + 1}{Style.RESET_ALL}")
                print(f"  User : {t.user_input}")
                print(f"  AI   : {t.ai_response}")
                print(f"  {Fore.MAGENTA}LLM latency  : {t.llm_latency_ms:.0f} ms{Style.RESET_ALL}")
                print(f"  {Fore.MAGENTA}Total latency: {t.total_latency_ms:.0f} ms{Style.RESET_ALL}")
                print(f"  {Fore.MAGENTA}Tokens       : {t.total_tokens}{Style.RESET_ALL}")
                if t.hallucination_score is not None:
                    c = Fore.RED if t.hallucination_score > 0.5 else Fore.GREEN
                    print(f"  {c}Hallucination: {t.hallucination_score:.3f} (0=grounded 1=hallucinated){Style.RESET_ALL}")
                if t.faithfulness_score is not None:
                    print(f"  {Fore.GREEN}Faithfulness : {t.faithfulness_score:.3f}{Style.RESET_ALL}")
                if t.evaluation_notes:
                    print(f"  Notes        : {t.evaluation_notes}")
                print()
            last = len(s.turns)
        if s.status in ("completed", "failed", "busy", "no-answer"):
            print(f"\n{Fore.CYAN}CALL ENDED — Status: {s.status} | Turns: {s.total_turns} | Duration: {s.call_duration_s:.0f}s{Style.RESET_ALL}")
            if s.avg_hallucination is not None:
                print(f"  Avg Hallucination: {s.avg_hallucination:.3f}")
            if s.avg_faithfulness is not None:
                print(f"  Avg Faithfulness : {s.avg_faithfulness:.3f}")
            return
        time.sleep(poll)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--number", "-n")
    parser.add_argument("--model", "-m")
    args = parser.parse_args()

    print(f"\n{Fore.CYAN}{'='*52}")
    print(f"  AI Outbound Caller — Django + MongoDB + Twilio")
    print(f"{'='*52}{Style.RESET_ALL}\n")

    print(f"{Fore.YELLOW}Loading knowledge base...{Style.RESET_ALL}")
    build_index()

    raw = args.number or input(f"{Fore.CYAN}Phone number (e.g. +919876543210): {Style.RESET_ALL}").strip()
    try:
        to_number = validate_number(raw)
    except ValueError as e:
        print(f"{Fore.RED}{e}{Style.RESET_ALL}")
        sys.exit(1)

    model_key = (args.model if args.model in MODEL_OPTIONS else None) or select_model()
    cfg = settings.MODEL_CONFIGS[model_key]
    print(f"\n{Fore.GREEN}Model: {cfg['display_name']} ({cfg['model_id']}){Style.RESET_ALL}")

    confirm = input(f"\n{Fore.YELLOW}Call {to_number} now? [y/N]: {Style.RESET_ALL}").strip().lower()
    if confirm != "y":
        print("Aborted.")
        sys.exit(0)

    session = create_session(to_number, model_key)
    print(f"{Fore.GREEN}Session ID: {session.id}{Style.RESET_ALL}")

    print(f"{Fore.CYAN}Placing call...{Style.RESET_ALL}")
    try:
        sid = initiate_call(to_number, str(session.id))
        session.call_sid = sid
        session.save()
        print(f"{Fore.GREEN}Call placed! SID: {sid}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Failed: {e}{Style.RESET_ALL}")
        sys.exit(1)

    try:
        watch(str(session.id))
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Stopped. Session: {session.id}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
